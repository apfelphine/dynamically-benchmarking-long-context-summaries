from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import os
import json

import aiofiles
from colorama import Fore

from core.document_alignment.batched_llm_aligner import BatchedLLMAligner
from core.document_alignment.hybrid_aligner import HybridDocumentAligner
from core.document_splitting.docling.docling_splitter import DoclingParagraphChunker
from core.embedding.openai_embedding import OpenAIEmbedding
from core.llms.openai_llm import OpenAILLM
from models.corpus import Bin, Summary
from core.llms.ollama_llm import OllamaLLM
from core.llms.azure_llm import AzureLLM
from core.summarization.single_pass_summarizer import SinglePassSummarizer
from core.document_alignment.embedding_aligner import EmbeddingAligner
from experiments.configuration import base_data_dir, base_experiment_dir

class ExperimentContext(ABC):
    """
    Abstract context holding configuration, resources, and data for an experiment.
    """
    base_experiment_dir = base_experiment_dir
    base_data_dir = base_data_dir

    def __init__(self, name: str, allowed_bins: Optional[List[int]] = None):
        self.name = name

        # Lazy loaded properties
        self.__llms = None
        self._summarizers = None
        self.__comparer = None
        self.__chunker = None

        print(f"{Fore.WHITE}Loading Corpus and Summaries...")

        # Load Corpus
        with open(os.path.join(self.base_experiment_dir, "corpus.json"), "r") as f:
            self.corpus = [Bin(**b) for b in json.load(f)]

        # Load Summaries
        self.summaries = self.load_summaries()
        if allowed_bins is not None:
            self.summaries = [s for s in self.summaries if s.bin_idx in allowed_bins]

    @property
    def base_summaries_dir(self):
        return os.path.join(self.base_experiment_dir, "summaries", self.name)

    @property
    def base_evaluations_dir(self):
        return os.path.join(self.base_experiment_dir, "evaluation", self.name)

    @property
    def llms(self):
        if not self.__llms:
            self.__initialize_llms()
        return self.__llms

    @property
    def summarizers(self):
        if not self._summarizers:
            self._initialize_summarizers()
        return self._summarizers

    @property
    def comparer(self):
        if not self.__comparer:
            self.__initialize_comparer()
        return self.__comparer

    @property
    def chunker(self):
        if not self.__chunker:
            self.__initialize_chunker()
        return self.__chunker

    def __initialize_llms(self):
        self.__llms = {
            "qwen3-4b-instruct-2507-q8_0": OllamaLLM(
                model="qwen3:4b-instruct-2507-q8_0",
                num_ctx=128_000,
                num_gpu=37,
                tokenizer_name="Qwen/Qwen3-4B-Instruct-2507",
            ),
            "gpt-4.1-mini": AzureLLM(
                model="gpt-4.1-mini",
                max_tokens_per_minute=14_900 * 20,
                max_requests_per_minute=60 * 5,
            ),
            "gpt-4.1": AzureLLM(
                model="gpt-4.1",
                max_tokens_per_minute=3_000 * 70,
                max_requests_per_minute=60 * 5,
            ),
            "gemma-4-E4B-it-UD-Q8_K_XL": OpenAILLM(
                model="workstation/gpu/gemma-4-E4B-it-UD-Q8_K_XL",
                tokenizer_name="unsloth/gemma-4-E4B-it",
                base_url=os.environ.get("LLAMASWAP_HOST"),
                api_key=None,
                ctx_window=131072,
            ),
            "qwen3_5-4B-UD-Q8_K_XL": OpenAILLM(
                model="workstation/gpu/qwen3_5-4B-UD-Q8_K_XL",
                tokenizer_name="unsloth/Qwen3.5-4B",
                base_url=os.environ.get("LLAMASWAP_HOST"),
                api_key=None,
                ctx_window=131072,
            ),
            "ministral-3-3B-instruct-UD-Q8_K_XL": OpenAILLM(
                model="workstation/gpu/ministral-3-3B-instruct-UD-Q8_K_XL",
                tokenizer_name="unsloth/Ministral-3-3B-Instruct-2512",
                base_url=os.environ.get("LLAMASWAP_HOST"),
                api_key=None,
                ctx_window=131072,
            )
        }

    def _initialize_summarizers(self):
        self._summarizers = {
            "single-pass": lambda llm: SinglePassSummarizer(llm),
        }

    def __initialize_comparer(self):
        self.__comparer = HybridDocumentAligner(
            EmbeddingAligner(
                similarity="cosine",
                embedding=OpenAIEmbedding(
                    model="workstation/gpu/qwen3-embedding-8B-Q8_0",
                    base_url=os.environ.get("LLAMASWAP_HOST"),
                ),
                split_summary=True,
            ),
            BatchedLLMAligner(
                llm=OpenAILLM(
                    "workstation/gpu/qwen3_6-35B-A3B-UD-IQ4_NL-32768ctx-6p",
                    ctx_window=65535,
                    base_url=os.environ.get("LLAMASWAP_HOST"),
                    tokenizer_name="Qwen/Qwen3.6-35B-A3B",
                    timeout=360
                ),
                batch_size=8,
                all_summary_chunks=False,
            ),
            config={"threshold": 0.45},
        )

    def __initialize_chunker(self):
        self.__chunker = DoclingParagraphChunker(
            merge_peers=True,
            contextualize=True,
            tokenizer=DoclingParagraphChunker.get_openai_tokenizer(
                "cl100k_base",
                4096
            ),
        )

    @abstractmethod
    def load_summaries(self) -> List[Summary]:
        """Load the list of summaries to process."""
        pass

    @abstractmethod
    def get_save_path(self, summary: Summary) -> str:
        """Determine the root directory where a specific summary output should be saved."""
        pass

    @abstractmethod
    async def read_files(
        self, b: Bin, summary: Summary
    ) -> Tuple[List[str], Optional[str]]:
        """
        Read the source text files for a specific summary.
        Returns: (List of text content, Optional focus document title)
        """
        pass

    async def get_documents_for_summary(self, document_order: List[int], doc_paths: List[str]):
        documents = []
        for doc_idx in document_order:
            f_path = os.path.join(self.base_data_dir, doc_paths[doc_idx], "document.md")
            if os.path.exists(f_path):
                async with aiofiles.open(f_path, "r", encoding="utf-8") as f:
                    documents.append(await f.read())
            else:
                raise Exception(f"Could not find document: {f_path}")
        return documents