import json
import os
from typing import Optional, List, Tuple

from core.document_splitting.docling.docling_splitter import DoclingParagraphChunker
from core.summarization.hierarchical_merge_summarizer import HierarchicalMergingSummarizer
from core.summarization.incremental_update_summarizer import IncrementalUpdatingSummarizer
from experiments.base_experiment_context import ExperimentContext
from models.corpus import Bin, Summary, GenericSummary
from utils.async_json import json_load, json_dump


class GenericSummaryContext(ExperimentContext):
    def __init__(self, allowed_bins: Optional[List[int]] = None):
        super().__init__("generic_summaries", allowed_bins)

    def load_summaries(self) -> List[Summary]:
        summary_path = os.path.join(self.base_experiment_dir, "generic_summaries.json")
        with open(summary_path, "r") as f:
            generic_summaries: List[GenericSummary] = [GenericSummary(**s) for s in json.load(f)]

        return generic_summaries

    def get_save_path(self, summary: Summary) -> str:
        if not isinstance(summary, GenericSummary):
            pass

        return os.path.join(
            self.base_summaries_dir,
            str(summary.bin_idx),
            str(summary.document_order[0]),
        )

    async def read_files(
        self, b: Bin, summary: Summary
    ) -> Tuple[List[str], Optional[str]]:
        if not isinstance(summary, GenericSummary):
            raise ValueError("Summary must be an GenericSummary for experiment 1")

        save_path = self.get_save_path(summary)
        os.makedirs(save_path, exist_ok=True)

        documents_file = os.path.join(save_path, "documents.json")

        # Check Cache
        if os.path.exists(documents_file):
            documents = await json_load(documents_file)
            return documents, None

        doc_paths = b.document_paths
        document_order = summary.document_order
        documents = await self.get_documents_for_summary(document_order, doc_paths)

        await json_dump(documents_file, documents)

        return documents, None

    def _initialize_summarizers(self):
        super()._initialize_summarizers()

        # Additionally inject chunked summarizers
        chunked_summary_splitter = DoclingParagraphChunker(
            merge_peers=True,
            contextualize=True,
            tokenizer=DoclingParagraphChunker.get_openai_tokenizer("cl100k_base", 4096),
            min_tokens=2048,
        )
        self._summarizers["hierarchical-merge"] = lambda llm: HierarchicalMergingSummarizer(
            batch_size=16,
            splitter=chunked_summary_splitter,
            llm=llm,
        )
        self._summarizers["incremental-update"] =  lambda llm: IncrementalUpdatingSummarizer(
            splitter=chunked_summary_splitter,
            llm=llm,
            batch_size=2,
        )
