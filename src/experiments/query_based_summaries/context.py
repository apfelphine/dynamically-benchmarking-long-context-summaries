import json
import os
from typing import Optional, List, Tuple

import aiofiles
import numpy as np
import tiktoken

from core.document_splitting.docling.docling_splitter import DoclingParagraphChunker
from experiments.base_experiment_context import ExperimentContext
from models.corpus import Summary, QueryBasedSummary, Bin, Document
from models.document_splitting import Chunk
from utils.async_json import json_load, json_dump


class QueryBasedSummaryContext(ExperimentContext):
    def __init__(self, allowed_bins: Optional[List[int]] = None):
        super().__init__("query_based_summaries", allowed_bins)

        # Initialize Tokenizer and Chunker
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.partition_chunker = DoclingParagraphChunker(
            merge_peers=False,
            contextualize=True,
            tokenizer=DoclingParagraphChunker.get_openai_tokenizer("cl100k_base", 8192),
        )

        self.graphs_path = os.path.join(self.base_evaluations_dir, "wiki_linked_pages")

        self.summaries: List[QueryBasedSummary] = self.summaries

    def load_summaries(self) -> List[Summary]:
        """
        Loads QueryBasedSummary objects from query_based_summaries.json.
        """
        summary_path = os.path.join(self.base_experiment_dir, "query_based_summaries.json")
        with open(summary_path, "r") as f:
            query_based_summaries: List[QueryBasedSummary] = [
                QueryBasedSummary(**s) for s in json.load(f)
            ]
        return query_based_summaries

    def get_save_path(self, summary: Summary) -> str:
        """
        Determines save path based on whether the experiment is 'partitioned' or 'positioned'.
        """
        if not isinstance(summary, QueryBasedSummary):
            raise ValueError("Summary must be a QueryBasedSummary for experiment 2")

        summary_dir = os.path.join(
            self.base_summaries_dir,
            str(summary.bin_idx),
            str(summary.focus_doc_idx),
        )

        chunk_order = summary.chunk_order
        if chunk_order is not None:
            return os.path.join(summary_dir, "partitioned", str(chunk_order[0]))
        else:
            # If chunk_order is None, document_order includes the focus doc
            focus_position = summary.document_order.index(summary.focus_doc_idx)
            return os.path.join(summary_dir, "positioned", str(focus_position))

    async def read_files(
        self, b: Bin, summary: Summary
    ) -> Tuple[List[str], Optional[str]]:
        if not isinstance(summary, QueryBasedSummary):
            raise ValueError("Summary must be a QueryBasedSummary for experiment 2")

        doc_paths = b.document_paths
        focus_document_idx = summary.focus_doc_idx
        document_order = summary.document_order
        chunk_order = summary.chunk_order

        summary_dir = self.get_save_path(summary)

        # Get Focus Document Title (used for LLM context)
        doc_meta_file = os.path.join(
            self.base_data_dir, doc_paths[focus_document_idx], "metadata.json"
        )
        if os.path.exists(doc_meta_file):
            focus_title = Document(**await json_load(doc_meta_file)).title
        else:
            raise Exception(f"Could not find metadata file: {doc_meta_file}")

        documents_file = os.path.join(summary_dir, "documents.json")
        os.makedirs(summary_dir, exist_ok=True)

        if os.path.exists(documents_file):
            return await json_load(documents_file), focus_title

        documents = await self.get_documents_for_summary(document_order, doc_paths)

        # Handle Partitioning Logic (If applicable)
        if chunk_order:
            focus_doc_dir = os.path.join(
                self.base_summaries_dir,
                str(summary.bin_idx),
                str(focus_document_idx),
            )
            focus_chunk_path = os.path.join(
                focus_doc_dir, "partitioned", "focus_chunks.json"
            )
            os.makedirs(os.path.dirname(focus_chunk_path), exist_ok=True)

            # Get or Create Partitions
            if os.path.exists(focus_chunk_path):
                partition_focus_chunks: List[str] = await json_load(focus_chunk_path)
            else:
                # Read original focus document text
                f_path = os.path.join(
                    self.base_data_dir, doc_paths[focus_document_idx], "document.md"
                )
                if os.path.exists(f_path):
                    async with aiofiles.open(f_path, "r", encoding="utf-8") as f:
                        focus_doc_text = await f.read()
                else:
                    raise Exception(f"Could not find focus document: {f_path}")

                # Chunk and Partition
                focus_chunks = await self.partition_chunker.split(focus_doc_text)
                n_partitions = len(chunk_order)
                partition_focus_chunks: List[str] = []

                # Group chunks into partitions roughly equal in token size
                for p in self._partition_chunks(focus_chunks, n_partitions):
                    partition_focus_chunks.append(self._merge_chunks(p))

                await json_dump(focus_chunk_path, partition_focus_chunks)

            # Order Partitions
            ordered_partitions = []
            for part_idx in chunk_order:
                ordered_partitions.append(partition_focus_chunks[part_idx])

            documents = self._distribute_pinned(documents, ordered_partitions)

        await json_dump(documents_file, documents)
        return documents, focus_title

    @staticmethod
    def _distribute_pinned(
        remaining_documents: List[str], focus_chunks: List[str]
    ) -> List[str]:
        """
        Distributes focus chunks evenly among remaining documents.
        First focus chunk is always pinned to the start.
        """
        if not focus_chunks:
            return remaining_documents

        if len(focus_chunks) == 1:
            return [focus_chunks[0]] + remaining_documents

        result = [focus_chunks[0]]
        step = len(remaining_documents) / (len(focus_chunks) - 1)

        for i in range(1, len(focus_chunks)):
            start_idx = int((i - 1) * step)
            end_idx = int(i * step)
            result.extend(remaining_documents[start_idx:end_idx])
            result.append(focus_chunks[i])

        # Ensure any remaining docs are appended (floating point safeguards)
        # Though the logic usually covers it, strictly speaking:
        last_end_idx = int((len(focus_chunks) - 1) * step)
        if last_end_idx < len(remaining_documents):
            result.extend(remaining_documents[last_end_idx:])

        return result

    def _partition_chunks(
        self, chunks: List[Chunk], n_partitions: int
    ) -> List[List[Chunk]]:
        """
        Splits a list of Chunks into N partitions of roughly equal token size.
        """
        data = [len(self.enc.encode(c.text)) for c in chunks]
        data_arr = np.array(data)
        cumulative_sum = np.cumsum(data_arr)
        total_sum = cumulative_sum[-1]

        edges = np.linspace(0, total_sum, n_partitions + 1)
        part_indices = np.digitize(cumulative_sum, edges) - 1
        part_indices = np.clip(part_indices, 0, n_partitions - 1)

        partition_focus_chunk_list: List[List[Chunk]] = [
            [] for _ in range(n_partitions)
        ]
        for c, p_idx in zip(chunks, part_indices):
            partition_focus_chunk_list[p_idx].append(c)

        return partition_focus_chunk_list

    @staticmethod
    def _merge_chunks(chunks: List[Chunk]) -> str:
        """
        Merges chunks into a single string, cleaning up redundant headers.
        """
        text = "\n".join([c.text for c in chunks])
        clean_text = ""
        existing_headers = []

        for line in text.splitlines():
            if line.startswith("#"):
                if line not in existing_headers:
                    existing_headers.append(line)
                    clean_text += "\n" + line
            else:
                clean_text += "\n" + line

        return clean_text.strip()

    @staticmethod
    def get_focus_indices(n_remaining, n_focus_chunks):
        if n_focus_chunks == 0:
            return []
        step = n_remaining / (n_focus_chunks - 1)
        indices = []
        for i in range(n_focus_chunks):
            indices.append(i + int(i * step))
        return indices

    @staticmethod
    def get_document_order(summary: QueryBasedSummary):
        focus_indices = QueryBasedSummaryContext.get_focus_indices(
            len(summary.document_order),
            len(summary.chunk_order) if summary.chunk_order else 0,
        )

        new_doc_order = []
        c = 0
        for i in range(len(focus_indices) + len(summary.document_order)):
            if i in focus_indices:
                new_doc_order.append(summary.focus_doc_idx)
            else:
                new_doc_order.append(summary.document_order[c])
                c += 1
        return new_doc_order
