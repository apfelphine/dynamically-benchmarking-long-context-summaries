import os
from abc import ABC
from collections import defaultdict
from typing import Dict

from experiments.experiment_evaluator import ExperimentEvaluator
from experiments.generic_summaries.context import GenericSummaryContext
from models.corpus import GenericSummary, Document
from models.document_splitting import Chunk
from utils.async_json import json_load


class GenericSummaryBaseEvaluator(ExperimentEvaluator, ABC):
    def __init__(self, context: GenericSummaryContext, base_eval_dir: str):
        super().__init__(context, base_eval_dir)

    async def prepare_data(self):
        # Pre-load chunks and document metadata
        chunks_per_bin = defaultdict(lambda: defaultdict(list))
        chunk_tokens_per_bin = defaultdict(lambda: defaultdict(list))
        docs_per_bin = defaultdict(list)
        for idx, b in enumerate(self.context.corpus):
            for doc_p in b.document_paths:
                docs_per_bin[idx].append(
                    Document(
                        **await json_load(
                            os.path.join(self.context.base_data_dir, doc_p, "metadata.json")
                        )
                    )
                )

            chunks = [
                Chunk(**c)
                for c in await json_load(
                    os.path.join(
                        self.context.base_summaries_dir, str(idx), "chunks.json"
                    )
                )
            ]
            for c in chunks:
                chunks_per_bin[idx][c.document_idx].append(c)
                chunk_tokens_per_bin[idx][c.document_idx].append(
                    len(self.ENC.encode(c.text))
                )

        return chunks_per_bin, chunk_tokens_per_bin, docs_per_bin

    @staticmethod
    def get_ordered_data(prepared_data: Dict, summary: GenericSummary):
        bin_idx = summary.bin_idx
        chunks_per_bin, chunk_tokens_per_bin, docs_per_bin = prepared_data
        chunks = chunks_per_bin[bin_idx]
        chunk_tokens = chunk_tokens_per_bin[bin_idx]
        documents = docs_per_bin[bin_idx]

        ordered_chunks = []
        ordered_chunk_tokens = []
        ordered_doc_tokens = []
        for d_idx in summary.document_order:
            ordered_doc_tokens.append(documents[d_idx].document.tokens)
            ordered_chunks.extend(chunks[d_idx])
            ordered_chunk_tokens.extend(chunk_tokens[d_idx])

        return ordered_chunks, ordered_chunk_tokens, ordered_doc_tokens
