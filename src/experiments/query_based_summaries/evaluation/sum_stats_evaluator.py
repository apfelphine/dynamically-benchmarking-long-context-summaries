import asyncio
import os
from typing import List

import numpy as np
import pandas as pd
import tiktoken

from experiments.query_based_summaries.context import QueryBasedSummaryContext
from experiments.query_based_summaries.evaluation.base_evaluator import QueryBasedSummaryBaseEvaluator
from models.corpus import QueryBasedSummary
from models.document_splitting import Chunk
from utils.async_json import json_load
from utils.partitioning import aggregate_partition_usage
from utils.pretty_table import pretty_table


class SummaryStatsEvaluator(QueryBasedSummaryBaseEvaluator):
    def __init__(self, context: QueryBasedSummaryContext):
        super().__init__(context, "summary_statistics")
        self.ENC = tiktoken.get_encoding("cl100k_base")
        self.context: QueryBasedSummaryContext = context

    async def evaluate_single_summary(
        self,
        b,
        summary_idx,
        summary: QueryBasedSummary,
        prepared_data,
        summarizer_name,
        llm_name,
    ):
        ordered_chunks = prepared_data[summary_idx]
        summary_text, metadata, binary_sim_matrix, _ = await self._load_llm_data(
            summary, summarizer_name, llm_name
        )

        doc_usage = aggregate_partition_usage(
            len(b.document_paths),
            binary_sim_matrix,
            lambda j, i: ordered_chunks[j].document_idx == i,
            np.any,
        )
        return {
            "summary_idx": summary_idx,
            "summarizer_name": summarizer_name,
            "llm_name": llm_name,
            "bin_idx": summary.bin_idx,
            "bin_name": self.context.corpus[summary.bin_idx].name,
            "duration": metadata["duration"],
            "summary_tokens": len(self.ENC.encode(summary_text)),
            "n_summary_chunks": binary_sim_matrix.shape[1],
            "chunks_per_chunk": (
                sum(np.any(binary_sim_matrix, axis=1)) / binary_sim_matrix.shape[1]
                if sum(np.any(binary_sim_matrix, axis=1))
                else 0
            ),
            "docs_per_chunk": (
                sum(doc_usage) / binary_sim_matrix.shape[1] if sum(doc_usage) else 0
            ),
        }

    async def evaluate_overall(self, overall_results, prepared_data):
        df = pd.DataFrame(overall_results)
        df.to_csv(os.path.join(self.overall_analysis_dir, "stats.csv"), index=False)

        # Plot Averages
        self.print_metrics(
            df,
            aggregate_by=["summarizer_name", "llm_name"],
            title="Overall performance",
        )
        self.print_metrics(
            df, aggregate_by=["llm_name"], title="Performance by LLM"
        )
        self.print_metrics(
            df,
            aggregate_by=["summarizer_name"],
            title="Performance by Summary Strategy",
        )
        self.print_metrics(
            df, aggregate_by=["bin_idx", "bin_name"], title="Performance by Bin"
        )

    @staticmethod
    def print_metrics(df, aggregate_by: List[str], title: str):
        my_metrics_config = [
            {
                "col": "docs_per_chunk",
                "header": "Docs per Sum-Chunk",
                "high_is_good": True,
            },
            {
                "col": "chunks_per_chunk",
                "header": "Docs per Sum-Chunk",
                "high_is_good": True,
            },
            {
                "col": "n_summary_chunks",
                "header": "N Sum-Chunks",
                "high_is_good": True,
            },
            {
                "col": "summary_tokens",
                "header": "Sum. Tokens",
                "high_is_good": False,
            },
            {
                "col": "duration",
                "header": "Duration",
                "unit": "s",
                "high_is_good": False,
            },
        ]

        pretty_table(
            df=df,
            group_cols=aggregate_by,
            metrics_config=my_metrics_config,
            title=title,
        )

    async def prepare_data(self):
        chunks_per_summary = {}

        for idx, s in enumerate(self.context.summaries):
            doc_chunks = [
                Chunk(**c)
                for c in await json_load(
                    os.path.join(self.context.get_save_path(s), "document_chunks.json")
                )
            ]
            doc_order = self.context.get_document_order(s)

            for chunk in doc_chunks:
                chunk.document_idx = doc_order[chunk.document_idx]
            chunks_per_summary[idx] = doc_chunks

        return chunks_per_summary


async def main():
    evaluator = SummaryStatsEvaluator(QueryBasedSummaryContext())
    await evaluator.evaluate()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
