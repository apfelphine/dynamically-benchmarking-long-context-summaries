import asyncio
import os
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.generic_summaries.context import GenericSummaryContext
from experiments.generic_summaries.evaluation.base_evaluator import GenericSummaryBaseEvaluator
from models.corpus import GenericSummary
from utils.partitioning import aggregate_partition_usage
from utils.pretty_table import pretty_table


class SummaryStatsEvaluator(GenericSummaryBaseEvaluator):
    def __init__(self, context: GenericSummaryContext):
        super().__init__(context, "summary_statistics")

    async def evaluate_single_summary(
        self,
        b,
        summary_idx,
        summary: GenericSummary,
        prepared_data,
        summarizer_name,
        llm_name,
    ):
        ordered_chunks, ordered_chunk_tokens, ordered_doc_tokens = (
            GenericSummaryBaseEvaluator.get_ordered_data(prepared_data, summary)
        )
        summary_text, metadata, binary_sim_matrix, _ = await self._load_llm_data(
            summary, summarizer_name, llm_name
        )

        doc_usage = aggregate_partition_usage(
            len(b.document_paths),
            binary_sim_matrix,
            lambda j, i: ordered_chunks[j].document_idx == i,
            np.any,
        )
        chunk_usage = np.any(binary_sim_matrix, axis=1)

        summary_tokens = len(self.ENC.encode(summary_text))
        return {
            "summary_idx": summary_idx,
            "summarizer_name": summarizer_name,
            "llm_name": llm_name,
            "bin_idx": summary.bin_idx,
            "bin_name": self.context.corpus[summary.bin_idx].name,
            "duration": metadata["duration"],
            "summary_tokens": summary_tokens,
            "n_summary_chunks": binary_sim_matrix.shape[1],
            "chunks_per_chunk": sum(chunk_usage) / binary_sim_matrix.shape[1],
            "docs_per_chunk": sum(doc_usage) / binary_sim_matrix.shape[1],
            # Compression Rate
            # 1.0: Summary has same length as document
            # 0.5: Summary is half the length of the document
            "compression_rate": summary_tokens / b.doc_tokens,
        }

    async def evaluate_overall(self, overall_results, prepared_data):
        df = pd.DataFrame(overall_results)
        df.to_csv(os.path.join(self.overall_analysis_dir, "stats.csv"), index=False)

        # Aggregate metrics
        grouped_df = (
            df.groupby(["summarizer_name", "llm_name"])
            .agg(
                {
                    "duration": "mean",
                    "summary_tokens": "mean",
                    "docs_per_chunk": "mean",
                    "n_summary_chunks": "mean",
                    "chunks_per_chunk": "mean",
                    "compression_rate": "mean"
                }
            )
            .reset_index()
        )

        self.print_metrics(
            df,
            aggregate_by=["summarizer_name", "llm_name"],
            title="Overall performance",
        )
        self.print_metrics(
            df, aggregate_by=["llm_name"], title="Performance per LLM"
        )
        self.print_metrics(
            df,
            aggregate_by=["summarizer_name"],
            title="Performance per Summary Strategy",
        )
        self.print_metrics(
            df, aggregate_by=["bin_idx", "bin_name"], title="Performance per Bin"
        )

        self.analyse_per_bin(df)

    @staticmethod
    def print_metrics(df, aggregate_by: List[str], title: str):
        my_metrics_config = [
            {
                "col": "docs_per_chunk",
                "header": "Docs pro Sum-Chunk",
                "high_is_good": True,
            },
            {
                "col": "chunks_per_chunk",
                "header": "Chunks pro Sum-Chunk",
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
            {
                "col": "compression_rate",
                "header": "Compress. Rate",
                "high_is_good": False,
                "decimals": 4
            },
        ]

        pretty_table(
            df=df,
            group_cols=aggregate_by,
            metrics_config=my_metrics_config,
            title=title,
        )

    def analyse_per_bin(self, df):
        grouped = df.groupby("bin_idx")

        for bin_idx, group in tqdm(
            grouped,
            total=grouped.ngroups,
            desc="Analyse the different bins...",
        ):
            base_save_path = os.path.join(self.bin_analysis_dir, str(bin_idx))

            grouped_df = (
                group.groupby(["summarizer_name", "llm_name"])
                .agg(
                    {
                        "duration": "mean",
                        "summary_tokens": "mean",
                        "docs_per_chunk": "mean",
                        "n_summary_chunks": "mean",
                        "chunks_per_chunk": "mean",
                    }
                )
                .reset_index()
            )

            os.makedirs(base_save_path, exist_ok=True)
            group = group.reset_index()
            group.to_csv(os.path.join(base_save_path, "stats.csv"), index=False)


async def main():
    evaluator = SummaryStatsEvaluator(GenericSummaryContext())
    await evaluator.evaluate()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
