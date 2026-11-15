import asyncio
import os
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.generic_summaries.context import GenericSummaryContext
from experiments.generic_summaries.evaluation._visualization import (
    plot_doc_usage_frequency,
    plot_chunk_usage,
    plot_avg_metric,
    plot_bin_heatmap,
    plot_chunk_usage_overall_interpolated,
    plot_chunk_usage_lines,
)
from experiments.generic_summaries.evaluation.base_evaluator import GenericSummaryBaseEvaluator
from models.corpus import GenericSummary
from utils.partitioning import aggregate_partition_usage
from utils.pretty_table import pretty_table
from utils.vis import apply_name_mappings


class CoverageEvaluator(GenericSummaryBaseEvaluator):
    def __init__(self, context: GenericSummaryContext):
        super().__init__(context, "input_coverage")

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
        _, _, binary_sim_matrix, raw_sim_matrix = await self._load_llm_data(
            summary, summarizer_name, llm_name
        )

        tokens_by_doc = defaultdict(list)
        chunk_count_per_doc = defaultdict(int)
        for chunk, token_count in zip(ordered_chunks, ordered_chunk_tokens):
            tokens_by_doc[chunk.document_idx].append(token_count)
            chunk_count_per_doc[chunk.document_idx] += 1

        # Document Usage
        doc_usage = aggregate_partition_usage(
            len(b.document_paths),
            binary_sim_matrix,
            lambda j, i: ordered_chunks[j].document_idx == i,
            np.any,
        )

        # Chunk Usage
        chunk_usage = np.any(binary_sim_matrix, axis=1)

        # Stats
        return {
            "summary_idx": summary_idx,
            "summarizer_name": summarizer_name,
            "llm_name": llm_name,
            "bin_idx": summary.bin_idx,
            "bin_name": self.context.corpus[summary.bin_idx].name,
            "percentage_docs_used": np.mean(doc_usage > 0),
            "percentage_chunks_used": np.mean(chunk_usage > 0),
            "doc_order": summary.document_order,
            "doc_usage": doc_usage,
            "doc_lengths": ordered_doc_tokens,
            "chunk_usage": chunk_usage,
            "chunk_usage_raw": np.mean(raw_sim_matrix, axis=1),
            "chunk_tokens_by_doc": dict(tokens_by_doc),
            "chunk_count_per_doc": chunk_count_per_doc,
        }

    async def evaluate_bin(
        self, bin_idx, b, summarizer_name, llm_name, bin_results, prepared_data
    ):
        base_save_path = os.path.join(
            self.bin_analysis_dir, str(bin_idx), summarizer_name, llm_name
        )
        os.makedirs(base_save_path, exist_ok=True)

        save_llm_name = llm_name.replace("_", r"\_")
        plot_bin_heatmap(
            [item["doc_usage"] for item in bin_results],
            [item["doc_order"] for item in bin_results],
            b,
            sub_titles=[b.name, f"{summarizer_name} - {save_llm_name}"],
            save_path=os.path.join(base_save_path, f"heatmap.png"),
        )
        plot_doc_usage_frequency(
            [item["doc_usage"] for item in bin_results],
            b,
            title=f"{b.name}\n{summarizer_name} - {save_llm_name}",
            save_path=os.path.join(base_save_path, f"doc_usage.png"),
        )
        plot_chunk_usage(
            [item["doc_order"] for item in bin_results],
            [item["chunk_usage"] for item in bin_results],
            b,
            prepared_data[0][bin_idx],
            save_path=os.path.join(base_save_path, f"chunk_usage"),
        )

        return bin_results

    async def evaluate_overall(self, overall_results, prepared_data):
        os.makedirs(
            os.path.join(self.overall_analysis_dir, "chunk_usage_per_doc"),
            exist_ok=True,
        )

        chunk_parts = [3, 5, 10, 15, 20]

        plot_chunk_usage_lines(
            overall_results,
            self.context.corpus,
            save_path=os.path.join(self.overall_analysis_dir, "doc_usage_per_topic"),
            include_ci=False,
        )
        plot_chunk_usage_lines(
            overall_results,
            self.context.corpus,
            save_path=os.path.join(self.overall_analysis_dir, "doc_usage_per_topic_ci"),
            include_ci=True,
        )
        for p in chunk_parts:
            plot_chunk_usage_overall_interpolated(
                overall_results,
                save_path=os.path.join(
                    self.overall_analysis_dir, "chunk_usage_per_doc", str(p)
                ),
                n_partitions=p,
            )


        df = pd.DataFrame(overall_results)
        df.to_csv(
            os.path.join(self.overall_analysis_dir, "coverage_stats.csv"), index=False
        )

        # Aggregate metrics
        grouped_df = (
            df.groupby(["summarizer_name", "llm_name"])
            .agg(
                {
                    "percentage_docs_used": "mean",
                    "percentage_chunks_used": "mean",
                }
            )
            .reset_index()
        )

        # Plot Averages
        plot_df = grouped_df.copy()
        apply_name_mappings(plot_df, rename_columns=False)
        self.plot_metrics(plot_df, self.overall_analysis_dir)

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

        self.analyse_per_bin(df, prepared_data)

    @staticmethod
    def print_metrics(df, aggregate_by: List[str], title: str):
        my_metrics_config = [
            {
                "col": "percentage_docs_used",
                "header": "% Documents",
                "mult": 100,
                "unit": "%",
                "high_is_good": True,
            },
            {
                "col": "percentage_chunks_used",
                "header": "% Segments",
                "mult": 100,
                "unit": "%",
                "high_is_good": True,
            },
        ]

        pretty_table(
            df=df,
            group_cols=aggregate_by,
            metrics_config=my_metrics_config,
            title=title,
        )

    def analyse_per_bin(self, df, prepared_data):
        grouped = df.groupby("bin_idx")

        for bin_idx, group in tqdm(
            grouped,
            total=grouped.ngroups,
            desc="Analyse the different bins...",
        ):
            base_save_path = os.path.join(self.bin_analysis_dir, str(bin_idx))
            b = self.context.corpus[bin_idx]
            doc_usages = group["doc_usage"].tolist()
            doc_orders = group["doc_order"].tolist()
            chunk_usages = group["chunk_usage"].tolist()

            normalized_chunk_usages = [
                arr / len(b.document_paths) for arr in chunk_usages
            ]
            normalized_doc_usages = [arr / len(b.document_paths) for arr in doc_usages]

            plot_doc_usage_frequency(
                normalized_doc_usages,
                b,
                title=f"{b.name}",
                save_path=os.path.join(base_save_path, f"doc_usage.png"),
            )
            plot_chunk_usage(
                doc_orders,
                normalized_chunk_usages,
                b,
                prepared_data[0][bin_idx],
                save_path=os.path.join(base_save_path, f"chunk_usage"),
            )
            plot_bin_heatmap(
                normalized_doc_usages,
                doc_orders,
                b,
                sub_titles=[b.name],
                save_path=os.path.join(base_save_path, f"heatmap.png"),
            )

            grouped_df = (
                group.groupby(["summarizer_name", "llm_name"])
                .agg(
                    {
                        "percentage_docs_used": "mean",
                        "percentage_chunks_used": "mean",
                    }
                )
                .reset_index()
            )

            plot_df = grouped_df.copy()
            apply_name_mappings(plot_df, rename_columns=False)
            self.plot_metrics(plot_df, base_save_path)

    @staticmethod
    def plot_metrics(df, save_path: str):
        plot_avg_metric(
            df,
            metric="percentage_docs_used",
            metric_type="\% Documents",
            save_path=os.path.join(save_path, "avg_docs_used"),
        )
        plot_avg_metric(
            df,
            metric="percentage_chunks_used",
            metric_type="\% Segments",
            save_path=os.path.join(save_path, "avg_chunks_used"),
        )


async def main():
    evaluator = CoverageEvaluator(GenericSummaryContext())
    await evaluator.evaluate()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
