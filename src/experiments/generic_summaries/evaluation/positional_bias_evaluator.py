import asyncio
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from colorama import Fore
from tqdm import tqdm

from experiments.generic_summaries.context import GenericSummaryContext
from experiments.generic_summaries.evaluation._visualization import (
    plot_partition_usage,
    plot_partition_heatmap,
    plot_overall_partition_usage,
)
from experiments.generic_summaries.evaluation.base_evaluator import GenericSummaryBaseEvaluator
from models.corpus import GenericSummary, Bin
from utils.partitioning import (
    aggregate_partition_usage,
    partition_centered,
)
from utils.pretty_table import pretty_table
from utils.vis import apply_name_mappings


class PositionalBiasEvaluator(GenericSummaryBaseEvaluator):
    def __init__(self, context: GenericSummaryContext, n_partitions: int = 25):
        super().__init__(
            context,
            os.path.join(
                f"positional_bias",
                f"{n_partitions}_partitions",
            ),
        )
        self.N_PARTITIONS = n_partitions

    async def evaluate_single_summary(
        self,
        b: Bin,
        summary_idx: int,
        summary: GenericSummary,
        prepared_data,
        summarizer_name: str,
        llm_name: str,
    ):
        ordered_chunks, ordered_chunk_tokens, _ = (
            GenericSummaryBaseEvaluator.get_ordered_data(prepared_data, summary)
        )
        _, _, binary_sim_matrix, _ = await self._load_llm_data(
            summary, summarizer_name, llm_name
        )

        # Calculate Partition Indices
        partition_indices = partition_centered(ordered_chunk_tokens, self.N_PARTITIONS)

        # Calculate Usage per Partition
        part_usage = (
            aggregate_partition_usage(
                self.N_PARTITIONS,
                binary_sim_matrix,
                lambda j, i: i == partition_indices[j],  # noqa
                lambda x: np.count_nonzero(np.any(x, axis=1)),
                normalize=True,
            )
        )

        row_stats: Dict[str, Any] = {
            "summary_idx": summary_idx,
            "summarizer_name": summarizer_name,
            "llm_name": llm_name,
            "bin_idx": summary.bin_idx,
            "bin_name": b.name,
            "part_usage": part_usage,
        }
        for i, usage_val in enumerate(part_usage):
            row_stats[f"partition_{i}"] = usage_val

        return row_stats

    async def evaluate_llm_summarizer(
        self, summarizer_llm_results, summarizer_name, llm_name
    ):
        base_save_path = os.path.join(
            self.overall_analysis_dir, summarizer_name, llm_name
        )
        os.makedirs(base_save_path, exist_ok=True)

        plot_partition_usage(
            np.mean([item["part_usage"] for item in summarizer_llm_results], axis=0),
            title=f"{summarizer_name} / {llm_name}",
            save_path=os.path.join(base_save_path, "avg_partition_usage.png"),
        )
        plot_partition_heatmap(
            pd.DataFrame(summarizer_llm_results),
            title=f"{summarizer_name} / {llm_name}",
            save_path=os.path.join(base_save_path, "partition_heatmap.png"),
        )
        return summarizer_llm_results

    async def evaluate_overall(self, overall_results, prepared_data):
        # Save aggregated CSV and Overall Partition Plot
        df = pd.DataFrame(overall_results)
        df.to_csv(
            os.path.join(self.overall_analysis_dir, "partition_stats.csv"), index=False
        )

        plot_partition_heatmap(
            df,
            save_path=os.path.join(self.overall_analysis_dir, "partition_heatmap.png"),
        )

        # Aggregate columns partition_0 to partition_N
        agg_rules = {f"partition_{i}": "mean" for i in range(self.N_PARTITIONS)}
        df_by_llm_strat = (
            df.groupby(["summarizer_name", "llm_name"]).agg(agg_rules).reset_index()
        )
        df_by_llm_strat.to_csv(
            os.path.join(
                self.overall_analysis_dir, "partition_stats_per_llm_strat.csv"
            ),
            index=False,
        )

        for summarizer_name, group_df in df.groupby("summarizer_name"):
            plot_partition_heatmap(
                group_df,
                save_path=os.path.join(
                    self.overall_analysis_dir, summarizer_name, "partition_heatmap.png"
                ),
                title=f"Summarization strategy: {summarizer_name}",
            )

        # Apply the mapping
        plot_df = df.copy()
        apply_name_mappings(plot_df, rename_columns=False)

        self.plot_partition_usage(plot_df, self.overall_analysis_dir)

        df_by_bin = df.groupby(["bin_idx", "bin_name"]).agg(agg_rules).reset_index()
        df_by_bin.to_csv(
            os.path.join(self.bin_analysis_dir, "partition_stats_per_bin.csv"),
            index=False,
        )

        self.analyse_per_bin(df)

        self.pretty_print(df, "llm_name", "Partition Usage By LLM")
        self.pretty_print(
            df, "summarizer_name", "Partition Usage By Summary Strategy"
        )
        self.pretty_print(df, "bin_idx", "Overview Over Partition Usage")

    def analyse_per_bin(self, df: pd.DataFrame):
        grouped = df.groupby(["bin_idx"])

        for key, group in tqdm(
            grouped,
            total=grouped.ngroups,
            desc="Analyse the different bins...",
        ):

            agg_rules = {f"partition_{i}": "mean" for i in range(self.N_PARTITIONS)}
            grouped_df = (
                group.groupby(["summarizer_name", "llm_name"])
                .agg(agg_rules)
                .reset_index()
            )

            path = os.path.join(self.bin_analysis_dir, str(key[0]))
            os.makedirs(path, exist_ok=True)

            self.plot_partition_usage(group, path)
            grouped_df.to_csv(os.path.join(path, "partition_stats.csv"), index=False)

    def pretty_print(self, df: pd.DataFrame, aggregate_by: str, title):
        my_parts = [f"partition_{i}" for i in range(self.N_PARTITIONS)]

        # Group by the metadata columns and calculate mean for partition columns
        grouped_df = df.groupby(aggregate_by)[my_parts].mean()
        stacked_df = grouped_df.stack()

        stacked_df.index.names = [aggregate_by] + ["partition"]
        final_df = stacked_df.unstack([aggregate_by])

        final_df.columns.name = None
        final_df = final_df.reset_index()

        my_metrics_config = [
            {
                "col": c,
                "header": c,
                "mult": 1,
                "unit": "%",
                "high_is_good": True,
            }
            for c in final_df.columns.tolist()
            if not c == "partition"
        ]

        final_df["partition"] = (
            final_df["partition"].str.removeprefix("partition_").astype(int)
        )

        pretty_table(
            df=final_df,
            group_cols=["partition"],
            metrics_config=my_metrics_config,
            title=title,
        )

    @staticmethod
    def plot_partition_usage(df, save_path):
        plot_overall_partition_usage(
            df,
            save_path=os.path.join(save_path, "partition_usage_by_strategy"),
            errorbar=None,
        )

        plot_overall_partition_usage(
            df,
            save_path=os.path.join(save_path, "partition_usage_chunked"),
            errorbar=None,
            summary_strategies=["Hierarchical Merge", "Incremental Update"]
        )
        plot_overall_partition_usage(
            df,
            save_path=os.path.join(save_path, "partition_usage_single_pass"),
            errorbar=None,
            summary_strategies=["Single Pass"]
        )
        plot_overall_partition_usage(
            df,
            save_path=os.path.join(save_path, "partition_usage_by_strategy_ci"),
        )


async def main():
    ctx = GenericSummaryContext()

    print(f"{Fore.CYAN}>>> Running Partition Evaluation (25)... <<<")
    part_evaluator_25 = PositionalBiasEvaluator(ctx, n_partitions=25)
    await part_evaluator_25.evaluate()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
