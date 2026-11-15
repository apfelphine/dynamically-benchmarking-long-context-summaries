import asyncio
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from colorama import Fore, Style

from experiments.generic_summaries.context import GenericSummaryContext
from experiments.generic_summaries.evaluation.base_evaluator import GenericSummaryBaseEvaluator
from models.corpus import GenericSummary, Bin
from utils.partitioning import (
    aggregate_partition_usage,
    partition_centered,
)
from utils.pretty_table import pretty_table
from utils.vis import apply_name_mappings


class AreaOfDeficitEvaluator(GenericSummaryBaseEvaluator):
    def __init__(
            self,
            context: GenericSummaryContext,
            n_partitions: int = 25,
            aggregation_mode: str = "per_input",
    ):
        super().__init__(
            context,
            os.path.join("area_of_deficit", f"{n_partitions}_partitions", aggregation_mode),
        )
        self.baseline = "single-pass / gpt-4.1-mini"
        self.N_PARTITIONS = n_partitions
        self.aggregation_mode = aggregation_mode

    @staticmethod
    def calculate_aod(usage_array: np.ndarray) -> float:
        """
        Calculates the Normalized Area of Deficit (AoD) for an array.
        Formula: (y_max - mu) / y_max
        """
        if len(usage_array) == 0:
            return 0.0
        y_max = np.max(usage_array)
        if y_max == 0:
            return 0.0
        mu = np.mean(usage_array)
        return float((y_max - mu) / y_max)

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

        row_stats: Dict[str, Any] = {
            "summary_idx": summary_idx,
            "summarizer_name": summarizer_name,
            "llm_name": llm_name,
            "bin_idx": summary.bin_idx,
            "bin_name": b.name,
        }

        # Calculate Partition Indices
        partition_indices = partition_centered(ordered_chunk_tokens, self.N_PARTITIONS)

        # Calculate Usage per Partition (returns % values)
        usage_array = (
            aggregate_partition_usage(
                self.N_PARTITIONS,
                binary_sim_matrix,
                lambda j, i: i == partition_indices[j],  # noqa
                lambda x: np.count_nonzero(np.any(x, axis=1)),
                normalize=True,
            )
            * 100
        )

        row_stats["part_usage"] = usage_array
        for i, usage_val in enumerate(usage_array):
            row_stats[f"partition_{i}"] = usage_val

        return row_stats

    async def evaluate_overall(self, overall_results, prepared_data):
        df = pd.DataFrame(overall_results)
        partition_cols = [f"partition_{i}" for i in range(self.N_PARTITIONS)]

        if self.aggregation_mode == "global":
            print(f"{Fore.CYAN}>>> Mode: Globally Aggregated Curve AoD with Curve-Level Bootstrap CIs <<<")
            aggregated_rows = []
            np.random.seed(42)

            model_groups = df.groupby(["summarizer_name", "llm_name"])

            for (sum_name, llm_name), group in model_groups:
                arrays = group[partition_cols].values  # Shape: (n_inputs, N_PARTITIONS)
                n_inputs = len(arrays)
                if n_inputs == 0:
                    continue

                # 1. Calculate true Global Curve AoD
                mean_curve = np.mean(arrays, axis=0)
                true_aod = self.calculate_aod(mean_curve)

                # 2. Curve-Level Bootstrap (1000 resamples of the macro curve)
                boot_aods = []
                for _ in range(1000):
                    indices = np.random.choice(n_inputs, size=n_inputs, replace=True)
                    resampled_arrays = arrays[indices]
                    boot_mean_curve = np.mean(resampled_arrays, axis=0)
                    boot_aods.append(self.calculate_aod(boot_mean_curve))

                ci_lower, ci_upper = np.percentile(boot_aods, [2.5, 97.5])

                aggregated_rows.append({
                    "summarizer_name": sum_name,
                    "llm_name": llm_name,
                    "mean_aod": float(true_aod),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                })

            df_agg = pd.DataFrame(aggregated_rows)

            self.print_aod(df_agg, "Area Of Deficit (Global Curve with Bootstrap CIs)")

            df_agg.to_csv(
                os.path.join(self.overall_analysis_dir, "aod_global_curve_ci.csv"),
                index=False,
            )
            print(f"{Fore.GREEN}>>> Global curve bootstrap results saved successfully. <<<")

        elif self.aggregation_mode == "per_bin":
            print(f"{Fore.CYAN}>>> Mode: Macro-Averaged Per-Bin Curve AoD with Bootstrap CIs <<<")
            aggregated_rows = []
            np.random.seed(42)

            model_groups = df.groupby(["summarizer_name", "llm_name"])

            for (sum_name, llm_name), group in model_groups:
                # Extract partition arrays for each bin separately
                bin_groups = [b_group[partition_cols].values for _, b_group in group.groupby("bin_name")]

                if not bin_groups:
                    continue

                # 1. Calculate true AoD per bin, then macro-average them
                bin_aods = []
                for bg in bin_groups:
                    if len(bg) > 0:
                        mean_bin_curve = np.mean(bg, axis=0)
                        bin_aods.append(self.calculate_aod(mean_bin_curve))

                true_macro_aod = np.mean(bin_aods) if bin_aods else 0.0

                # 2. Bootstrap CI by resampling inputs *within* each bin
                boot_macro_aods = []
                for _ in range(1000):
                    boot_bin_aods = []
                    for bg in bin_groups:
                        n_inputs = len(bg)
                        if n_inputs > 0:
                            indices = np.random.choice(n_inputs, size=n_inputs, replace=True)
                            resampled_bg = bg[indices]
                            boot_mean_curve = np.mean(resampled_bg, axis=0)
                            boot_bin_aods.append(self.calculate_aod(boot_mean_curve))

                    boot_macro_aods.append(np.mean(boot_bin_aods))

                ci_lower, ci_upper = np.percentile(boot_macro_aods, [2.5, 97.5])

                aggregated_rows.append({
                    "summarizer_name": sum_name,
                    "llm_name": llm_name,
                    "mean_aod": float(true_macro_aod),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                })

            df_agg = pd.DataFrame(aggregated_rows)

            self.print_aod(df_agg, "Area Of Deficit (Macro-Averaged Per-Bin with Bootstrap CIs)")

            df_agg.to_csv(
                os.path.join(self.overall_analysis_dir, "aod_macro_per_bin_curve_ci.csv"),
                index=False,
            )
            print(f"{Fore.GREEN}>>> Macro-averaged per-bin bootstrap results saved successfully. <<<")

        elif self.aggregation_mode == "per_input":
            print(f"{Fore.CYAN}>>> Mode: Per-Input AoD with Bootstrap CIs & Statistical Tests <<<")
            # 1. Compute AoD per individual input summary
            df["input_aod"] = df["part_usage"].apply(lambda arr: self.calculate_aod(arr))

            # 2. Compute Bootstrap 95% CIs and Mean per (summarizer_name, llm_name)
            aggregated_rows = []
            np.random.seed(42)

            for (sum_name, llm_name), group in df.groupby(["summarizer_name", "llm_name"]):
                vals = group["input_aod"].values
                mean_aod = float(np.mean(vals))

                if len(vals) > 1:
                    boot_means = [
                        np.mean(np.random.choice(vals, size=len(vals), replace=True))
                        for _ in range(1000)
                    ]
                    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
                else:
                    ci_lower, ci_upper = mean_aod, mean_aod

                aggregated_rows.append({
                    "summarizer_name": sum_name,
                    "llm_name": llm_name,
                    "mean_aod": mean_aod,
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                })

            df_agg = pd.DataFrame(aggregated_rows)

            self.print_aod(df_agg, "Area Of Deficit (Per-Input Mean with Bootstrap CIs)")

            df_agg.to_csv(
                os.path.join(self.overall_analysis_dir, "aod_per_llm_strat_ci.csv"),
                index=False,
            )


    @staticmethod
    def print_aod(df, title):
        df = df.copy()
        df = apply_name_mappings(df, rename_columns=False)

        column_ranges = {}
        for llm in df["llm_name"].unique():
            subset = df[df["llm_name"] == llm]["mean_aod"]
            column_ranges[llm] = {
                "min": subset.min(),
                "max": subset.max()
            }

        def format_relative_colored(row):
            val = row['mean_aod']
            low = row['ci_lower']
            upp = row['ci_upper']
            llm = row['llm_name']

            text_str = f"{val:.2f} [{low:.2f}, {upp:.2f}]"

            r = column_ranges.get(llm, {"min": val, "max": val})
            min_val, max_val = r["min"], r["max"]

            if max_val == min_val:
                color = Style.RESET_ALL
            else:
                if val == min_val:
                    color = Fore.GREEN
                elif val == max_val:
                    color = Fore.RED
                else:
                    color = Fore.RESET

            return f"{color}{text_str}{Style.RESET_ALL}"

        # Apply formatting to create the display string column
        df["formatted_val"] = df.apply(format_relative_colored, axis=1)

        # Pivot so summarizers are rows and LLMs are columns using the formatted string
        pivot_df = df.pivot_table(
            index="summarizer_name",
            columns="llm_name",
            values="formatted_val",
            aggfunc="first"
        ).reset_index()

        pivot_df.columns.name = None
        llm_columns = [c for c in pivot_df.columns if c != "summarizer_name"]

        my_metrics_config = [
            {
                "col": c,
                "header": c,
                "agg": "first",
                "decimals": None
            }
            for c in llm_columns
        ]

        pretty_table(
            df=pivot_df,
            group_cols=["summarizer_name"],
            metrics_config=my_metrics_config,
            title=title,
        )


async def main():
    ctx = GenericSummaryContext()

    print(f"{Fore.CYAN}>>> Running Partition Evaluation (25) / Global... <<<")
    await AreaOfDeficitEvaluator(ctx, n_partitions=25, aggregation_mode="global").evaluate()
    print(f"{Fore.CYAN}>>> Running Partition Evaluation (25) / Per Bin... <<<")
    await AreaOfDeficitEvaluator(ctx, n_partitions=25, aggregation_mode="per_bin").evaluate()
    print(f"{Fore.CYAN}>>> Running Partition Evaluation (25) / Per Input... <<<")
    await AreaOfDeficitEvaluator(ctx, n_partitions=25, aggregation_mode="per_input").evaluate()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())