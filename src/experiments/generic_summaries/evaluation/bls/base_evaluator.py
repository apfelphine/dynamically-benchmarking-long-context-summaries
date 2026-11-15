import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.generic_summaries.evaluation.base_evaluator import GenericSummaryBaseEvaluator
from models.corpus import GenericSummary, Bin
from utils.partitioning import aggregate_partition_usage


class BaseRegressionEvaluator(GenericSummaryBaseEvaluator, ABC):
    """
    Shared base class for regression-based evaluations (OLS and WLS).
    Configurable for either normalized relative position bucketing or raw absolute positions.
    """

    def __init__(
        self,
        context,
        evaluator_name: str,
        use_relative_bucketing: bool = True,
        n_relative_bins: int = 4 # 4 buckets -> quartiles
    ):
        suffix = "absolute" if not use_relative_bucketing else "relative"
        super().__init__(context, "bls/" +  evaluator_name + "/" + suffix)
        self.use_relative_bucketing = use_relative_bucketing
        self.n_relative_bins = n_relative_bins

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

        doc_usage_mean = aggregate_partition_usage(
            n_partitions=len(b.document_paths),
            sim_matrix=binary_sim_matrix,
            check_chunk=lambda j, i: ordered_chunks[j].document_idx == i,
            func=lambda x: np.mean(np.any(x, axis=1)),
        )

        doc_usage_sum = aggregate_partition_usage(
            n_partitions=len(b.document_paths),
            sim_matrix=binary_sim_matrix,
            check_chunk=lambda j, i: ordered_chunks[j].document_idx == i,
            func=lambda x: np.sum(np.any(x, axis=1)),
        )

        total_docs = len(summary.document_order)
        doc_utilizations = {}
        for pos, doc_idx in enumerate(summary.document_order):
            if self.use_relative_bucketing:
                rel_pos = pos / (total_docs - 1) if total_docs > 1 else 0.0
                position_metric = min(int(rel_pos * self.n_relative_bins), self.n_relative_bins - 1)
                metric_key = "relative_bucket"
            else:
                position_metric = pos
                metric_key = "absolute_position"

            doc_utilizations[doc_idx] = {
                "position": pos,
                metric_key: position_metric,
                "utilization_mean": float(doc_usage_mean[doc_idx]),
                "utilization_sum": float(doc_usage_sum[doc_idx])
            }

        return {
            "summarizer_name": summarizer_name,
            "llm_name": llm_name,
            "bin_idx": summary.bin_idx,
            "doc_utilizations": doc_utilizations
        }

    def _extract_global_cells(self, group: pd.DataFrame) -> np.ndarray:
        """Extracts evaluation cells globally based on whether relative bucketing is enabled."""
        cells = []
        article_mapping = {}
        for row in group.itertuples():
            b_idx = row.bin_idx
            for a_idx, data in row.doc_utilizations.items():
                global_a_idx = f"{b_idx}_{a_idx}"
                if global_a_idx not in article_mapping:
                    article_mapping[global_a_idx] = len(article_mapping)

                mapped_a_idx = article_mapping[global_a_idx]

                # Dynamically choose metric depending on configuration flag
                pos_val = data["relative_bucket"] if self.use_relative_bucketing else data["absolute_position"]

                cells.append((
                    mapped_a_idx,
                    pos_val,
                    data["utilization_mean"],
                    data["utilization_sum"],
                    b_idx
                ))
        return np.array(cells, dtype=float)

    @abstractmethod
    def _run_regression_for_target(
            self, y_target: np.ndarray, a_idx_arr: np.ndarray, p_idx_arr: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, float], np.ndarray]:
        pass

    def _save_model_artifacts(self, summarizer: str, llm: str, curve_mean: np.ndarray, curve_sum: np.ndarray,
                              prefix: str):
        save_dir = os.path.join(self.overall_analysis_dir, summarizer, llm)
        os.makedirs(save_dir, exist_ok=True)

        mode_label = "relative_bucket" if self.use_relative_bucketing else "absolute_position"

        curve_df = pd.DataFrame({
            mode_label: range(len(curve_mean)),
            "centered_effect_mean": curve_mean,
            "centered_effect_sum": curve_sum
        })
        curve_df.to_csv(os.path.join(save_dir, f"{prefix}_{mode_label}_curves.csv"), index=False)

        plt.figure(figsize=(8, 4))
        plt.plot(curve_df[mode_label], curve_df["centered_effect_mean"], marker='o', label="Mean (Fraction)",
                 color='blue')
        plt.plot(curve_df[mode_label], curve_df["centered_effect_sum"], marker='s', linestyle='--', label="Sum (Count)",
                 color='orange')
        plt.axhline(0, color='gray', linestyle=':', alpha=0.7)

        title_mode = f"Relative ({self.n_relative_bins} Bins)" if self.use_relative_bucketing else "Absolute Positions"
        plt.title(f"{prefix.upper()} Positional Bias Curve ({title_mode}): {summarizer} / {llm}")
        plt.xlabel("Prompt Position")
        plt.ylabel("Centered Effect on Utilization")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_{mode_label}_bias.png"), dpi=300)
        plt.close()

    def _save_global_comparison_plot(self, stats_df: pd.DataFrame, prefix: str):
        """Generates a clean 3-panel faceted bar chart grouped by strategy."""
        if stats_df.empty:
            return

        strategies = stats_df["summarizer_name"].unique()
        fig, axes = plt.subplots(1, len(strategies), figsize=(16, 5), sharey=True)
        if len(strategies) == 1:
            axes = [axes]

        for ax, strategy in zip(axes, strategies):
            strat_data = stats_df[stats_df["summarizer_name"] == strategy]
            llms = strat_data["llm_name"]
            x = np.arange(len(llms))
            width = 0.35

            ax.bar(x - width / 2, strat_data["mean_share_pos"], width, label="Positional Share", color='crimson')
            ax.bar(x + width / 2, strat_data["mean_share_art"], width, label="Content Share", color='teal')

            ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
            ax.set_title(strategy)
            ax.set_xticks(x)
            ax.set_xticklabels(llms, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            ax.legend(loc='upper right')

        mode_title = "Normalized" if self.use_relative_bucketing else "Absolute"
        fig.suptitle(f"Tug-of-War Across Strategies ({prefix.upper()} - {mode_title})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.overall_analysis_dir, f"{prefix}_share_faceted_global.png"), dpi=300)
        plt.close()