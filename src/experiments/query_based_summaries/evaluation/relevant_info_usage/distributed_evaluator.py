import asyncio
import math
import os
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from experiments.query_based_summaries.context import QueryBasedSummaryContext
from experiments.query_based_summaries.evaluation.relevant_info_usage.base_correct_doc_usage import (
    CorrectDocUsageEvaluator,
)
from models.corpus import QueryBasedSummary

from utils.vis import apply_name_mappings


class DistributedUsageEvaluator(CorrectDocUsageEvaluator):
    # I calculate P, R, F1 based on the usage of the sections of the focus document at positions 0-3.
    # Since positions 0, 1, 2, and 3 always contain ONLY the focus document,
    # the precision can only be < 1 if the part is NOT matched AT ALL (for 0/0 -> 0).

    # The focus document is distributed across positions 0, 1, 2, 3.
    # Usage is only checked for these positions.
    # Therefore, precision can only be 0 if not a single part of the document was used at this position (0 / 0 + 0),
    # since no other documents are ever present.

    def __init__(self, context: QueryBasedSummaryContext):
        super().__init__(context, "distributed_chunks")
        self.context: QueryBasedSummaryContext = context  # Reassign for type hints

    async def evaluate_overall(self, overall_results, prepared_data):
        overall_results = [r for r in overall_results if r["is_distributed"]]

        rows = []
        for res in overall_results:
            summ_name = res["summarizer_name"]
            llm_name = res["llm_name"]
            bin_idx = res["bin_idx"]
            bin_name = res["bin_name"]

            for idx, item in enumerate(res["chunk_order_data"]):
                row = {
                    "summarizer_name": summ_name,
                    "llm_name": llm_name,
                    "bin_idx": bin_idx,
                    "bin_name": bin_name,
                    "position_idx": idx,
                    **item,
                }

                rows.append(row)

        df_chunks = pd.DataFrame(rows)
        df_chunks["position_idx"] = pd.to_numeric(df_chunks["position_idx"])
        df_chunks["article_section_idx"] = pd.to_numeric(
            df_chunks["article_section_idx"]
        )
        df_chunks.to_csv(
            os.path.join(self.overall_analysis_dir, "chunk_distribution_metrics.csv"),
            index=False,
        )
        df_chunks.groupby(["summarizer_name", "llm_name"]).agg(
            {m: "mean" for m in self.metrics}
        ).reset_index().to_csv(
            os.path.join(self.overall_analysis_dir, "partition_metrics_aggregated.csv"),
            index=False,
        )

        self.print_metrics(df_chunks, ["summarizer_name", "llm_name"], "Overview")
        self.print_metrics(df_chunks, ["bin_name"], "Analysis per Bin")

        df_chunks = apply_name_mappings(df_chunks, rename_columns=False)
        self.plot_overall_position_section_heatmap(
            df_chunks, "recall_raw", self.overall_analysis_dir
        )
        self.plot_overall_llm_summarizer_heatmap(
            df_chunks, "recall_raw", self.overall_analysis_dir
        )

        self.plot_all_interaction_heatmaps(
            df_chunks, "recall_raw", self.overall_analysis_dir
        )

        df_no_llm = df_chunks.copy()
        df_no_llm["llm_name"] = ""

        grouped = df_chunks.groupby("bin_idx")

        for bin_idx, bin_df in tqdm(
            grouped,
            total=grouped.ngroups,
            desc="Analysing the different bins...",
        ):
            bin_path = os.path.join(self.bin_analysis_dir, str(bin_idx))
            os.makedirs(bin_path, exist_ok=True)

            self.plot_all_interaction_heatmaps(bin_df, "recall_raw", bin_path)

    def plot_overall_position_section_heatmap(
        self, df: pd.DataFrame, metric: str, save_dir: str
    ):
        """
        Plots a single heatmap aggregating all data to show average performance
        per Chunk Position vs. Article Section.
        """
        metric_label = self.metrics[metric]

        # Aggregation
        df_agg = (
            df.groupby(["article_section_idx", "position_idx"])[metric]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(10, 6))

        pivot_data = df_agg.pivot(
            index="article_section_idx", columns="position_idx", values=metric
        )

        sns.heatmap(
            pivot_data,
            cmap="RdYlGn",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": metric_label},
        )

        plt.title(f"Avg. {metric_label} by position and section")
        plt.ylabel("Section of the article")
        plt.xlabel("Position Index")
        plt.tight_layout()

        plt.savefig(
            os.path.join(save_dir, f"overall_heatmap_position_section_{metric}.png")
        )
        plt.close()

    def plot_overall_llm_summarizer_heatmap(
        self, df: pd.DataFrame, metric: str, save_dir: str
    ):
        """
        Plots a single heatmap aggregating all data to show average performance
        per LLM vs. Summarizer.
        """
        metric_label = self.metrics[metric]

        # Aggregation
        df_agg = (
            df.groupby(["llm_name", "summarizer_name"])[metric].mean().reset_index()
        )

        plt.figure(figsize=(10, 8))

        pivot_data = df_agg.pivot(
            index="llm_name", columns="summarizer_name", values=metric
        )

        sns.heatmap(
            pivot_data,
            cmap="RdYlGn",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": metric_label},
        )

        plt.title(f"Avg. {metric_label} by LLM and Summarization Strategy")
        plt.ylabel("LLM Name")
        plt.xlabel("Summarizer Name")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.savefig(
            os.path.join(save_dir, f"overall_heatmap_llm_summarizer_{metric}.png")
        )
        plt.close()

    def plot_all_interaction_heatmaps(
            self,
            df: pd.DataFrame,
            metric: str,
            save_dir: str,
            name: Optional[str] = None,
            acl_columns: int = 2  # Set to 1 for single-column width, 2 for double-column width
    ):
        name = f"full_grid_heatmap_{metric}" if not name else name
        metric_label = self.metrics[metric]

        # Create custom continuous colormap
        # Adjust the order here based on whether 'Navy' or 'Maroon' represents a higher metric score.
        # Currently set to: White (Low) -> Gold (Mid) -> Navy (High)
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_theme",
            [
                '#8F241E',  # Oxford Maroon
                '#FDFBF7',  # Warm off-white (like old paper) # '#8F241E',  # Oxford Maroon
                '#10316B'   # Deep Navy
            ]
        )

        # Aggregation
        df_agg = (
            df.groupby(
                ["summarizer_name", "llm_name", "article_section_idx", "position_idx"]
            )[metric]
            .mean()
            .reset_index()
        )

        # Calculate Global Min/Max for the Scale
        global_min = df_agg[metric].min()
        global_max = df_agg[metric].max()

        llms = sorted(df_agg["llm_name"].unique())
        summarizers = sorted(df_agg["summarizer_name"].unique())

        n_llms = len(llms)
        n_summs = len(summarizers)

        # ==========================================
        # SPECIAL LAYOUT: 1 Summarizer, N LLMs (e.g., 6)
        # ==========================================
        if n_summs == 1:
            summ = summarizers[0]

            # ACL Paper Proportions
            if acl_columns == 2:
                fig_width = 6.3  # Full page width in inches
                n_cols = 3  # 3 columns of LLMs
            else:
                fig_width = 3.1  # Single column width in inches
                n_cols = 2  # 2 columns of LLMs

            n_rows = math.ceil(n_llms / n_cols)
            fig_height = 1.6 * n_rows  # Keeps heatmaps roughly square

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(fig_width, fig_height),
                sharex=True, sharey=True
            )

            axes = axes.flatten()

            for i, llm in enumerate(llms):
                ax = axes[i]

                # Set axes background color (shows if there are empty cells)
                subset = df_agg[(df_agg["llm_name"] == llm) & (df_agg["summarizer_name"] == summ)]

                sns.heatmap(
                    subset.pivot(index="article_section_idx", columns="position_idx", values=metric),
                    cmap=custom_cmap,
                    annot=True,
                    fmt=".2f",
                    cbar=False,
                    vmin=global_min,
                    vmax=global_max,
                    ax=ax,
                    linewidths=0.5,  # Adds gridlines
                    linecolor='silver',  # Silver gridlines
                    annot_kws={"size": 7 if acl_columns == 1 else 8}
                )

                # Titles and Labels sized for ACL specs
                ax.set_title(llm, fontweight="bold", fontsize=9)

                if i % n_cols == 0:
                    ax.set_ylabel("Article Section", fontsize=8)
                else:
                    ax.set_ylabel("")

                if i >= n_llms - n_cols:
                    ax.set_xlabel("Position in Input", fontsize=8)
                else:
                    ax.set_xlabel("")

                ax.tick_params(labelsize=7)

            # Remove any empty subplots if n_llms isn't perfectly divisible
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            # Colorbar placement on the right
            plt.tight_layout(rect=[0, 0, 0.90, 1])
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02 if acl_columns == 2 else 0.03, 0.7])

            norm = mpl.colors.Normalize(vmin=global_min, vmax=global_max)
            sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
            sm.set_array([])

            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(metric_label, rotation=270, labelpad=12, fontweight="bold", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

        # ==========================================
        # FALLBACK LAYOUT: N Summarizers, M LLMs
        # ==========================================
        else:
            fig, axes = plt.subplots(n_llms, n_summs, figsize=(3.5 * n_summs, 3 * n_llms), sharex=True, sharey=True)

            if n_llms == 1: axes = [axes]
            if n_summs == 1: axes = [[ax] for ax in axes]

            for i, llm in enumerate(llms):
                for j, summ in enumerate(summarizers):
                    ax = axes[i][j] if n_llms > 1 and n_summs > 1 else (axes[j] if n_llms == 1 else axes[i])

                    subset = df_agg[(df_agg["llm_name"] == llm) & (df_agg["summarizer_name"] == summ)]

                    sns.heatmap(
                        subset.pivot(index="article_section_idx", columns="position_idx", values=metric),
                        cmap=custom_cmap, annot=True, fmt=".2f", cbar=False,
                        vmin=global_min, vmax=global_max, ax=ax,
                        linewidths=0.5,
                        linecolor='silver'
                    )

                    if j == 0:
                        ax.set_ylabel("Article Section", fontweight="bold", fontsize=10)
                        ax.text(-0.45, 0.5, llm, transform=ax.transAxes, rotation=90, ha="center", va="center",
                                fontsize=10, fontweight="bold")
                    else:
                        ax.set_ylabel("")

                    if i == 0:
                        ax.set_title(summ, fontweight="bold", fontsize=10)
                    else:
                        ax.set_title("")

                    if i == n_llms - 1:
                        ax.set_xlabel("Position in Document", fontsize=10)
                    else:
                        ax.set_xlabel("")

            plt.tight_layout(rect=[0.05, 0, 0.9, 1])
            cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
            norm = mpl.colors.Normalize(vmin=global_min, vmax=global_max)
            sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(metric_label, rotation=270, labelpad=15, fontweight="bold")

        # Save
        os.makedirs(save_dir, exist_ok=True)

        # Save with transparent=False and facecolor set so the warm paper background saves to the files
        plt.savefig(os.path.join(save_dir, name) + ".pdf", bbox_inches="tight", facecolor=fig.get_facecolor(),
                    transparent=False)
        plt.savefig(os.path.join(save_dir, name) + ".png", bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor(),
                    transparent=False)
        plt.close()

    def skip_summary(self, summary: QueryBasedSummary):
        return summary.chunk_order is None


async def main():
    await DistributedUsageEvaluator(QueryBasedSummaryContext()).evaluate()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
