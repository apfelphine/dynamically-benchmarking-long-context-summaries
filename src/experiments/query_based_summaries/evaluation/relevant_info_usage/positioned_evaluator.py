import asyncio
import os

import numpy as np
import pandas as pd
from colorama import Fore
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm

from experiments.query_based_summaries.context import QueryBasedSummaryContext
from experiments.query_based_summaries.evaluation.relevant_info_usage.base_correct_doc_usage import CorrectDocUsageEvaluator
from models.corpus import QueryBasedSummary
from utils import vis
from utils.vis import bold, apply_name_mappings, wrap_text


class PositionedEvaluator(CorrectDocUsageEvaluator):
    # Does the model find relevant info when it is present at position "x"
    # Recall: "Of all the relevant info at this position, how much did the model find?"
    # Precision: "Of the things the model used at this position (if there was relevant info here), how much was actually relevant?"
    # Weight: How much of the info was at this position

    # Why not just Recall:
    # Would reward models that simply always use EVERYTHING

    # Why not just Precision:
    # Would reward models that use information very rarely, but when they do, are accurate

    def __init__(self, context: QueryBasedSummaryContext):
        super().__init__(context, "hidden_documents")
        self.context: QueryBasedSummaryContext = context  # Reassign for type hints

    async def evaluate_overall(self, overall_results, prepared_data):
        overall_results = [r for r in overall_results if not r["is_distributed"]]
        df = pd.DataFrame(overall_results)
        agg_rows = []

        metrics = list(self.metrics.keys())

        for (summ, llm, bin_idx, bin_name), group in df.groupby(
            ["summarizer_name", "llm_name", "bin_idx", "bin_name"]
        ):
            row = {
                "summarizer_name": summ,
                "llm_name": llm,
                "bin_idx": bin_idx,
                "bin_name": bin_name,
            }

            for p_id in range(self.N_PARTITIONS):
                dicts_for_pid = [p_list[p_id] for p_list in group["partition_data"]]
                weights = np.array([d["weight"] for d in dicts_for_pid])

                for m in metrics:
                    vals = np.array([d[m] if m in d else np.nan for d in dicts_for_pid])

                    # Calculate Weighted Average
                    valid_mask = pd.notna(vals) & pd.notna(weights) & (weights > 0)

                    if np.any(valid_mask):
                        val = np.average(vals[valid_mask], weights=weights[valid_mask])
                    else:
                        val = np.nan

                    row[f"{m}_partition_{p_id}"] = val

            agg_rows.append(row)

        if not agg_rows:
            print("No data found for partition evaluation.")
            return

        df_agg = pd.DataFrame(agg_rows)
        melted_df = self._melt_partition_data(df_agg)

        # Save aggregated metrics
        df_agg.to_csv(
            os.path.join(self.overall_analysis_dir, "partition_metrics.csv"),
            index=False,
        )
        melted_df.to_csv(
            os.path.join(self.overall_analysis_dir, "partition_metrics_melted.csv"),
            index=False,
        )
        melted_df.groupby(["summarizer_name", "llm_name"]).agg(
            {m: "mean" for m in metrics}
        ).reset_index().to_csv(
            os.path.join(self.overall_analysis_dir, "partition_metrics_aggregated.csv"),
            index=False,
        )

        melted_df = apply_name_mappings(melted_df, rename_columns=False)

        for m in metrics:
            self.plot_partition_metric(
                melted_df,
                metric_name=m,
                save_path=os.path.join(
                    self.overall_analysis_dir, f"partition_{m}_ci"
                ),
                N_PARTITIONS=self.N_PARTITIONS,
                errorbar=("ci", 95),
            )
            self.plot_partition_metric(
                melted_df,
                metric_name=m,
                save_path=os.path.join(self.overall_analysis_dir, f"partition_{m}"),
                N_PARTITIONS=self.N_PARTITIONS,
                errorbar=None,
            )

        self.print_metrics(melted_df, ["summarizer_name", "llm_name"], "Overview")
        self.print_metrics(
            melted_df, ["summarizer_name"], "Analysis per Summary Strategy"
        )
        self.print_metrics(melted_df, ["llm_name"], "Analysis per LLM")
        self.print_metrics(melted_df, ["bin_name"], "Analyse per Bin")


    def _melt_partition_data(self, df):
        metrics = self.metrics.keys()

        df_long = pd.wide_to_long(
            df,
            stubnames=metrics,
            i=["summarizer_name", "llm_name", "bin_idx", "bin_name"],
            j="partition_idx",
            sep="_partition_",
            suffix=r"\d+",
        ).reset_index()

        # Sort if necessary for cleaner viewing
        df_long = df_long.sort_values(
            ["summarizer_name", "llm_name", "bin_idx", "partition_idx"]
        )

        return df_long

    def plot_partition_metric(
        self,
        df_long: pd.DataFrame,
        metric_name: str,
        save_path: str,
        N_PARTITIONS: int,
        errorbar,
    ):
        """
        Plots partition data for a specific metric using Seaborn (Simpler version).
        """
        df_long = df_long.copy()
        df_long["partition_idx"] = df_long["partition_idx"] + 1

        metric_label = self.metrics[metric_name]

        # Dynamically determine plot size based on number of summarizers
        num_summarizers = df_long["summarizer_name"].nunique()
        if num_summarizers == 1:
            plot_aspect = 1.5
            plot_height = 3.45 / plot_aspect
        else:
            plot_height = 5.0
            plot_aspect = 1.3

        g = sns.relplot(
            data=df_long,
            x="partition_idx",
            y=metric_name,
            col="summarizer_name",
            hue="llm_name",
            style="llm_name",
            kind="line",
            col_wrap=max(1, num_summarizers),
            height=plot_height,
            aspect=plot_aspect,
            markers=True,
            dashes=False,
            palette="paper_models",
            linewidth=2.5,
            facet_kws={"sharey": True, "sharex": True},
            errorbar=errorbar,
        )

        g.set_axis_labels(f"Bucket (1-{N_PARTITIONS})", metric_label)
        custom_ticks = list(range(5, df_long["partition_idx"].nunique() + 1, 5))
        custom_ticks = [1] + custom_ticks
        g.set(xticks=custom_ticks)

        if num_summarizers > 1:
            g.set_titles("Summarization Strategy: {col_name}")
        else:
            g.set_titles("")

        g.set(xlim=(0, N_PARTITIONS + 1), ylim=(0.0, 1.05))

        for ax in g.axes.flat:
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        if g.legend:
            handles = g.legend.legend_handles
            labels = [t.get_text() for t in g.legend.texts]
            if "llm_name" in labels:
                idx = labels.index("llm_name")
                del labels[idx]
                del handles[idx]

            g.legend.remove()

            # Standard tight_layout without hardcoded rect restrictions
            g.figure.tight_layout()

            # Add custom legend OUTSIDE the bottom of the figure
            g.figure.legend(
                handles=handles,
                labels=labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0),
                ncol=len(labels) if num_summarizers > 1 else len(labels) // 2,
                frameon=False,
                fontsize=12,
            )
        else:
            g.figure.tight_layout()

        # bbox_inches="tight" is required alongside pad_inches so the outer legend isn't cut off
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close()

    def skip_summary(self, summary: QueryBasedSummary):
        return summary.chunk_order is not None

async def main():
    ctx = QueryBasedSummaryContext()

    print(f"{Fore.MAGENTA}>> Evaluation: Strict Content")
    await PositionedEvaluator(ctx).evaluate()



if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
