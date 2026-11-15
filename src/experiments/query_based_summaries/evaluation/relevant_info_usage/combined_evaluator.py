import asyncio
import os

import pandas as pd
from colorama import Fore

from experiments.query_based_summaries.context import QueryBasedSummaryContext
from experiments.query_based_summaries.evaluation.relevant_info_usage.base_correct_doc_usage import CorrectDocUsageEvaluator
from utils.pretty_table import pretty_table
from utils.vis import apply_name_mappings


class CombinedUsageEvaluator(CorrectDocUsageEvaluator):
    def __init__(self, context: QueryBasedSummaryContext):
        super().__init__(context, "combined")
        self.context: QueryBasedSummaryContext = context

    async def evaluate_overall(self, overall_results, prepared_data):
        df = pd.DataFrame(overall_results)
        os.makedirs(self.overall_analysis_dir, exist_ok=True)

        df["found"] = (df["recall_raw"] > 0).astype(int)

        df.to_csv(
            os.path.join(self.overall_analysis_dir, "metrics_raw.csv"), index=False
        )

        cols = [
            "precision_raw",
            "recall_raw",
            "f1_raw",
            "found",
        ]
        grouped = df.groupby(["summarizer_name", "llm_name"])[cols].mean().reset_index()

        plot_df = grouped.copy()
        apply_name_mappings(plot_df, rename_columns=False)

        title =  "Content Usage"

        config = [
            {"col": "precision_raw", "header": "P (Raw)", "high_is_good": True},
            {"col": "recall_raw", "header": "R (Raw)", "high_is_good": True},
            {"col": "f1_raw", "header": "F1 (Raw)", "high_is_good": True},
            {"col": "found", "header": "% Found", "high_is_good": True, "decimals": 4},
        ]
        pretty_table(
            df[df["is_distributed"] == False],
            ["summarizer_name", "llm_name"],
            config,
            title + " - Target hidden in one piece",
        )
        pretty_table(
            df[df["is_distributed"] == True],
            ["summarizer_name", "llm_name"],
            config,
            title + " - Scattered",
        )
        pretty_table(
            df,
            ["summarizer_name", "llm_name"],
            config,
            title + " - All",
        )
        pretty_table(
            df,
            ["summarizer_name", "llm_name", "is_distributed"],
            config,
            title + " - Comparison",
        )


async def main():
    print(f"{Fore.MAGENTA}>> Evaluation: Strict Content Usage")
    await CombinedUsageEvaluator(
        QueryBasedSummaryContext(),
    ).evaluate()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
