import asyncio
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from colorama import Fore, Style

from experiments.generic_summaries.context import GenericSummaryContext
from experiments.generic_summaries.evaluation.bls.base_evaluator import BaseRegressionEvaluator
from utils.vis import apply_name_mappings


class OLSEvaluator(BaseRegressionEvaluator):
    """
    Standard Ordinary Least Squares Evaluator.

    OLS METRICS EXPLANATION:
    ------------------------
    ABSOLUTE VARIANCE (R^2) - Based on total LLM behavior (including randomness/noise):
    * r2_total: How much of the LLM's chunk utilization we can predict using BOTH the article and its position.
    * r2_art:   Variance driven strictly by the Article itself (absorbs chunk-length, topic, inherent relevance).
    * r2_pos:   Variance driven strictly by Position (the pure sequence bias, completely isolated from length).

    RELATIVE SHARE - Based ONLY on the behavior we can explain (r2_art + r2_pos = 1.0):
    * share_art (r2_art / r2_total): The % of explained behavior driven by content/length. (Good: should be high).
    * share_pos (r2_pos / r2_total): The % of explained behavior driven purely by position. (Bad: tracks positional bias).

    Note: This evaluator calculates two variations of the metrics to isolate the length confounder:
    * mean_* metrics: Target is continuous fraction of chunks used (favors shorter articles).
    * sum_* metrics: Target is absolute number of chunks used (favors longer articles).
    """

    def __init__(
        self,
        context: GenericSummaryContext,
    ):
        super().__init__(context, "ordinary_least_squares", use_relative_bucketing=False)

    @staticmethod
    def _run_regression_for_target(
        y_target: np.ndarray, a_idx_arr: np.ndarray, p_idx_arr: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, float], np.ndarray]:
        def onehot(idx, drop_first=True):
            u, inv = np.unique(idx, return_inverse=True)
            H = np.eye(len(u))[inv]
            return H[:, 1:] if drop_first else H

        A_art = onehot(a_idx_arr)
        A_pos = onehot(p_idx_arr)

        def r2_of(design):
            A = np.hstack([np.ones((len(y_target), 1)), design])
            beta, *_ = np.linalg.lstsq(A, y_target, rcond=None)
            resid = y_target - A @ beta
            y_var = ((y_target - y_target.mean()) ** 2).sum()
            if y_var == 0:
                return 0.0
            return 1 - (resid ** 2).sum() / y_var

        r2_pos = r2_of(A_pos)
        r2_art = r2_of(A_art)
        r2_full = r2_of(np.hstack([A_art, A_pos]))

        def partial_r2(r2_f, r2_r):
            denominator = 1 - r2_r
            return (r2_f - r2_r) / denominator if denominator != 0 else 0.0

        share_pos = partial_r2(r2_full, r2_art)
        share_art = partial_r2(r2_full, r2_pos)

        A_full = np.hstack([np.ones((len(y_target), 1)), A_art, A_pos])
        beta, *_ = np.linalg.lstsq(A_full, y_target, rcond=None)
        p0 = 1 + A_art.shape[1]
        position_curve = np.concatenate([[0.0], beta[p0:p0 + A_pos.shape[1]]])
        position_curve -= position_curve.mean()

        return {
            "r2_pos": r2_pos, "r2_art": r2_art, "r2_full": r2_full,
            "share_pos": share_pos, "share_art": share_art,
        }, position_curve

    async def evaluate_overall(self, overall_results, prepared_data):
        df_results = pd.DataFrame(overall_results)
        stats = []

        for (summarizer, llm), group in df_results.groupby(["summarizer_name", "llm_name"]):
            cells = self._extract_global_cells(group)
            if len(cells) == 0:
                continue

            stats_mean, curve_mean = self._run_regression_for_target(cells[:, 2], cells[:, 0].astype(int),
                                                                     cells[:, 1].astype(int))
            stats_sum, curve_sum = self._run_regression_for_target(cells[:, 3], cells[:, 0].astype(int),
                                                                   cells[:, 1].astype(int))

            stats.append({
                "summarizer_name": summarizer, "llm_name": llm,
                "mean_r2_pos": stats_mean["r2_pos"], "mean_r2_art": stats_mean["r2_art"],
                "mean_r2_full": stats_mean["r2_full"],
                "mean_share_pos": stats_mean["share_pos"], "mean_share_art": stats_mean["share_art"],
                "sum_r2_pos": stats_sum["r2_pos"], "sum_r2_art": stats_sum["r2_art"],
                "sum_r2_full": stats_sum["r2_full"],
                "sum_share_pos": stats_sum["share_pos"], "sum_share_art": stats_sum["share_art"],
            })
            self._save_model_artifacts(summarizer, llm, curve_mean, curve_sum, prefix="ols")

        stats_df = pd.DataFrame(stats)
        apply_name_mappings(stats_df, rename_columns=False)
        stats_df.to_csv(os.path.join(self.overall_analysis_dir, "ols_stats_dual_global.csv"), index=False)
        self._save_global_comparison_plot(stats_df, prefix="ols")

        print(f"\n{Fore.CYAN}=== Cleaned Dual OLS Evaluator Results (GLOBAL) ==={Style.RESET_ALL}")

        print(stats_df.to_string(index=False))

async def main():
    ctx = GenericSummaryContext()
    evaluator = OLSEvaluator(ctx)
    await evaluator.evaluate()


if __name__ == "__main__":
    asyncio.run(main())