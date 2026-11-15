import asyncio
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from colorama import Fore, Style

from experiments.generic_summaries.context import GenericSummaryContext
from experiments.generic_summaries.evaluation.bls.base_evaluator import BaseRegressionEvaluator
from utils.vis import apply_name_mappings


class WLSEvaluator(BaseRegressionEvaluator):
    """Weighted Least Squares Evaluator (Accounts for uneven bin lengths)."""

    def __init__(
        self,
        context: GenericSummaryContext,
    ):
        super().__init__(context, "weighted_least_squares")

    @staticmethod
    def _calculate_position_weights(p_idx_arr: np.ndarray, bin_idx_arr: np.ndarray) -> np.ndarray:
        pos_to_bin_count = {}
        for pos in np.unique(p_idx_arr):
            pos_to_bin_count[pos] = len(np.unique(bin_idx_arr[p_idx_arr == pos]))
        weights = np.array([pos_to_bin_count[p] for p in p_idx_arr], dtype=float)
        if weights.mean() > 0:
            weights /= weights.mean()
        return weights

    @staticmethod
    def _run_regression_for_target(
        y_target: np.ndarray, a_idx_arr: np.ndarray, p_idx_arr: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, float], np.ndarray]:
        weights = kwargs.get("weights")

        def onehot(idx, drop_first=True):
            u, inv = np.unique(idx, return_inverse=True)
            H = np.eye(len(u))[inv]
            return H[:, 1:] if drop_first else H

        A_art = onehot(a_idx_arr)
        A_pos = onehot(p_idx_arr)

        def weighted_r2_of(design):
            X = np.hstack([np.ones((len(y_target), 1)), design])
            sqrt_w = np.sqrt(weights)[:, np.newaxis]
            X_w, y_w = X * sqrt_w, y_target * np.sqrt(weights)
            beta, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
            resid_w = y_w - X_w @ beta
            w_mean = np.sum(weights * y_target) / np.sum(weights)
            y_var_w = np.sum(weights * (y_target - w_mean) ** 2)
            if y_var_w == 0:
                return 0.0
            return 1 - np.sum(resid_w ** 2) / y_var_w

        r2_pos = weighted_r2_of(A_pos)
        r2_art = weighted_r2_of(A_art)
        r2_full = weighted_r2_of(np.hstack([A_art, A_pos]))

        def partial_r2(r2_f, r2_r):
            denominator = 1 - r2_r
            return (r2_f - r2_r) / denominator if denominator != 0 else 0.0

        share_pos = partial_r2(r2_full, r2_art)
        share_art = partial_r2(r2_full, r2_pos)

        X_full = np.hstack([np.ones((len(y_target), 1)), A_art, A_pos])
        sqrt_w = np.sqrt(weights)[:, np.newaxis]
        beta, *_ = np.linalg.lstsq(X_full * sqrt_w, y_target * np.sqrt(weights), rcond=None)

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

            a_idx_arr = cells[:, 0].astype(int)
            p_idx_arr = cells[:, 1].astype(int)
            y_mean = cells[:, 2]
            y_sum = cells[:, 3]
            bin_idx_arr = cells[:, 4].astype(int)

            weights = self._calculate_position_weights(p_idx_arr, bin_idx_arr)

            stats_mean, curve_mean = self._run_regression_for_target(y_mean, a_idx_arr, p_idx_arr, weights=weights)
            stats_sum, curve_sum = self._run_regression_for_target(y_sum, a_idx_arr, p_idx_arr, weights=weights)

            stats.append({
                "summarizer_name": summarizer, "llm_name": llm,
                "mean_r2_pos": stats_mean["r2_pos"], "mean_r2_art": stats_mean["r2_art"],
                "mean_r2_full": stats_mean["r2_full"],
                "mean_share_pos": stats_mean["share_pos"], "mean_share_art": stats_mean["share_art"],
                "sum_r2_pos": stats_sum["r2_pos"], "sum_r2_art": stats_sum["r2_art"],
                "sum_r2_full": stats_sum["r2_full"],
                "sum_share_pos": stats_sum["share_pos"], "sum_share_art": stats_sum["share_art"],
            })
            self._save_model_artifacts(summarizer, llm, curve_mean, curve_sum, prefix="wls")

        stats_df = pd.DataFrame(stats)
        apply_name_mappings(stats_df, rename_columns=False)
        stats_df.to_csv(os.path.join(self.overall_analysis_dir, "wls_stats_dual_global.csv"), index=False)
        self._save_global_comparison_plot(stats_df, prefix="wls")

        print(f"\n{Fore.CYAN}=== Cleaned Dual WLS Evaluator Results (GLOBAL) ==={Style.RESET_ALL}")
        print(stats_df.to_string(index=False))


async def main():
    ctx = GenericSummaryContext()
    evaluator = WLSEvaluator(ctx)
    await evaluator.evaluate()


if __name__ == "__main__":
    asyncio.run(main())