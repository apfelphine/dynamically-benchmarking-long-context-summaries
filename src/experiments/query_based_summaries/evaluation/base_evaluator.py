from abc import ABC

import numpy as np
from experiments.experiment_evaluator import ExperimentEvaluator


class QueryBasedSummaryBaseEvaluator(ExperimentEvaluator, ABC):
    def __init__(self, context, base_eval_dir: str):
        super().__init__(context, base_eval_dir)

    @staticmethod
    def get_focus_indices(n_remaining, n_focus_chunks):
        if n_focus_chunks == 0:
            return []
        step = n_remaining / (n_focus_chunks - 1)
        indices = []
        for i in range(n_focus_chunks):
            indices.append(i + int(i * step))
        return indices

    async def get_used_chunk_indices(self, summary, summarizer_name, llm_name):
        _, _, binary_matrix, _ = await self._load_llm_data(
            summary, summarizer_name, llm_name
        )
        chunk_usage = binary_matrix.any(axis=1).astype(int)
        indices = np.where(chunk_usage == 1)[0]
        return indices.tolist()
