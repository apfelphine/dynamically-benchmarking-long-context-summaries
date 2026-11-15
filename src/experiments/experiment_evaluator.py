import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

import aiofiles
import numpy as np
import tiktoken
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from experiments.base_experiment_context import ExperimentContext
from models.corpus import Bin, Summary
from utils import vis
from utils.alignment import make_binary
from utils.async_json import json_load

vis.initialize()


class ExperimentEvaluator(ABC):
    ENC = tiktoken.get_encoding("cl100k_base")

    def __init__(self, context: ExperimentContext, base_eval_dir: str):
        self.context = context

        self.evaluation_dir = os.path.join(
            self.context.base_evaluations_dir, base_eval_dir
        )

        self.overall_analysis_dir = os.path.join(self.evaluation_dir, "overall")
        self.bin_analysis_dir = os.path.join(self.evaluation_dir, "per_bin")

        os.makedirs(self.evaluation_dir, exist_ok=True)
        os.makedirs(self.overall_analysis_dir, exist_ok=True)
        os.makedirs(self.bin_analysis_dir, exist_ok=True)

    @abstractmethod
    async def evaluate_single_summary(
        self,
        b: Bin,
        summary_idx: str,
        summary: Summary,
        data: Dict,
        summarizer_name: str,
        llm_name: str,
    ):
        pass

    async def evaluate_bin(
        self,
        bin_idx: int,
        b: Bin,
        summarizer_name: str,
        llm_name: str,
        bin_results: List[Dict],
        prepared_data,
    ):
        # Default Implementation - Do nothing
        return bin_results

    async def evaluate_llm_summarizer( # noqa
        self, summarizer_llm_results, summarizer_name: str, llm_name: str # noqa
    ):
        # Default Implementation - Do nothing
        return summarizer_llm_results

    @abstractmethod
    async def evaluate_overall(self, overall_results, prepared_data):
        pass

    @abstractmethod
    async def prepare_data(self):
        pass

    async def collect_available_summaries(
        self, context: Optional[ExperimentContext] = None
    ):
        if not context:
            context = self.context
        pair_to_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)

        for idx, summary in enumerate(
            context.summaries,
        ):
            summary_path = context.get_save_path(summary)
            if not os.path.isdir(summary_path):
                continue
            for summarizer in os.listdir(summary_path):
                if not summarizer in context.summarizers.keys():
                    continue
                summarizer_path = os.path.join(summary_path, summarizer)
                if not os.path.isdir(summarizer_path):
                    continue
                for llm in os.listdir(summarizer_path):
                    if not llm in context.llms.keys():
                        continue

                    if os.path.isdir(os.path.join(summarizer_path, llm)):
                        if os.path.isfile(os.path.join(summarizer_path, llm, "sim_matrix.npy")):
                            pair_to_indices[(summarizer, llm)].append(idx)

        return pair_to_indices

    async def evaluate(self):
        prepared_data = await self.prepare_data()
        pair_to_indices = await self.collect_available_summaries()

        overall_results = []

        for (summarizer, llm), summary_indices in tqdm(
            pair_to_indices.items(), desc="Processing summaries...", position=0
        ):
            summarizer_llm_results = []

            bin_to_data_indices = defaultdict(list)
            for s_idx in summary_indices:
                summary = self.context.summaries[s_idx]
                bin_to_data_indices[summary.bin_idx].append(s_idx)

            for bin_idx, s_indices in tqdm(
                bin_to_data_indices.items(),
                desc=f"  > {summarizer} - {llm}",
                leave=False,
                position=1,
            ):
                b = self.context.corpus[bin_idx]

                tasks = []
                for s_idx in s_indices:
                    summary = self.context.summaries[s_idx]
                    tasks.append(
                        self.evaluate_single_summary(
                            b, s_idx, summary, prepared_data, summarizer, llm
                        )
                    )

                bin_results = await tqdm_asyncio.gather(
                    *tasks, desc=f"    > Bin: {b.name}", leave=False, position=2
                )

                summarizer_llm_results.extend(
                    await self.evaluate_bin(
                        bin_idx, b, summarizer, llm, bin_results, prepared_data
                    )
                )

            overall_results.extend(
                await self.evaluate_llm_summarizer(
                    summarizer_llm_results, summarizer, llm
                )
            )

        await self.evaluate_overall(overall_results, prepared_data)

        if not os.listdir(self.overall_analysis_dir):
            os.rmdir(self.overall_analysis_dir)
        if not os.listdir(self.bin_analysis_dir):
            os.rmdir(self.bin_analysis_dir)

    async def _load_llm_data(
        self, summary, summarizer_name, llm_name, threshold: int = 0
    ):
        """Helper to load text and similarity matrix."""
        llm_path = os.path.join(
            self.context.get_save_path(summary), summarizer_name, llm_name
        )
        async with aiofiles.open(
            os.path.join(llm_path, "summary.md"), "r", encoding="utf-8"
        ) as f:
            text = await f.read()
        try:
            metadata = await json_load(os.path.join(llm_path, "metadata.json"))
            sim_matrix = np.load(os.path.join(llm_path, "sim_matrix.npy"))
            return text, metadata, make_binary(sim_matrix, threshold=threshold), sim_matrix
        except:
            print(f"Failed to load for {llm_path}")
            raise
