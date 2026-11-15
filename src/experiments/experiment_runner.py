import asyncio
import datetime
import os
import time
from typing import Dict
from tqdm import tqdm as std_tqdm
import aiofiles
from colorama import Fore
from core.summarization.base_summarizer import BaseSummarizer
from core.llms.base_llm import BaseLLM
from experiments.base_experiment_context import ExperimentContext
from models.corpus import Bin, Summary
from utils.async_json import json_dump


class ExperimentRunner:
    """
    Orchestrates the running of summarization experiments based on a provided context.
    """

    def __init__(self, context: ExperimentContext, n_parallel=4):
        self.context = context
        self.n_parallel = n_parallel

    async def _process_single_summary(
        self,
        s: Summary,
        b: Bin,
        summarizer: BaseSummarizer,
        summarizer_name: str,
        llm_name: str,
        semaphore: asyncio.Semaphore,
        pbar: std_tqdm,
    ):
        """
        Internal method to process a single summary task.
        """
        # Delegate path resolution and file reading to the context
        save_path = self.context.get_save_path(s)
        os.makedirs(save_path, exist_ok=True)

        # Structure: <save_path>/<summarizer_name>/<llm_name>/
        summary_dir = os.path.join(save_path, summarizer_name, llm_name)
        summary_file = os.path.join(summary_dir, "summary.md")
        metadata_file = os.path.join(summary_dir, "metadata.json")

        # Check if already exists (resume capability)
        if os.path.exists(summary_file) and os.path.exists(metadata_file):
            pbar.update(1)
            return

        # Perform Summarization
        try:
            start_time = time.perf_counter()
            async with semaphore:
                texts_in_order, focus_doc_title = await self.context.read_files(b, s)
                summary = await summarizer.asummarize_multi(
                    texts_in_order, focus_doc_title
                )
            end_time = time.perf_counter()

            os.makedirs(summary_dir, exist_ok=True)
            async with aiofiles.open(summary_file, "w", encoding="utf-8") as f:
                await f.write(summary)

            await json_dump(
                metadata_file,
                {
                    "duration": end_time - start_time,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
            )

        except Exception as e:
            if not hasattr(pbar, "failed_cnt"):
                pbar.failed_cnt = 0
            pbar.failed_cnt += 1
            pbar.set_postfix(failed=pbar.failed_cnt)
            std_tqdm.write(
                f"{Fore.RED}[ERROR] Bin {s.bin_idx} | {llm_name} - {summarizer_name}: {e} ({type(e).__name__})"
            )
            if isinstance(e, PermissionError):
                exit(-1)
        finally:
            pbar.update(1)

    async def _process_llm_chain(
        self,
        llm_name: str,
        llm_instance: BaseLLM,
        pbars: Dict[str, std_tqdm],
    ):
        """
        Orchestrates the processing for a specific LLM across all summarizers and summaries.
        """
        # Limit concurrency per LLM to prevent rate limits
        sem = asyncio.Semaphore(self.n_parallel)

        tasks = []
        for summarizer_name, summarizer_cls in self.context.summarizers.items():
            summarizer = summarizer_cls(llm_instance)
            key = f"{llm_name}-{summarizer_name}"
            current_pbar = pbars[key]

            for s in self.context.summaries:
                task = self._process_single_summary(
                    s=s,
                    b=self.context.corpus[s.bin_idx],
                    summarizer=summarizer,
                    summarizer_name=summarizer_name,
                    llm_name=llm_name,
                    semaphore=sem,
                    pbar=current_pbar,
                )
                tasks.append(task)

        await asyncio.gather(*tasks)


    async def run(self, sequential: bool = False):
        print(f"{Fore.CYAN}Initializing Progress Bars...")

        # Create Progress Bars
        pbars = {}
        bar_position = 0
        total_tasks = len(self.context.summaries)
        bar_colors = ["green", "blue", "magenta", "cyan", "yellow", "red", "white"]

        for llm_name in self.context.llms.keys():
            for summarizer_name in self.context.summarizers.keys():
                key = f"{llm_name}-{summarizer_name}"
                desc = f"{llm_name} | {summarizer_name}"
                current_color = bar_colors[bar_position % len(bar_colors)]

                pbars[key] = std_tqdm(
                    total=total_tasks,
                    desc=desc,
                    leave=True,
                    colour=current_color,
                    position=bar_position,
                    mininterval=0,
                )
                bar_position += 1

        # Launch Tasks
        try:
            if sequential:
                # Process sequentially: await each LLM chain before moving to the next
                for llm_name, llm_inst in self.context.llms.items():
                    await self._process_llm_chain(
                        llm_name=llm_name,
                        llm_instance=llm_inst,
                        pbars=pbars,
                    )
            else:
                # Process in parallel: gather all tasks
                llm_tasks = []
                for llm_name, llm_inst in self.context.llms.items():
                    llm_tasks.append(
                        self._process_llm_chain(
                            llm_name=llm_name,
                            llm_instance=llm_inst,
                            pbars=pbars,
                        )
                    )

                await asyncio.gather(*llm_tasks)
        finally:
            for pbar in pbars.values():
                pbar.close()

        print(f"\n{Fore.YELLOW}--- Experiment Finished ---")
