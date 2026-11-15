import os
import numpy as np
import aiofiles
from typing import List
from tqdm import tqdm as std_tqdm
from colorama import Fore

from experiments.base_experiment_context import ExperimentContext
from models.corpus import Summary
from models.document_splitting import Chunk
from utils.async_json import json_dump, json_load


class ExperimentAligner:
    """
    Handles the alignment of generated summaries against source documents using the resources provided in the context.
    """

    def __init__(
        self, context: ExperimentContext, enable_embedding=True, enable_llm=True
    ):
        self.context = context
        self.enable_embedding = enable_embedding
        self.enable_llm = enable_llm

    @staticmethod
    def shorten(text: str, n: int = 15, add_closing_bracket=False) -> str:
        if len(text) <= n:
            res = text.ljust(n)
        else:
            res = text[: n - 3] + "..." if n >= 3 else text[:n]

        return res + ")" if add_closing_bracket else res

    async def _get_doc_chunks(self, save_path: str) -> List[Chunk]:
        """Loads or creates document chunks from documents.json"""
        chunk_file = os.path.join(save_path, "document_chunks.json")
        if os.path.exists(chunk_file):
            return [Chunk(**c) for c in await json_load(chunk_file)]

        doc_file = os.path.join(save_path, "documents.json")
        if not os.path.exists(doc_file):
            raise Exception(f"Document file {doc_file} does not exist")

        texts = await json_load(doc_file)
        chunks = []
        for idx, text in enumerate(texts):
            # Using chunker from context
            splits = await self.context.chunker.split(text)
            for c in splits:
                c_data = c.model_dump(exclude_unset=True)
                c_data["document_idx"] = idx
                chunks.append(Chunk(**c_data))

        await json_dump(chunk_file, [c.model_dump() for c in chunks])
        return chunks

    async def _align_single_target(
        self,
        target_chunks: List[Chunk],
        doc_chunks: List[Chunk],
        output_dir: str,
    ):
        emb_file = os.path.join(output_dir, "sim_matrix_embedding.npy")
        llm_file = os.path.join(output_dir, "sim_matrix.npy")

        if os.path.exists(llm_file):
            return

        try:
            if os.path.exists(emb_file):
                emb_matrix = np.load(emb_file)
            elif self.enable_embedding:
                # Using comparer from context
                emb_matrix = await self.context.comparer.execute_first_comparer(
                    doc_chunks, target_chunks
                )
                np.save(emb_file, emb_matrix)
            else:
                return

            if self.enable_llm:
                # Using comparer from context
                matrix = await self.context.comparer.execute_second_comparer(
                    doc_chunks, target_chunks, emb_matrix
                )
                np.save(llm_file, matrix)
        except Exception as e:
            std_tqdm.write(f"{Fore.RED}[Alignment Failed] {output_dir}: {e}")

    async def _align_single_summary(self, run_path: str, doc_chunks: List[Chunk]):
        summ_file = os.path.join(run_path, "summary.md")
        if not os.path.exists(summ_file):
            return

        chunk_file = os.path.join(run_path, "summary_chunks.json")
        if os.path.exists(chunk_file):
            summ_chunks = [Chunk(**c) for c in await json_load(chunk_file)]
        else:
            async with aiofiles.open(summ_file, "r", encoding="utf-8") as f:
                content = await f.read()
            # Using chunker from context
            summ_chunks = await self.context.chunker.split(content)
            await json_dump(chunk_file, [c.model_dump() for c in summ_chunks])

        await self._align_single_target(summ_chunks, doc_chunks, run_path)

    async def align(self):
        print(f"{Fore.MAGENTA}Starting Alignment...")
        corpus_map = {idx: b for idx, b in enumerate(self.context.corpus)}

        desc = "Aligning Summaries"
        with std_tqdm(total=len(self.context.summaries), desc=desc) as pbar:
            for s in self.context.summaries:
                save_path = self.context.get_save_path(s)
                if not os.path.exists(save_path):
                    pbar.update(1)
                    continue

                bin_name = f"Bin {s.bin_idx}"
                summary_desc = f"{desc} | {bin_name}"

                doc_chunks = await self._get_doc_chunks(save_path)
                if not doc_chunks:
                    raise Exception("Could not get document chunks")

                if hasattr(s, "focus_doc_idx"):
                    focus_name = corpus_map[s.bin_idx].document_names[s.focus_doc_idx]
                    summary_desc = f"{summary_desc} | {self.shorten(focus_name)}"

                    await self.align_hook(
                        s,
                        save_path,
                        doc_chunks,
                    )
                else:
                    first_name = corpus_map[s.bin_idx].document_names[
                        s.document_order[0]
                    ]
                    summary_desc = f"{summary_desc} | First: {self.shorten(first_name)}"

                # Align Generated Summaries
                for summ_name in os.listdir(save_path):
                    s_dir = os.path.join(save_path, summ_name)
                    if not summ_name in self.context.summarizers.keys():
                        continue
                    if not os.path.isdir(s_dir) or summ_name in [
                        "partitioned",
                        "positioned",
                    ]:
                        continue

                    for llm_name in os.listdir(s_dir):
                        if not llm_name in self.context.llms.keys():
                            continue
                        run_dir = os.path.join(s_dir, llm_name)
                        if not os.path.isdir(run_dir):
                            continue
                        pbar.set_description(
                            f"{summary_desc} ({self.shorten(summ_name)} | {self.shorten(llm_name, add_closing_bracket=True)}"
                        )

                        await self._align_single_summary(run_dir, doc_chunks)

                pbar.update(1)
        print(f"{Fore.GREEN}Alignment Complete.")

    async def align_hook(
        self, summary: Summary, save_path: str, doc_chunks: List[Chunk]
    ):
        """Optional hook for specific alignment logic (can be overridden)."""
        pass
