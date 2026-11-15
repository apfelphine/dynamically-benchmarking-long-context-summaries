import hashlib
import os
import time
from pathlib import Path
from typing import Callable, List, Tuple, Dict
import numpy as np
from core.document_alignment.base_document_aligner import BaseDocumentAligner
from core.document_splitting.spacy_sentence_splitter import SpacySentenceSplitter
from core.embedding.base_embedding import BaseEmbedding
from models.document_splitting import Chunk
from utils.prompts import load_jinja_prompt_template

CACHE_DIR = Path("cache/embeddings")
os.makedirs(CACHE_DIR, exist_ok=True)


class EmbeddingAligner(BaseDocumentAligner):
    def __init__(
        self,
        embedding: BaseEmbedding,
        similarity: str | Callable[[np.ndarray, np.ndarray], float] = "cosine",
        split_summary: bool = False,
    ):
        self.embedding = embedding
        self.split_summary = split_summary
        self.similarity_fn = self._get_similarity_fn(similarity)

        self.passage_template = load_jinja_prompt_template(
            "prompts/document_alignment/embedding/passage.j2"
        )
        self.query_template = load_jinja_prompt_template(
            "prompts/document_alignment/embedding/query.j2"
        )

        self.sentence_splitter = SpacySentenceSplitter()
        self.prune_cache()

    async def _split_into_sentences(self, chunk: Chunk) -> List[str]:
        if self.split_summary:
            lines = chunk.text.splitlines()
            headers = ""
            remaining_text = ""
            for line in lines:
                if line.startswith("#"):
                    headers += line + "\n"
                else:
                    remaining_text += line + "\n"
            return [
                headers + c.text
                for c in await self.sentence_splitter.split(remaining_text)
            ]
        return [chunk.text]

    async def _continue_similarity_matrix(
        self,
        document_chunks: List[Chunk],
        summary_chunks: List[Chunk],
        matrix: np.ndarray,
    ) -> np.ndarray:
        doc_texts = [self.__prepare_document_chunk(c) for c in document_chunks]
        doc_embs = await self._get_embeddings(doc_texts)
        all_summ_inputs: List[str] = []
        summ_map: Dict[int, Tuple[int, int]] = {}

        start_idx = 0
        for j, summ_chunk in enumerate(summary_chunks):
            if self.split_summary:
                sentences = self._split_into_sentences(summ_chunk)
                prepared_sentences = [
                    self.__prepare_summary_text(s) for s in await sentences
                ]
                all_summ_inputs.extend(prepared_sentences)
                end_idx = start_idx + len(prepared_sentences)
                summ_map[j] = (start_idx, end_idx)
                start_idx = end_idx
            else:
                prepared_chunk = self.__prepare_summary_text(summ_chunk.text)
                all_summ_inputs.append(prepared_chunk)
                summ_map[j] = (start_idx, start_idx + 1)
                start_idx += 1

        all_summ_embs = await self._get_embeddings(all_summ_inputs)

        for i, _ in enumerate(document_chunks):
            doc_emb = doc_embs[i]

            for j, _ in enumerate(summary_chunks):
                if np.isnan(matrix[i, j]):
                    start, end = summ_map[j]

                    summ_embs_for_chunk = all_summ_embs[start:end]
                    if not summ_embs_for_chunk:
                        matrix[i, j] = 0
                        continue

                    scores = [
                        self.similarity_fn(doc_emb, s_emb)
                        for s_emb in summ_embs_for_chunk
                    ]

                    matrix[i, j] = max(scores) if scores else 0

        return matrix

    @staticmethod
    def _get_similarity_fn(
        similarity: str | Callable[[np.ndarray, np.ndarray], float],
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        if callable(similarity):
            return similarity
        elif similarity == "cosine":
            return lambda a, b: float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            )
        elif similarity == "dot":
            return lambda a, b: float(np.dot(a, b))
        elif similarity == "euclidean":
            return lambda a, b: 1 / (1 + np.linalg.norm(a - b))
        else:
            raise ValueError(f"Unknown similarity function: {similarity}")

    async def _get_embeddings(
        self, texts: List[str], with_cache=False
    ) -> List[np.ndarray]:
        results: List[np.ndarray | None] = [None] * len(texts)
        texts_to_fetch = []
        indices_to_fetch = []

        for i, text in enumerate(texts):
            if not with_cache:
                texts_to_fetch.append(text)
                indices_to_fetch.append(i)
                continue

            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            file_path = (
                CACHE_DIR
                / self.embedding.model.replace(":", "").replace("/", "")
                / f"{text_hash}.npy"
            )

            if file_path.exists():
                try:
                    results[i] = np.load(file_path)
                    file_path.touch()
                except:  # noqa
                    texts_to_fetch.append(text)
                    indices_to_fetch.append(i)
            else:
                texts_to_fetch.append(text)
                indices_to_fetch.append(i)

        total = len(texts)
        print(f"CACHE HIT for {total-len(texts_to_fetch)}/{total} texts!")

        if texts_to_fetch:
            new_embeddings = await self.embedding.embed(
                input=texts_to_fetch,
            )
            for original_idx, embedding, text in zip(
                indices_to_fetch, new_embeddings, texts_to_fetch
            ):
                if with_cache:
                    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                    file_path = (
                        CACHE_DIR
                        / self.embedding.model.replace(":", "").replace("/", "")
                        / f"{text_hash}.npy"
                    )
                    np.save(file_path, embedding)
                results[original_idx] = embedding

        return results

    async def acompare(self, document_chunk: Chunk, summary_chunk: Chunk) -> float:
        doc_input = self.__prepare_document_chunk(document_chunk)
        doc_emb = (await self._get_embeddings([doc_input]))[0]

        if self.split_summary:
            sentences = self._split_into_sentences(summary_chunk)
            summ_inputs = [self.__prepare_summary_text(s) for s in await sentences]
        else:
            summ_inputs = [self.__prepare_summary_text(summary_chunk.text)]

        summ_embs = await self._get_embeddings(summ_inputs)

        scores = [self.similarity_fn(doc_emb, s_emb) for s_emb in summ_embs]

        return max(scores) if scores else 0.0

    def __prepare_document_chunk(self, chunk: Chunk) -> str:
        return self.__prepare_document_text(chunk.text)

    def __prepare_summary_text(self, text: str) -> str:
        return self.query_template.render(text=text)

    def __prepare_document_text(self, text: str) -> str:
        return self.passage_template.render(text=text)

    def prune_cache(self, max_age_days: int = 7):
        now = time.time()
        cutoff = now - (max_age_days * 86400)  # 86400 seconds in a day

        deleted_count = 0
        freed_space_mb = 0

        path = CACHE_DIR / self.embedding.model.replace(":", "").replace("/", "")
        os.makedirs(path, exist_ok=True)

        for file_path in path.glob("*.npy"):
            try:
                mtime = file_path.stat().st_mtime

                if mtime < cutoff:
                    size = file_path.stat().st_size
                    file_path.unlink()

                    deleted_count += 1
                    freed_space_mb += size / (1024 * 1024)
            except:  # noqa
                pass

        if deleted_count > 0:
            print(
                f"Cache Pruned: Deleted {deleted_count} files, freed {freed_space_mb:.2f} MB."
            )
