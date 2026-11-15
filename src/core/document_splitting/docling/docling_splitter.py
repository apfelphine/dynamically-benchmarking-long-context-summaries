import math
import re
import uuid
from typing import List, Optional, Union

import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, MarkdownFormatOption
from docling_core.transforms.chunker import (
    HierarchicalChunker,
    DocChunk,
    DocMeta,
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc import (
    DocItemLabel,
    TextItem,
)

from core.document_splitting.base_document_splitter import BaseDocumentSplitter
from core.document_splitting.docling.custom_md_backend import MarkdownDocumentBackend
from core.document_splitting.spacy_sentence_splitter import SpacySentenceSplitter
from models.document_splitting import Chunk


class DoclingParagraphChunker(BaseDocumentSplitter):
    def __init__(
        self,
        contextualize: bool = False,
        merge_peers: bool = False,
        max_sentences: Optional[int] = None,
        tokenizer=None,
        min_tokens: Optional[int] = None,
    ):
        self.converter = DocumentConverter(
            format_options={
                InputFormat.MD: MarkdownFormatOption(backend=MarkdownDocumentBackend),
            }
        )
        self.contextualize = contextualize
        self.merge_peers = merge_peers
        self.min_tokens = min_tokens

        if not tokenizer:
            # Docling Default
            self.tokenizer = self.get_hugging_face_tokenizer(
                "sentence-transformers/all-MiniLM-L6-v2", 512
            )
        else:
            self.tokenizer = tokenizer

        if max_sentences is not None:
            if max_sentences <= 0:
                raise ValueError("max_sentences must be positive.")
            if merge_peers:
                raise ValueError("Max sentences is not supported with merge_peers")
        self.max_sentences = max_sentences

        self.chunker = HierarchicalChunker()

    async def _split(self, text) -> List[Chunk]:
        raise Exception("Not implemented")

    async def split(self, text: str) -> List[Chunk]:
        text = text.replace("<br>", "    ")
        doc = self.converter.convert_string(
            text, format=InputFormat.MD, name=f"temp_docling_{uuid.uuid4().hex}.md"
        ).document

        chunks: List[DocChunk] = list(self.chunker.chunk(doc))  # noqa
        if self.merge_peers:
            chunks = self.__custom_merge_peers(chunks)

        if self.max_sentences is not None:
            chunks = await self.__split_max_sentences(chunks)

        ct = 0
        result = []
        for c in chunks:
            if not c.text:
                continue

            try:
                start_idx, end_idx = self.get_index(text, c.text)
            except ValueError:
                start_idx, end_idx = 0, 0
                ct += 1
                # try:
                #     texts = self.__get_text_from_docitems(c.meta.doc_items)
                #     start_idx, end_idx = self.get_index(text, " ".join(texts))
                # except ValueError:
                #     start_idx, end_idx = 0, 0

            original_text = text[start_idx:end_idx]
            chunk_text = self.__contextualize(c)
            result.append(
                Chunk(
                    text=chunk_text,
                    text_original=original_text,
                    idx_start=start_idx,
                    idx_end=end_idx,
                )
            )

        # Get Original Text for remaining
        last_end_idx = 0
        last_possible_idx = len(text) - 1
        for idx, c in enumerate(result):

            if c.idx_start == 0 and c.idx_end == 0:
                new_end_idx = (
                    last_possible_idx
                    if idx == len(result) - 1
                    else result[idx + 1].idx_start
                )

                if not new_end_idx == 0 and last_end_idx is not None:
                    c.idx_start = last_end_idx
                    c.idx_end = new_end_idx
                    c.text_original = text[c.idx_start : c.idx_end]
                    last_end_idx = c.idx_end
                else:
                    last_end_idx = None
            else:
                last_end_idx = c.idx_end

        if ct:
            print(f"WARN: count with invalid={ct}")
        if not result:
            result = [
                Chunk(
                    text=text,
                    text_original=text,
                    idx_start=0,
                    idx_end=len(text),
                )
            ]
        return result

    def get_index(self, text: str, chunk: str) -> tuple[int, int]:
        words = re.findall(r"[a-zA-Z]+", chunk)
        if not words:
            # Fallback
            super().get_index(text, chunk)

        escaped_words = [re.escape(word) for word in words]
        pattern = r"[^a-zA-Z]*".join(escaped_words)
        match = re.search(pattern, text)

        if not match:
            raise ValueError("No match found.")

        return match.start(), match.end()

    @staticmethod
    def __get_text_from_docitems(docitems):
        texts = []
        for item in docitems:
            children = []
            if hasattr(item, "text"):
                texts.append(item.text)
            if hasattr(item, "data"):
                children.append(item.data)
            if hasattr(item, "children"):
                children.extend(item.children)
            if hasattr(item, "grid"):
                for row in item.grid:
                    children.extend(row)
            if children:
                texts.extend(DoclingParagraphChunker.__get_text_from_docitems(children))
        return texts

    def __contextualize(self, c):
        if self.contextualize:
            if c.meta.headings:
                header_text = ""
                for idx, header in enumerate(c.meta.headings):
                    header_text += "#" * (idx + 1) + " " + header + "\n"

                return header_text + c.text
            else:
                return self.chunker.contextualize(c)
        return c.text

    async def __split_max_sentences(self, chunks):
        if self.max_sentences is None:
            return chunks

        sentence_splitter = SpacySentenceSplitter()

        res = []
        for c in chunks:
            sentences = await sentence_splitter.split(c.text)
            if len(sentences) > self.max_sentences:
                n_splits = math.ceil(len(sentences) / self.max_sentences)
                sent_per_split = math.floor(len(sentences) / n_splits)
                remaining_sentencnes = int(len(sentences) - (n_splits * sent_per_split))

                for i in range(n_splits):
                    start = i * sent_per_split + min(i, remaining_sentencnes)
                    end = start + sent_per_split + min(i + 1, remaining_sentencnes)
                    selected_sentences = sentences[start:end]

                    res.append(
                        DocChunk(
                            text="\n".join(
                                [sent.text_original for sent in selected_sentences]
                            ),
                            meta=c.meta,
                        )
                    )
            else:
                res.append(c)
        return res

    def __custom_merge_peers(self, chunks):
        custom_ref_count = 0

        # Loosely adapted from _merge_chunks_with_matching_metadata in the hybrid chunker
        output_chunks = []
        window_start = 0
        window_end = 0
        num_chunks = len(chunks)
        current_headings = None
        first_chunk_of_window = None
        new_chunk = None
        current_new_headings = []

        current_block_tokens = 0

        while window_end < num_chunks:
            chunk = chunks[window_end]
            headings = chunk.meta.headings
            ready_to_append = False

            if window_start == window_end:
                current_headings = headings
                current_new_headings = []
                window_end += 1
                first_chunk_of_window = chunk

                current_block_tokens = self.__count_text_tokens(
                    self.__contextualize(chunk)
                )
            else:
                chunks = chunks[window_start : window_end + 1]
                doc_items = [it for chk in chunks for it in chk.meta.doc_items]
                candidate = DocChunk(
                    text="\n".join([chk.text for chk in chunks]),
                    meta=DocMeta(
                        doc_items=doc_items,
                        headings=current_headings,
                        origin=chunk.meta.origin,
                    ),
                )

                # Calculate candidate size once
                candidate_tokens = self.__count_text_tokens(
                    self.__contextualize(candidate)
                )

                if candidate_tokens <= self.__max_tokens:
                    shared_headings = []
                    extra_headings_new = []
                    extra_headings_old = []

                    idx = 0

                    if current_headings:
                        for h, c_h in zip(headings, current_headings):
                            if h == c_h:
                                shared_headings.append(h)
                            else:
                                if not h in current_new_headings:
                                    extra_headings_new.append("#" * (idx + 1) + f" {h}")
                                    current_new_headings.append(h)
                                extra_headings_old.append(c_h)
                            idx += 1

                        if len(headings) < len(current_headings):
                            for c_h_idx in range(len(headings), len(current_headings)):
                                h = current_headings[c_h_idx]
                                extra_headings_old.append(h)

                        elif len(current_headings) < len(headings):
                            for h_idx in range(len(current_headings), len(headings)):
                                h = headings[h_idx]
                                if not h in current_new_headings:
                                    extra_headings_new.append("#" * (idx + 1) + f" {h}")
                                    current_new_headings.append(h)

                    if not extra_headings_new and not extra_headings_old:
                        # there is room to include the new chunk so add it to the window and continue
                        window_end += 1
                        new_chunk = candidate
                        current_block_tokens = candidate_tokens
                    elif self.min_tokens and current_block_tokens < self.min_tokens:
                        current_headings = shared_headings + extra_headings_old
                        if self.contextualize:
                            if extra_headings_new:
                                if hasattr(
                                    chunks[window_end].meta.doc_items[0], "text"
                                ):
                                    chunks[window_end].meta.doc_items[0].text = (
                                        "\n".join(extra_headings_new)
                                        + "\n"
                                        + chunks[window_end].meta.doc_items[0].text
                                    )
                                else:
                                    chunks[window_end].meta.doc_items.insert(
                                        0,
                                        TextItem(
                                            parent=chunks[window_end]
                                            .meta.doc_items[0]
                                            .parent,
                                            text="\n".join(extra_headings_new),
                                            label=DocItemLabel.TEXT,
                                            self_ref=f"#/texts/{1000000+custom_ref_count}",
                                            orig="\n".join(extra_headings_new),
                                        ),
                                    )
                                    custom_ref_count += 1
                                chunks[window_end].text = (
                                    "\n".join(extra_headings_new)
                                    + "\n"
                                    + chunks[window_end].text
                                )

                        # Refresh the candidate
                        chunks = chunks[window_start : window_end + 1]
                        doc_items = [it for chk in chunks for it in chk.meta.doc_items]
                        candidate = DocChunk(
                            text="\n".join([chk.text for chk in chunks]),
                            meta=DocMeta(
                                doc_items=doc_items,
                                headings=current_headings,
                                origin=chunk.meta.origin,
                            ),
                        )
                        window_end += 1

                        new_chunk = candidate
                        current_block_tokens = self.__count_text_tokens(
                            self.__contextualize(candidate)
                        )
                    else:
                        ready_to_append = True
                else:
                    ready_to_append = True

            if ready_to_append or window_end == num_chunks:
                # no more room OR the start of new metadata.  Either way, end the block
                # and use the current window_end as the start of a new block
                if window_start + 1 == window_end:
                    # just one chunk so use it as is
                    output_chunks.append(first_chunk_of_window)
                else:
                    output_chunks.append(new_chunk)
                # no need to reset window_text, etc. because that will be reset in the
                # next iteration if window_start == window_end block
                window_start = window_end

        return output_chunks

    def __count_text_tokens(self, text: Optional[Union[str, list[str]]]):
        if text is None:
            return 0
        elif isinstance(text, list):
            total = 0
            for t in text:
                total += self.__count_text_tokens(t)
            return total
        return self.tokenizer.count_tokens(text=text)

    @property
    def __max_tokens(self) -> int:
        return self.tokenizer.get_max_tokens()

    @staticmethod
    def get_hugging_face_tokenizer(model_name: str, max_tokens: int):
        return HuggingFaceTokenizer.from_pretrained(
            model_name=model_name,
            max_tokens=max_tokens,
        )

    @staticmethod
    def get_openai_tokenizer(encoding: str, max_tokens: int):
        return OpenAITokenizer(
            tokenizer=tiktoken.get_encoding(encoding), max_tokens=max_tokens
        )
