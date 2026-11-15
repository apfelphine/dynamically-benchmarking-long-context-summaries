import os
from typing import Optional, List, Annotated

from pydantic import BaseModel, AfterValidator


class TextStats(BaseModel):
    chars: int
    tokens: int


class Document(BaseModel):
    id: int
    title: str
    url: str
    path: str

    summary: TextStats
    document: TextStats

    @property
    def name(self):
        return self.path.split("\\")[-1]


def fix_path(path: str) -> str:
    if os.name == "nt":
        return path
    return path.replace("\\", "/")


def convert_path_for_current_os(path: str | List[str]) -> str | List[str]:
    if isinstance(path, list):
        return [fix_path(p) for p in path]
    else:
        return fix_path(path)


class Bin(BaseModel):
    topic: str
    name: str
    doc_tokens: int
    sum_tokens: int
    document_paths: Annotated[List[str], AfterValidator(convert_path_for_current_os)]

    @property
    def document_names(self):
        return [p.split("\\")[-1] for p in self.document_paths]


class Summary(BaseModel):
    bin_idx: int

class GenericSummary(Summary):
    document_order: List[int]

class QueryBasedSummary(GenericSummary):
    chunk_order: Optional[List[int]] = None
    focus_doc_idx: int
