from typing import Optional

from pydantic import BaseModel


class Chunk(BaseModel):
    document_idx: Optional[int] = None

    text: str
    text_original: str

    idx_start: int
    idx_end: int
