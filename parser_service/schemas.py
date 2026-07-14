from typing import Literal

from pydantic import BaseModel, Field


class CharBox(BaseModel):
    char: str
    x0: float
    top: float
    x1: float
    bottom: float
    page: int = Field(ge=1)
    is_boilerplate: bool = False
    # "char": a real per-glyph rectangle from Docling. "word": Docling does not
    # expose per-character geometry under this parse configuration, so this box
    # is the character's containing word's full bounding box, not an estimate
    # of the glyph's own extent — consumers must not treat "word"-precision
    # boxes as exact-span highlighting coordinates. No default: every CharBox
    # must state which one it actually is.
    precision: Literal["char", "word"]


class PageIndex(BaseModel):
    page: int = Field(ge=1)
    text: str
    char_map: list[CharBox]
