from pydantic import BaseModel, Field


class CharBox(BaseModel):
    char: str
    x0: float
    top: float
    x1: float
    bottom: float
    page: int = Field(ge=1)


class PageIndex(BaseModel):
    page: int = Field(ge=1)
    text: str
    char_map: list[CharBox]
