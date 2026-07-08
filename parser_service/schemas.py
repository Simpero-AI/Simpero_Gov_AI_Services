from pydantic import BaseModel, Field


class ParsedParagraph(BaseModel):
    page: int = Field(ge=1)
    paragraph: int = Field(ge=1)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
