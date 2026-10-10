from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Page:
    document: str
    page: int  # 1-based PDF page, not necessarily the printed page label
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float

    @property
    def citation(self) -> str:
        return f"{self.chunk.document}, PDF p. {self.chunk.page}"


@dataclass
class Answer:
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    mode: str = "extractive"
    warning: str = ""

