"""데이터 모델 (core/models.py)

문항은 JSON 직렬화 가능한 dict를 기본 교환 형식으로 쓴다
(세션 저장·샘플 JSON·엑셀과 그대로 호환되기 때문).
스키마:
    {"id": "Q1", "text": "...", "type": "single", "options": [...]}

Question dataclass는 타입 안전성이 필요할 때 쓰는 선택적 래퍼다.
"""

from dataclasses import dataclass, field
from typing import TypedDict


@dataclass
class Question:
    """문항 타입 래퍼. dict와 상호 변환 가능."""

    id: str
    text: str
    type: str
    options: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(
            id=str(d.get("id", "")),
            text=str(d.get("text", "")),
            type=str(d.get("type", "")),
            options=[str(o) for o in (d.get("options") or [])],
        )

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "type": self.type,
                "options": list(self.options)}


class LearnedDist(TypedDict, total=False):
    """learn_distributions() 반환값 1건 분량."""
    matched_col: str
    type: str
    dist: dict[str, float]
    n: int


class Persona(TypedDict, total=False):
    """{차원: 구분} 예: {"성별": "여성", "연령대": "30대"}"""


class Transcript(TypedDict, total=False):
    name: str
    text: str
