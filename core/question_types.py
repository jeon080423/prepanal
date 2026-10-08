"""문항 유형 레지스트리 (core/question_types.py)

새 문항 유형(순위형·매트릭스·상수합 등) 추가 방법:
    1. QuestionType을 상속한 클래스 1개 작성
    2. 클래스 위에 @register_question_type 데코레이터 1줄
끝. 생성 전략·UI·분포 학습이 자동으로 새 유형을 인식한다.

필수 인터페이스:
    validate(q)            -> list[str]  오류 메시지 목록 (빈 리스트 = 유효)
    prompt_fragment(q)     -> str        프롬프트에 들어가는 문항 한 줄
    parse_response(q, raw) -> str        원시 응답을 정식 답변 문자열로 정규화
    learn_distribution(q, series) -> dict | None  경험분포 추출 (None = 학습 제외)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import pandas as pd

from config import LIKERT_OPTIONS, MULTI_SEP


class UnknownQuestionTypeError(ValueError):
    """등록되지 않은 문항 유형을 요청했을 때 발생."""


class QuestionType(ABC):
    name: str = ""            # 레지스트리 키 (예: "single")
    label: str = ""           # 한국어 표시명
    needs_options: bool = False   # 보기 입력 UI 표시 여부
    is_choice: bool = False       # 선택형(확률분포 기반 생성 대상) 여부

    # -- 필수 인터페이스 -----------------------------------------------------
    @abstractmethod
    def validate(self, q: dict) -> list[str]:
        """문항 dict 검증. 오류 메시지 목록 반환 (유효하면 [])."""
        ...

    @abstractmethod
    def prompt_fragment(self, q: dict) -> str:
        """LLM 프롬프트용 문항 한 줄."""
        ...

    @abstractmethod
    def parse_response(self, q: dict, raw) -> str:
        """원시 응답 값을 정식 답변 문자열로 정규화."""
        ...

    @abstractmethod
    def learn_distribution(self, q: dict, series: pd.Series) -> dict | None:
        """실측 시리즈에서 경험분포 {보기: 확률} 추출. None이면 학습 제외."""
        ...

    # -- 선택 오버라이드 (기본 구현 제공) ------------------------------------
    def options_for(self, q: dict) -> list[str]:
        """유효 보기 목록 (리커트는 고정 보기)."""
        return [str(o) for o in (q.get("options") or [])]

    def dist_prompt_fragment(self, q: dict) -> str:
        """분포 추출 프롬프트용 한 줄 (기본 = prompt_fragment)."""
        return self.prompt_fragment(q)

    def probs_from_elicited(self, q: dict, raw: dict) -> dict[str, float]:
        """LLM이 준 보기별 확률을 {보기: 0~1} 로 정리."""
        opts = self.options_for(q)
        return {opt: min(1.0, max(0.0, float(raw.get(opt, 0) or 0))) for opt in opts}

    def sample_from_learned(self, q: dict, dist: dict[str, float],
                            rng: random.Random) -> str:
        """학습된 경험분포에서 1건 샘플링."""
        return _weighted_pick(dist, rng)

    def sample_from_elicited(self, q: dict, probs: dict[str, float],
                             rng: random.Random) -> str:
        """LLM 추출 분포에서 1건 샘플링 (기본 = 정규화 후 가중 샘플링)."""
        return _weighted_pick(_normalize(probs, self.options_for(q)), rng)


def _normalize(probs: dict[str, float], options: list[str]) -> dict[str, float]:
    """확률 정규화. 누락 보기는 0, 합이 0이면 균등분포."""
    norm = {opt: max(0.0, float(probs.get(opt, 0) or 0)) for opt in options}
    total = sum(norm.values())
    if total <= 0:
        return {opt: 1.0 / len(options) for opt in options}
    return {opt: v / total for opt, v in norm.items()}


def _weighted_pick(dist: dict[str, float], rng: random.Random) -> str:
    total = sum(dist.values())
    if total <= 0:
        return next(iter(dist))
    r = rng.uniform(0, total)
    upto = 0.0
    for k, v in dist.items():
        upto += v
        if r <= upto:
            return k
    return next(reversed(dist))


# ----------------------------------------------------------------------------
# 기본 4종
# ----------------------------------------------------------------------------
class SingleChoiceType(QuestionType):
    name = "single"
    label = "단일선택"
    needs_options = True
    is_choice = True

    def validate(self, q: dict) -> list[str]:
        errs = []
        if not str(q.get("text", "")).strip():
            errs.append("문항 내용을 입력하세요.")
        if len(self.options_for(q)) < 2:
            errs.append("보기를 2개 이상 입력하세요.")
        return errs

    def prompt_fragment(self, q: dict) -> str:
        return f"{q['id']}. {q['text']} [단일선택: {'/'.join(self.options_for(q))}]"

    def parse_response(self, q: dict, raw) -> str:
        return str(raw).strip()

    def learn_distribution(self, q: dict, series: pd.Series) -> dict | None:
        counts = series.astype(str).str.strip().value_counts(normalize=True)
        return {str(k): float(v) for k, v in counts.items()}


class MultiChoiceType(QuestionType):
    name = "multi"
    label = "다중선택"
    needs_options = True
    is_choice = True

    def validate(self, q: dict) -> list[str]:
        errs = []
        if not str(q.get("text", "")).strip():
            errs.append("문항 내용을 입력하세요.")
        if len(self.options_for(q)) < 2:
            errs.append("보기를 2개 이상 입력하세요.")
        return errs

    def prompt_fragment(self, q: dict) -> str:
        return (f"{q['id']}. {q['text']} "
                f"[다중선택: {'/'.join(self.options_for(q))}] (복수 선택 가능, '{MULTI_SEP}'로 구분)")

    def dist_prompt_fragment(self, q: dict) -> str:
        return (f"{q['id']}. {q['text']} [다중선택 | 보기: {'/'.join(self.options_for(q))}]"
                f" — 각 보기별 독립 선택확률")

    def parse_response(self, q: dict, raw) -> str:
        opts = self.options_for(q)
        parts = [p.strip() for p in str(raw).split(MULTI_SEP) if p.strip()]
        valid = [p for p in parts if p in opts]
        return MULTI_SEP.join(sorted(valid or parts[:1]))

    def learn_distribution(self, q: dict, series: pd.Series) -> dict | None:
        exploded: list[str] = []
        for val in series.astype(str):
            exploded.extend(p.strip() for p in val.split(MULTI_SEP) if p.strip())
        if not exploded:
            return None
        counts = pd.Series(exploded).value_counts(normalize=True)
        return {str(k): float(v) for k, v in counts.items()}

    def sample_from_learned(self, q: dict, dist: dict[str, float],
                            rng: random.Random) -> str:
        # 경험분포에서 1~3개 보기를 중복 없이 뽑는다 (기존 동작 유지).
        k = rng.randint(1, min(3, len(dist)))
        picks = {_weighted_pick(dist, rng) for _ in range(k)}
        return MULTI_SEP.join(sorted(picks))

    def sample_from_elicited(self, q: dict, probs: dict[str, float],
                             rng: random.Random) -> str:
        # 보기별 독립 확률로 Bernoulli 샘플링. 최소 1개 보장.
        if sum(probs.values()) <= 0:
            probs = {opt: 0.5 for opt in self.options_for(q)}
        picks = [opt for opt, p in probs.items() if rng.random() < min(1.0, max(0.0, p))]
        if not picks:
            picks = [max(probs, key=lambda o: probs[o])]
        return MULTI_SEP.join(sorted(picks))


class LikertType(QuestionType):
    name = "likert"
    label = "5점 리커트"
    needs_options = False
    is_choice = True

    def validate(self, q: dict) -> list[str]:
        if not str(q.get("text", "")).strip():
            return ["문항 내용을 입력하세요."]
        return []

    def options_for(self, q: dict) -> list[str]:
        return list(LIKERT_OPTIONS)

    def prompt_fragment(self, q: dict) -> str:
        return f"{q['id']}. {q['text']} [5점 리커트: {'/'.join(LIKERT_OPTIONS)}]"

    def parse_response(self, q: dict, raw) -> str:
        return str(raw).strip()

    def learn_distribution(self, q: dict, series: pd.Series) -> dict | None:
        counts = series.astype(str).str.strip().value_counts(normalize=True)
        return {str(k): float(v) for k, v in counts.items()}


class OpenType(QuestionType):
    name = "open"
    label = "주관식"
    needs_options = False
    is_choice = False

    def validate(self, q: dict) -> list[str]:
        if not str(q.get("text", "")).strip():
            return ["문항 내용을 입력하세요."]
        return []

    def prompt_fragment(self, q: dict) -> str:
        return f"{q['id']}. {q['text']} [주관식: 1-2문장으로 자연스럽게]"

    def parse_response(self, q: dict, raw) -> str:
        return str(raw).strip()

    def learn_distribution(self, q: dict, series: pd.Series) -> dict | None:
        return {}  # 주관식은 분포를 만들지 않음 (n만 기록)


# ----------------------------------------------------------------------------
# 레지스트리
# ----------------------------------------------------------------------------
QUESTION_TYPE_REGISTRY: dict[str, QuestionType] = {}


def register_question_type(cls: type[QuestionType]) -> type[QuestionType]:
    """새 문항 유형 등록. 사용법: 클래스 정의 위에 @register_question_type."""
    inst = cls()
    if not inst.name:
        raise ValueError(f"{cls.__name__}.name 이 비어 있습니다.")
    QUESTION_TYPE_REGISTRY[inst.name] = inst
    return cls


def get_question_type(name: str) -> QuestionType:
    """등록된 문항 유형 반환. 없으면 등록 목록을 안내하는 오류 발생."""
    try:
        return QUESTION_TYPE_REGISTRY[name]
    except KeyError:
        raise UnknownQuestionTypeError(
            f"알 수 없는 문항 유형: '{name}'. "
            f"등록된 유형: {sorted(QUESTION_TYPE_REGISTRY)}"
        ) from None


# 기본 4종 등록 (이 4줄이 전부 — 새 유형도 같은 방식으로 1줄 추가)
register_question_type(SingleChoiceType)
register_question_type(MultiChoiceType)
register_question_type(LikertType)
register_question_type(OpenType)
