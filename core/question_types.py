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

변경 이력 (캘리브레이션 2차, 워커 A):
    - prompt_fragment: 보기를 "/" 한 줄로 붙이지 않고 줄바꿈·번호 목록으로
      제시. 보기 안에 "/"가 있어도 구분자가 아니며, 보기 문자열을 목록
      그대로 복사하라는 지시 추가. (갤럽 684호 재측정에서 복합 보기
      "어느 쪽도 아니다 / 잘 모르겠다(의견 유보)"가 "/" 기준으로 쪼개져
      반환되는 기계적 손실 방지)
    - probs_from_elicited: 반환을 dict에서 (probs, unmatched) 튜플로 변경.
      정확 일치에 실패해도 조용히 0이 되지 않고, 매칭 규칙(a)~(d)으로
      복원하거나 unmatched에 질량을 남긴다. 호출부는 strategies.py
      3곳(및 benchmarks 2곳)의 언패킹을 함께 바꿔야 한다.
"""

from __future__ import annotations

import random
import unicodedata
from abc import ABC, abstractmethod

import pandas as pd

from config import LIKERT_OPTIONS, MULTI_SEP


# 매칭되지 못한 raw 질량의 합이 이 값을 넘으면 분포 추출이 신뢰할 수
# 없을 수 있다. 재시도 여부는 strategies.py 담당이 이 값으로 판단한다.
UNMATCHED_MASS_THRESHOLD = 0.02


class UnknownQuestionTypeError(ValueError):
    """등록되지 않은 문항 유형을 요청했을 때 발생."""


def _norm_key(value) -> str:
    """매칭용 키 정규화: NFKC 정규화 + 앞뒤 공백 제거 + 내부 공백 축소."""
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def _options_block(options: list[str], kind: str, extra: str = "") -> str:
    """보기 목록 블록: 줄바꿈·번호 목록. '/'로 보기를 붙이지 않는다.

    보기 문자열에 '/'가 포함돼 있어도 그것은 보기 내부 문자일 뿐 구분자가
    아니므로, 모델이 보기를 쪼개거나 합치지 않고 목록 그대로 복사하게 한다.
    """
    lines = [
        f"[{kind}] 보기 목록 — 아래 보기 문자열을 목록 그대로 복사해서 답하세요. "
        "보기를 나누거나 합치지 마세요. 보기 안에 '/'가 있어도 구분자가 아닙니다.",
    ]
    lines += [f"  {i + 1}) {opt}" for i, opt in enumerate(options)]
    if extra:
        lines.append(f"  {extra}")
    return "\n".join(lines)


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

    def probs_from_elicited(self, q: dict, raw: dict) -> tuple[dict[str, float], dict[str, float]]:
        """LLM이 준 보기별 확률을 정리해 (probs, unmatched) 튜플로 반환.

        probs: 모든 보기를 키로 갖는 {보기: 0~1} (매칭되지 않은 보기는 0).
        unmatched: 어떤 보기에도 매칭되지 못한 raw 키와 그 질량
            {받은 그대로의 raw 키: 질량}. 조용히 버려지지 않는다.
            sum(unmatched.values()) > UNMATCHED_MASS_THRESHOLD 이면
            strategies.py 재시도 대상이다.

        매칭 규칙 (위에서부터 순서대로, 먼저 걸린 규칙이 이긴다):
          (a) 정규화(NFKC + 앞뒤 공백 제거 + 내부 공백 축소) 후 정확 일치.
          (b) 분할 키 병합: raw 키가 어떤 보기를 "/"로 쪼갠 조각 중
              하나와 정규화 후 일치하면 그 보기로 질량 합산.
          (c) 유일 부분일치: 정규화된 raw 키가 정확히 하나의 보기의
              부분 문자열이거나, 그 반대(보기가 raw 키의 부분 문자열)
              인 경우 그 보기로 합산. 2개 이상 보기에 걸리면 모호하므로
              unmatched로 남긴다.
          (d) 그 외는 unmatched.
        """
        opts = self.options_for(q)
        probs: dict[str, float] = {opt: 0.0 for opt in opts}
        unmatched: dict[str, float] = {}
        if not isinstance(raw, dict):
            return probs, unmatched

        norm_opts = {_norm_key(opt): opt for opt in opts}
        norm_opt_list = [(_norm_key(opt), opt) for opt in opts]
        # (b)용: 보기 -> "/" 조각(정규화) 목록
        fragments: dict[str, list[str]] = {
            opt: [_norm_key(f) for f in str(opt).split("/") if _norm_key(f)]
            for opt in opts
        }

        for raw_key, raw_val in raw.items():
            try:
                mass = float(raw_val or 0)
            except (TypeError, ValueError):
                # 숫자로 바꿀 수 없는 값은 질량을 셀 수 없으므로
                # 질량 0.0으로 unmatched에 남겨 흔적을 보존한다.
                unmatched[str(raw_key)] = 0.0
                continue
            if mass < 0:
                mass = 0.0
            key = _norm_key(raw_key)
            # (a) 정확 일치
            if key in norm_opts:
                probs[norm_opts[key]] = min(
                    1.0, probs[norm_opts[key]] + mass)
                continue
            # (b) 분할 키 병합 (조각이 2개 이상 보기에 겹치면 모호)
            frag_hits = [opt for opt, frags in fragments.items()
                         if key and key in frags]
            if len(frag_hits) == 1:
                probs[frag_hits[0]] = min(
                    1.0, probs[frag_hits[0]] + mass)
                continue
            if len(frag_hits) > 1:
                unmatched[str(raw_key)] = mass
                continue
            # (c) 유일 부분일치 (양방향 부분 문자열)
            sub_hits = [opt for norm_opt, opt in norm_opt_list
                        if key and (key in norm_opt or norm_opt in key)]
            if len(sub_hits) == 1:
                probs[sub_hits[0]] = min(
                    1.0, probs[sub_hits[0]] + mass)
                continue
            # (d) 그 외 (모호한 부분일치 포함)는 unmatched
            unmatched[str(raw_key)] = mass
        return probs, unmatched

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
        return (f"{q['id']}. {q['text']}\n"
                f"{_options_block(self.options_for(q), '단일선택')}")

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
        return (f"{q['id']}. {q['text']}\n"
                f"{_options_block(self.options_for(q), '다중선택', f'복수 선택 가능 — 선택한 보기는 {MULTI_SEP!r}로 구분하세요.')}")

    def dist_prompt_fragment(self, q: dict) -> str:
        return (f"{q['id']}. {q['text']}\n"
                f"{_options_block(self.options_for(q), '다중선택', '각 보기마다 독립적인 선택 확률(0~1)을 주세요. 합이 1이 아니어도 됩니다.')}")

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
        return (f"{q['id']}. {q['text']}\n"
                f"{_options_block(list(LIKERT_OPTIONS), '5점 리커트')}")

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
