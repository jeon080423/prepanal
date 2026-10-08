"""생성 전략 (core/strategies.py) — Strategy 패턴

3개 모드(분포학습/분포추출/고정밀) + 주관식 생성이 Strategy 클래스다.
새 전략 추가 방법:
    1. GenerationStrategy를 상속한 클래스 1개 작성
       (supports / generate 구현, 필요하면 persona_texts 오버라이드)
    2. 하단 register_strategy(...) 한 줄 추가
    3. (선택) config.STRATEGY_ORDER에 우선순위 반영
끝. GenerationService가 자동으로 새 전략을 문항에 매칭한다.

공통 인터페이스:
    supports(q, ctx) -> bool
        이 전략이 문항 q + 컨텍스트에 적용되는지
    generate(ctx, questions, personas, rng) -> list[dict]
        응답자별 [{qid: 답변}, ...] 반환 (len == len(personas))
    persona_texts(ctx, personas, rng) -> list[str]  (오버라이드 가능)
"""

from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from config import STRATEGY_ORDER
from . import providers as providers_mod
from .persona import build_persona_texts, persona_to_text
from .question_types import get_question_type


class UnknownStrategyError(ValueError):
    """등록되지 않은 전략을 요청했을 때 발생."""


@dataclass
class GenerationContext:
    """전략들이 공유하는 생성 컨텍스트."""

    provider_id: str
    model: str
    learned: dict = field(default_factory=dict)        # {qid: LearnedDist dict}
    transcripts: list[dict] = field(default_factory=list)
    use_precision: bool = False
    batch_size: int = 5
    cache: dict = field(default_factory=dict)
    generate_fn: Callable[[str, str, str], object] | None = None  # DI (테스트용)

    def text_generate(self, prompt: str) -> str:
        fn = self.generate_fn or providers_mod.generate
        return str(fn(self.provider_id, prompt, self.model))


# ----------------------------------------------------------------------------
# LLM I/O 헬퍼
# ----------------------------------------------------------------------------
def _cache_key(provider_id: str, model: str, prompt: str) -> str:
    return hashlib.md5(f"{provider_id}|{model}|{prompt}".encode("utf-8")).hexdigest()


def cached_generate(ctx: GenerationContext, prompt: str) -> str:
    key = _cache_key(ctx.provider_id, ctx.model, prompt)
    if key not in ctx.cache:
        ctx.cache[key] = ctx.text_generate(prompt)
    return ctx.cache[key]


def parse_json_array(text: str) -> list:
    """LLM 출력에서 JSON 배열을 추출한다."""
    t = text.strip()
    for fence in ("```json", "```"):
        if t.startswith(fence):
            t = t[len(fence):].strip()
            if t.endswith("```"):
                t = t[: -3].strip()
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON 배열을 찾을 수 없습니다.")
    return json.loads(t[start: end + 1])


def _check_batch(parsed: list, expected: int) -> None:
    if len(parsed) != expected:
        raise ValueError(
            f"LLM이 {expected}건 요청에 {len(parsed)}건을 반환했습니다. 다시 시도해 주세요."
        )


def _build_dist_prompt(choice_questions: list[dict],
                       persona_texts: list[str]) -> str:
    """Prolific 2026 방식: 강제 단일 응답이 아닌 보기별 선택 확률분포를 추출."""
    profiles = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(persona_texts))
    lines = [get_question_type(q["type"]).dist_prompt_fragment(q)
             for q in choice_questions]
    qblock = "\n".join(lines)
    return f"""당신은 설문조사 응답 분포 추정 전문가입니다.
각 응답자 프로필을 보고, 각 문항의 보기별 선택 확률(0~1 사이 실수)을 추정하세요.
특정 보기를 하나로 찍지 말고 반드시 확률분포 형태로 답하세요.

[응답자 프로필]
{profiles}

[문항]
{qblock}

[출력 형식]
JSON 배열로만 출력하세요. 설명이나 코드펜스 없이 아래 형식 그대로:
[{{"Q1": {{"보기1": 0.5, "보기2": 0.3}}, "Q3": {{"전혀 그렇지 않다": 0.1, ...}}}}, ...]
- 키는 문항 번호를 그대로 사용하세요.
- 단일선택/리커트는 한 문항의 확률 합이 1이 되게 하세요.
- 다중선택은 보기마다 독립적인 선택 확률(합이 1이 아니어도 됨)을 주세요.
- 프로필(연령·직업·지역 등)에 따라 현실적인 차이를 반영하세요."""


def _build_open_prompt(open_questions: list[dict],
                       persona_texts: list[str]) -> str:
    profiles = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(persona_texts))
    lines = [get_question_type(q["type"]).prompt_fragment(q) for q in open_questions]
    qblock = "\n".join(lines)
    return f"""당신은 설문조사 가상 응답자 시뮬레이터입니다.
아래 {len(persona_texts)}명의 응답자가 주관식 문항에 답한 것처럼 응답을 생성하세요.

[응답자 프로필]
{profiles}

[주관식 문항]
{qblock}

[출력 형식]
JSON 배열로만 출력하세요. 설명이나 코드펜스 없이 아래 형식 그대로:
[{{"Q5": "..."}}, ...]
- 키는 문항 번호를 그대로 사용하세요.
- 구어체 1-2문장, 각 응답자의 프로필에 맞는 자연스러운 말투로 작성하세요.
- 모든 응답자가 비슷하게 답하지 마세요."""


# ----------------------------------------------------------------------------
# Strategy 베이스
# ----------------------------------------------------------------------------
class GenerationStrategy(ABC):
    name: str = ""
    label: str = ""
    description: str = ""

    @abstractmethod
    def supports(self, q: dict, ctx: GenerationContext) -> bool:
        ...

    @abstractmethod
    def generate(self, ctx: GenerationContext, questions: list[dict],
                 personas: list[dict], rng: random.Random) -> list[dict]:
        ...

    def persona_texts(self, ctx: GenerationContext, personas: list[dict],
                      rng: random.Random) -> list[str]:
        """기본: 인구통계 페르소나. 고정밀 컨텍스트면 인터뷰 기반."""
        use_t = ctx.use_precision and bool(ctx.transcripts)
        return build_persona_texts(
            personas, rng, transcripts=ctx.transcripts if use_t else None)


def _has_learned_dist(q: dict, ctx: GenerationContext) -> bool:
    info = ctx.learned.get(q["id"])
    return bool(info and info.get("dist"))


# ----------------------------------------------------------------------------
# 1) 분포학습 — 학습된 경험분포에서 직접 샘플링 (API 호출 없음)
# ----------------------------------------------------------------------------
class LearnedDistributionStrategy(GenerationStrategy):
    name = "learned"
    label = "분포학습"
    description = "업로드된 실측 데이터의 경험분포에서 직접 샘플링 (API 호출 없음)"

    def supports(self, q: dict, ctx: GenerationContext) -> bool:
        return get_question_type(q["type"]).is_choice and _has_learned_dist(q, ctx)

    def generate(self, ctx, questions, personas, rng) -> list[dict]:
        out = []
        for _ in personas:
            ans = {}
            for q in questions:
                qtype = get_question_type(q["type"])
                ans[q["id"]] = qtype.sample_from_learned(
                    q, ctx.learned[q["id"]]["dist"], rng)
            out.append(ans)
        return out


# ----------------------------------------------------------------------------
# 2) 분포추출 — Prolific 2026 방식 (LLM 확률분포 추출 후 샘플링)
# ----------------------------------------------------------------------------
class DistributionElicitationStrategy(GenerationStrategy):
    name = "elicitation"
    label = "분포추출"
    description = ("Prolific 2026 방식: LLM이 페르소나별 보기별 선택 확률분포를 "
                   "추정하면 샘플링 (분포 오차 35~48% 감소 보고)")

    def supports(self, q: dict, ctx: GenerationContext) -> bool:
        return get_question_type(q["type"]).is_choice and not _has_learned_dist(q, ctx)

    def generate(self, ctx, questions, personas, rng) -> list[dict]:
        ptexts = self.persona_texts(ctx, personas, rng)
        raw = cached_generate(ctx, _build_dist_prompt(questions, ptexts))
        parsed = parse_json_array(raw)
        _check_batch(parsed, len(personas))
        out = []
        for i in range(len(personas)):
            ans = {}
            prow = parsed[i] if isinstance(parsed[i], dict) else {}
            for q in questions:
                qtype = get_question_type(q["type"])
                praw = prow.get(q["id"], {})
                probs = qtype.probs_from_elicited(
                    q, praw if isinstance(praw, dict) else {})
                ans[q["id"]] = qtype.sample_from_elicited(q, probs, rng)
            out.append(ans)
        return out


# ----------------------------------------------------------------------------
# 3) 고정밀 — 인터뷰 트랜스크립트 기반 (스탠포드 Generative Agents 방식)
# ----------------------------------------------------------------------------
class PrecisionGenerationStrategy(DistributionElicitationStrategy):
    name = "precision"
    label = "고정밀 (인터뷰 기반)"
    description = ("스탠포드 Generative Agents 연구: 정규화 정확도 83%. "
                   "인터뷰 트랜스크립트를 페르소나 컨텍스트로 사용")

    def supports(self, q: dict, ctx: GenerationContext) -> bool:
        return (super().supports(q, ctx)
                and ctx.use_precision and bool(ctx.transcripts))

    def persona_texts(self, ctx, personas, rng) -> list[str]:
        # 고정밀 모드에서는 항상 인터뷰 발췌문 기반 (명시적 보장)
        return build_persona_texts(personas, rng, transcripts=ctx.transcripts)


# ----------------------------------------------------------------------------
# 4) 주관식 — LLM 배치 생성
# ----------------------------------------------------------------------------
class OpenEndedGenerationStrategy(GenerationStrategy):
    name = "open_ended"
    label = "주관식 생성"
    description = "LLM 배치 호출로 주관식 답변 생성"

    def supports(self, q: dict, ctx: GenerationContext) -> bool:
        return not get_question_type(q["type"]).is_choice

    def generate(self, ctx, questions, personas, rng) -> list[dict]:
        ptexts = self.persona_texts(ctx, personas, rng)
        raw = cached_generate(ctx, _build_open_prompt(questions, ptexts))
        parsed = parse_json_array(raw)
        _check_batch(parsed, len(personas))
        out = []
        for i in range(len(personas)):
            ans = {}
            prow = parsed[i] if isinstance(parsed[i], dict) else {}
            for q in questions:
                qtype = get_question_type(q["type"])
                if q["id"] in prow:
                    ans[q["id"]] = qtype.parse_response(q, prow[q["id"]])
            out.append(ans)
        return out


# ----------------------------------------------------------------------------
# 레지스트리
# ----------------------------------------------------------------------------
STRATEGY_REGISTRY: dict[str, GenerationStrategy] = {}


def register_strategy(strategy: GenerationStrategy) -> GenerationStrategy:
    """새 생성 전략 등록. 사용법: register_strategy(MyStrategy())."""
    if not strategy.name:
        raise ValueError(f"{type(strategy).__name__}.name 이 비어 있습니다.")
    STRATEGY_REGISTRY[strategy.name] = strategy
    return strategy


def get_strategy(name: str) -> GenerationStrategy:
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError:
        raise UnknownStrategyError(
            f"알 수 없는 생성 전략: '{name}'. "
            f"등록된 전략: {sorted(STRATEGY_REGISTRY)}"
        ) from None


def select_strategy(q: dict, ctx: GenerationContext,
                    order: list[str] | None = None) -> GenerationStrategy:
    """문항에 적용할 전략을 우선순위대로 선택."""
    for name in (order or STRATEGY_ORDER):
        strategy = STRATEGY_REGISTRY.get(name)
        if strategy is not None and strategy.supports(q, ctx):
            return strategy
    raise UnknownStrategyError(
        f"문항 {q.get('id')}에 적용 가능한 전략이 없습니다. "
        f"(유형: {q.get('type')})"
    )


register_strategy(LearnedDistributionStrategy())
register_strategy(PrecisionGenerationStrategy())
register_strategy(DistributionElicitationStrategy())
register_strategy(OpenEndedGenerationStrategy())
