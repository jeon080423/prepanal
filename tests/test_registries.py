"""레지스트리 플러그인 포인트 테스트.

핵심 명제: 새 provider/문항유형/전략 추가 = 클래스 1개 + 등록 1줄.
"""
import random

import pandas as pd
import pytest

from core import (
    AIProvider,
    GenerationStrategy,
    PROVIDER_REGISTRY,
    QUESTION_TYPE_REGISTRY,
    STRATEGY_REGISTRY,
    GenerationContext,
    QuestionType,
    UnknownProviderError,
    UnknownQuestionTypeError,
    UnknownStrategyError,
    generate,
    get_provider,
    get_question_type,
    get_strategy,
    register_provider,
    register_question_type,
    register_strategy,
    select_strategy,
)


# ----------------------------------------------------------------------------
# Provider 레지스트리
# ----------------------------------------------------------------------------
class DummyProvider(AIProvider):
    def _call_cli(self, prompt, model, timeout=180):
        # 가짜 CLI 경로가 존재한다고 가정하고 디스패치 경로를 테스트
        return f"DUMMY:{model}:{prompt[:10]}"

    def _call_rest(self, api_key, prompt, model, timeout=180):
        raise AssertionError("REST 경로로 오면 안 됨")


def test_default_providers_registered():
    assert "gemini" in PROVIDER_REGISTRY
    assert "openai" in PROVIDER_REGISTRY
    assert get_provider("gemini").label == "Gemini"


def test_register_new_provider_one_line(monkeypatch):
    # cli_path가 존재하는 것처럼 속여 CLI 디스패치 경로를 테스트
    monkeypatch.setattr("os.path.exists", lambda p: True)
    register_provider("dummy", DummyProvider(
        provider_id="dummy", label="Dummy", default_model="dummy-1",
        cli="/fake/dummy-cli"))
    try:
        assert get_provider("dummy").label == "Dummy"
        # 이름 분기 없이 레지스트리 디스패치로 호출됨
        out = generate("dummy", "hello", "dummy-1")
        assert out.startswith("DUMMY:dummy-1:")
    finally:
        del PROVIDER_REGISTRY["dummy"]


def test_unknown_provider_clear_error():
    with pytest.raises(UnknownProviderError) as e:
        get_provider("nope-not-real")
    assert "nope-not-real" in str(e.value)
    assert "gemini" in str(e.value)  # 등록 목록 안내


# ----------------------------------------------------------------------------
# 문항 유형 레지스트리
# ----------------------------------------------------------------------------
def test_default_question_types():
    assert set(QUESTION_TYPE_REGISTRY) == {"single", "multi", "likert", "open"}
    assert get_question_type("single").label == "단일선택"
    assert get_question_type("likert").is_choice
    assert not get_question_type("open").is_choice


def test_register_new_question_type_one_line():
    # 새 유형 추가 = 클래스 1개 + 등록 데코레이터 1줄 (테스트 안에서 등록·정리)
    @register_question_type
    class RankingType(QuestionType):
        """테스트용 새 유형: 순위형 (향후 확장 예시)."""
        name = "ranking"
        label = "순위형"
        needs_options = True
        is_choice = True

        def validate(self, q):
            errs = []
            if not str(q.get("text", "")).strip():
                errs.append("문항 내용을 입력하세요.")
            if len(self.options_for(q)) < 2:
                errs.append("보기를 2개 이상 입력하세요.")
            return errs

        def prompt_fragment(self, q):
            return f"{q['id']}. {q['text']} [순위형: {'/'.join(self.options_for(q))}]"

        def parse_response(self, q, raw):
            return str(raw).strip()

        def learn_distribution(self, q, series):
            return None  # v1에서는 학습 제외

    try:
        qt = get_question_type("ranking")
        assert qt.label == "순위형"
        q = {"id": "Q9", "text": "선호도 순위를 매겨주세요.",
             "type": "ranking", "options": ["A", "B", "C"]}
        assert qt.validate(q) == []
        assert "순위형" in qt.prompt_fragment(q)
        assert qt.parse_response(q, " A ") == "A"
        # 전략 매칭도 자동으로 됨 (is_choice=True → 분포추출 전략이 받음)
        ctx = GenerationContext(provider_id="dummy", model="dummy-1")
        assert select_strategy(q, ctx).name == "elicitation"
    finally:
        del QUESTION_TYPE_REGISTRY["ranking"]


def test_unknown_question_type_clear_error():
    with pytest.raises(UnknownQuestionTypeError) as e:
        get_question_type("matrix")
    assert "matrix" in str(e.value)
    assert "single" in str(e.value)


def test_question_type_validation():
    bad = {"id": "Q1", "text": "  ", "type": "single", "options": ["하나"]}
    errs = get_question_type("single").validate(bad)
    assert len(errs) == 2  # 내용 없음 + 보기 부족


# ----------------------------------------------------------------------------
# 전략 레지스트리
# ----------------------------------------------------------------------------
def test_default_strategies_registered():
    assert set(STRATEGY_REGISTRY) == {"learned", "precision",
                                      "elicitation", "open_ended"}


class DummyStrategy(GenerationStrategy):
    name = "dummy_strat"
    label = "더미"

    def supports(self, q, ctx):
        return False

    def generate(self, ctx, questions, personas, rng):
        return [{} for _ in personas]


def test_register_new_strategy_one_line():
    register_strategy(DummyStrategy())
    try:
        assert get_strategy("dummy_strat").label == "더미"
    finally:
        del STRATEGY_REGISTRY["dummy_strat"]


def test_unknown_strategy_clear_error():
    with pytest.raises(UnknownStrategyError) as e:
        get_strategy("nope")
    assert "nope" in str(e.value)


def test_strategy_selection_priority():
    rng = random.Random(0)
    q_single = {"id": "Q1", "text": "t", "type": "single",
                "options": ["A", "B"]}
    q_open = {"id": "Q5", "text": "t", "type": "open", "options": []}

    # 학습 분포 있음 → learned
    ctx = GenerationContext(provider_id="d", model="m",
                            learned={"Q1": {"dist": {"A": 0.5, "B": 0.5},
                                            "n": 10, "type": "single",
                                            "matched_col": "c"}})
    assert select_strategy(q_single, ctx).name == "learned"

    # 학습 분포 없음 → elicitation
    ctx2 = GenerationContext(provider_id="d", model="m")
    assert select_strategy(q_single, ctx2).name == "elicitation"

    # 고정밀 컨텍스트 → precision
    ctx3 = GenerationContext(provider_id="d", model="m", use_precision=True,
                             transcripts=[{"name": "t1", "text": "인터뷰"}])
    assert select_strategy(q_single, ctx3).name == "precision"

    # 주관식 → open_ended (고정밀에서도 마찬가지)
    assert select_strategy(q_open, ctx2).name == "open_ended"
    assert select_strategy(q_open, ctx3).name == "open_ended"
