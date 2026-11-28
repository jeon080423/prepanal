"""2차 캘리브레이션 C0+C1 신규 테스트 (워커 E).

검증 대상 — 갤럽 재측정에서 유보 보기가 0이 된 기계적 손실의 수정분:
  (A) question_types.py — prompt_fragment 줄바꿈·번호 목록 + "보기 문자열을
      목록 그대로 복사" 지시, probs_from_elicited (probs, unmatched) 튜플
      계약 + NFKC·공백 정규화 + 분할 키 병합, UNMATCHED_MASS_THRESHOLD=0.02
  (B) strategies.py — _call_with_retry check 파라미터, 미매칭 질량 합이
      0.02 초과면 배치 재시도(최대 3회), GenerationContext에
      batch_estimates·batch_unmatched·last_call_attempts
  (C) diagnostics.py — diagnose(..., estimated=...)에 추정확률 0.02 이상인데
      출현 0인 보기 '보기 미출현(구조적 0)' + '매칭 손실 의심' 플래그.
      services.py generate_responses 결과에 "estimation"(mean_estimated_probs·
      unmatched{keys, total_mass}·retry_counts)과 "execution"(fallback_used·
      key_start_index)
  (D) providers.py — get_last_call_info() {"start_key_index", "fallback_used"}

실행 (base 트리 기준, 기존 테스트와 같은 관례):
    cd ~/workspace/prepanel-fixes/calib-2/base
    ~/workspace/.venv-synth/bin/python -m pytest tests/test_calib2.py -q
이 파일은 base/tests/가 아니라 work/E/tests/에 두므로, 베이스라인 확인 시
위와 같이 base를 cwd로 두고 파일 경로를 지정해 실행한다.

실제 LLM 호출 없음 — generate_fn 스텁 주입만 쓴다. 비밀값 없음.
보기 라벨 문자열만 픽스처로 쓰고, 갤럽 수치는 테스트에 넣지 않는다.
"""

import json
import random
import re

import pandas as pd
import pytest

from core.diagnostics import diagnose
from core.question_types import get_question_type
from core.services import GenerationService
from core.strategies import (
    DistributionElicitationStrategy,
    GenerationContext,
    _call_with_retry,
)

# 복합 보기 라벨 ("/"를 포함해 LLM이 쪼개기 쉬운 보기).
HOLDOUT = "어느 쪽도 아니다 / 잘 모르겠다(의견 유보)"


@pytest.fixture
def q_holdout():
    """유보 복합 보기를 포함한 단일선택 문항."""
    return {"id": "Q1", "text": "이 정책에 찬성하십니까?",
            "type": "single",
            "options": ["찬성", "반대", HOLDOUT]}


@pytest.fixture
def persona_dims():
    return {"성별": {"남성": 50.0, "여성": 50.0},
            "연령대": {"20대": 50.0, "30대": 50.0}}


def _profile_count(prompt: str) -> int:
    """분포 추출 프롬프트 안의 응답자 프로필 수 (test_core.py와 같은 방식)."""
    return len(re.findall(r"(?m)^\d+\.\s", prompt))


def _unmatched_keys_mass(unmatched):
    """unmatched를 (keys, total_mass)로 정규화한다.

    허용 형태: {"keys": [...], "total_mass": float} (services 계측 계약) 또는
    {raw_key: mass} (probs_from_elicited 계약).
    """
    if isinstance(unmatched, dict) and "keys" in unmatched:
        return list(unmatched["keys"]), float(unmatched.get("total_mass", 0.0))
    if isinstance(unmatched, dict):
        return list(unmatched.keys()), float(sum(unmatched.values()))
    raise AssertionError(f"알 수 없는 unmatched 형태: {unmatched!r}")


def _resolve_report(svc, result):
    """generate_responses 결과에서 (df, estimation, execution)을 찾는다.

    통합본의 반환 형태가 dict({"df", "estimation", "execution"})인지,
    DataFrame(.attrs) + 서비스 속성인지 STATUS.md만으로는 확정할 수 없어
    두 경로를 모두 허용한다. 어느 경로에도 계측이 없으면 실패한다.
    """
    df = estimation = execution = None
    if isinstance(result, dict):
        estimation = result.get("estimation")
        execution = result.get("execution")
        df = result.get("df") or result.get("dataframe") or result.get("data")
    if isinstance(result, pd.DataFrame):
        df = result
        attrs = getattr(result, "attrs", None) or {}
        estimation = estimation or attrs.get("estimation")
        execution = execution or attrs.get("execution")
    if estimation is None:
        estimation = getattr(svc, "last_estimation", None)
    if execution is None:
        execution = getattr(svc, "last_execution", None)
    meta = getattr(svc, "last_gen_meta", None)
    if isinstance(meta, dict):
        estimation = estimation or meta.get("estimation")
        execution = execution or meta.get("execution")
    assert df is not None, "generate_responses 결과에서 DataFrame을 찾지 못함"
    assert isinstance(estimation, dict), (
        f"result['estimation']이 없음: {type(result).__name__} / "
        f"last_estimation={getattr(svc, 'last_estimation', None)!r}")
    assert isinstance(execution, dict), (
        f"result['execution']이 없음: {type(result).__name__} / "
        f"last_execution={getattr(svc, 'last_execution', None)!r}")
    return df, estimation, execution


# ----------------------------------------------------------------------------
# 0. 임계값 상수
# ----------------------------------------------------------------------------
def test_unmatched_mass_threshold_is_0_02():
    from core.question_types import UNMATCHED_MASS_THRESHOLD
    assert UNMATCHED_MASS_THRESHOLD == pytest.approx(0.02)


# ----------------------------------------------------------------------------
# 1. 복합 라벨 분할 키 병합
# ----------------------------------------------------------------------------
def test_split_keys_merge_into_composite_option(q_holdout):
    """쪼개진 raw 키 {"어느 쪽도 아니다", "잘 모르겠다"}가 유보 보기로 병합돼
    질량이 0이 되지 않는다."""
    qtype = get_question_type("single")
    probs, unmatched = qtype.probs_from_elicited(q_holdout, {
        "찬성": 0.45, "반대": 0.45,
        "어느 쪽도 아니다": 0.05, "잘 모르겠다": 0.05,
    })
    assert isinstance(probs, dict) and isinstance(unmatched, dict)
    assert probs[HOLDOUT] == pytest.approx(0.10)
    assert "어느 쪽도 아니다" not in unmatched
    assert "잘 모르겠다" not in unmatched
    assert sum(unmatched.values()) == pytest.approx(0.0)
    # 질량 보존: 들어간 질량이 probs+unmatched에 그대로 남아 있다.
    assert sum(probs.values()) + sum(unmatched.values()) == pytest.approx(1.0)


# ----------------------------------------------------------------------------
# 2. 공백/괄호 변형 매칭
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("raw_key", [
    "잘 모르겠다",                    # 괄호 없는 축약형
    "  잘 모르겠다(의견 유보)  ",    # 앞뒤 공백
    "잘 모르겠다（의견 유보）",      # 전각 괄호
    "　잘 모르겠다(의견 유보)",      # 전각 공백 + 괄호형
    HOLDOUT,                         # 정확 일치 (대조군)
])
def test_whitespace_and_paren_variants_match(q_holdout, raw_key):
    qtype = get_question_type("single")
    probs, unmatched = qtype.probs_from_elicited(q_holdout, {
        "찬성": 0.5, "반대": 0.3, raw_key: 0.2,
    })
    assert probs[HOLDOUT] == pytest.approx(0.2)
    assert raw_key not in unmatched
    assert sum(unmatched.values()) == pytest.approx(0.0)


# ----------------------------------------------------------------------------
# 3. 미매칭 질량 임계 초과 시 배치 재시도
# ----------------------------------------------------------------------------
def _two_personas():
    return [{"성별": "남성", "연령대": "20대"},
            {"성별": "여성", "연령대": "30대"}]


def test_batch_retry_on_unmatched_mass_then_success(q_holdout):
    """1차 호출은 미매칭 질량 0.2인 쪼개진 키 JSON, 2차는 깨끗한 키 JSON이면
    전략 generate가 재시도 끝에 성공한다."""
    calls = {"n": 0}

    def stub(pid, prompt, model):
        calls["n"] += 1
        n = max(_profile_count(prompt), 1)
        if calls["n"] == 1:
            row = {"Q1": {"찬성": 0.4, "반대": 0.4,
                          "어느 쪽도 아니다/잘 모르겠다": 0.2}}
        else:
            row = {"Q1": {"찬성": 0.4, "반대": 0.4, HOLDOUT: 0.2}}
        return json.dumps([row for _ in range(n)], ensure_ascii=False)

    ctx = GenerationContext(provider_id="gemini", model="dummy",
                            generate_fn=stub)
    out = DistributionElicitationStrategy().generate(
        ctx, [q_holdout], _two_personas(), random.Random(1))
    assert len(out) == 2
    assert calls["n"] == 2, "미매칭 초과 시 재시도가 일어나야 한다"
    assert ctx.last_call_attempts == 2
    # 최종 배치의 추정·미매칭이 컨텍스트에 기록된다.
    assert len(ctx.batch_estimates) == 2
    assert len(ctx.batch_unmatched) == 2
    assert all(not u["Q1"] for u in ctx.batch_unmatched)
    assert all(o["Q1"] in q_holdout["options"] for o in out)


def test_call_with_retry_check_raises_then_retries():
    """_call_with_retry(check=...) — check이 ValueError를 던지면 재시도한다."""
    calls = {"n": 0}

    def stub(pid, prompt, model):
        calls["n"] += 1
        if calls["n"] == 1:
            rows = [{"Q1": {"모름": 0.5}}, {"Q1": {"모름": 0.5}}]
        else:
            rows = [{"Q1": {"찬성": 0.5}}, {"Q1": {"반대": 0.5}}]
        return json.dumps(rows, ensure_ascii=False)

    def check(parsed):
        for row in parsed:
            if isinstance(row, dict) and "모름" in (row.get("Q1") or {}):
                raise ValueError("미매칭 질량 초과")

    ctx = GenerationContext(provider_id="gemini", model="dummy",
                            generate_fn=stub)
    parsed = _call_with_retry(ctx, "prompt", 2, check=check)
    assert len(parsed) == 2
    assert calls["n"] == 2
    assert ctx.last_call_attempts == 2


def test_batch_retry_exhausted_raises_value_error(q_holdout):
    """3회 전부 미매칭 임계를 초과하면 ValueError가 올라온다."""
    calls = {"n": 0}

    def stub(pid, prompt, model):
        calls["n"] += 1
        n = max(_profile_count(prompt), 1)
        row = {"Q1": {"찬성": 0.4, "반대": 0.4,
                      "어느 쪽도 아니다/잘 모르겠다": 0.2}}
        return json.dumps([row for _ in range(n)], ensure_ascii=False)

    ctx = GenerationContext(provider_id="gemini", model="dummy",
                            generate_fn=stub)
    with pytest.raises(ValueError):
        DistributionElicitationStrategy().generate(
            ctx, [q_holdout], _two_personas(), random.Random(1))
    assert calls["n"] == 3, "최대 3회까지 시도해야 한다"
    assert ctx.last_call_attempts == 3


# ----------------------------------------------------------------------------
# 4. 구조적 0 플래그
# ----------------------------------------------------------------------------
def _answered_df():
    # 유보 보기는 단 한 번도 출현하지 않음 (찬성 10 / 반대 10).
    return pd.DataFrame({"Q1": ["찬성"] * 10 + ["반대"] * 10})


def test_structural_zero_flagged(q_holdout):
    """estimated>=0.02인데 출현 0인 보기에 '보기 미출현(구조적 0)' finding이
    나고 '매칭 손실 의심'이 함께 표시된다."""
    estimated = {"Q1": {"찬성": 0.45, "반대": 0.45, HOLDOUT: 0.10}}
    findings = diagnose(_answered_df(), [q_holdout], estimated=estimated)
    hits = [f for f in findings
            if "보기 미출현(구조적 0)" in str(f.get("문제", ""))]
    assert hits, f"구조적 0 finding이 없음: {findings!r}"
    assert any("매칭 손실 의심" in " ".join(str(v) for v in h.values())
               for h in hits)
    assert any(HOLDOUT in " ".join(str(v) for v in h.values())
               for h in hits)


def test_structural_zero_not_flagged_below_threshold(q_holdout):
    """추정확률이 임계(0.02) 미만이면 구조적 0 플래그가 나지 않는다."""
    estimated = {"Q1": {"찬성": 0.49, "반대": 0.50, HOLDOUT: 0.01}}
    findings = diagnose(_answered_df(), [q_holdout], estimated=estimated)
    assert not [f for f in findings
                if "보기 미출현(구조적 0)" in str(f.get("문제", ""))]


def test_structural_zero_not_flagged_without_estimated(q_holdout):
    """estimated를 안 주면(기존 호출) 구조적 0 진단은 건너뛴다."""
    findings = diagnose(_answered_df(), [q_holdout])
    assert not [f for f in findings
                if "보기 미출현(구조적 0)" in str(f.get("문제", ""))]


# ----------------------------------------------------------------------------
# 5. 계측 3항목 + 실행 정보
# ----------------------------------------------------------------------------
def test_generate_responses_reports_estimation_and_execution(
        q_holdout, persona_dims):
    """generate_responses 결과에 estimation 3항목과 execution 2항목이 있다."""
    def stub(pid, prompt, model):
        n = max(_profile_count(prompt), 1)
        row = {"Q1": {"찬성": 0.7, "반대": 0.2, HOLDOUT: 0.1}}
        return json.dumps([row for _ in range(n)], ensure_ascii=False)

    svc = GenerationService()
    result = svc.generate_responses(
        [q_holdout], persona_dims, 4,
        provider_id="gemini", model="dummy",
        batch_size=4, seed=1, generate_fn=stub,
    )
    df, estimation, execution = _resolve_report(svc, result)
    assert len(df) == 4
    assert set(df["Q1"].unique()) <= set(q_holdout["options"])

    assert "mean_estimated_probs" in estimation
    assert "unmatched" in estimation
    assert "retry_counts" in estimation
    assert "fallback_used" in execution
    assert "key_start_index" in execution
    assert isinstance(execution["fallback_used"], bool)
    assert execution["key_start_index"] is None or isinstance(
        execution["key_start_index"], int)

    # 깨끗한 입력이므로 미매칭 질량은 0, 재시도도 0이다.
    keys, mass = _unmatched_keys_mass(estimation["unmatched"])
    assert keys == []
    assert mass == pytest.approx(0.0)
    rc = estimation["retry_counts"]
    if isinstance(rc, dict):
        assert all(v == 0 for v in rc.values())
    elif isinstance(rc, list):
        assert all(v == 0 for v in rc)
    else:
        assert rc == 0

    # 평균 추정확률이 스텁이 준 분포와 일치한다.
    mean = estimation["mean_estimated_probs"]
    mean_q1 = mean.get("Q1", mean) if isinstance(mean, dict) else mean
    assert mean_q1["찬성"] == pytest.approx(0.7)
    assert mean_q1["반대"] == pytest.approx(0.2)
    assert mean_q1[HOLDOUT] == pytest.approx(0.1)


def test_get_last_call_info_shape():
    """providers.get_last_call_info()가 start_key_index·fallback_used를 준다."""
    from core.providers import get_last_call_info
    info = get_last_call_info()
    assert "start_key_index" in info
    assert "fallback_used" in info
    assert info["start_key_index"] is None or isinstance(
        info["start_key_index"], int)
    assert isinstance(info["fallback_used"], bool)


# ----------------------------------------------------------------------------
# 6. 모호한 키는 unmatched에 남는다
# ----------------------------------------------------------------------------
def test_ambiguous_key_stays_unmatched():
    """2개 이상 보기에 걸리는 키는 어느 보기에도 질량을 주지 않고
    unmatched에 남는다."""
    q = {"id": "Q1", "text": "두 안 중 어느 쪽입니까?",
         "type": "single",
         "options": ["A안 / B안(혼합)", "A안 / B안(기타)", "찬성", "반대"]}
    qtype = get_question_type("single")

    # (b) 조각 모호: "A안"은 두 복합 보기의 "/" 조각이다.
    probs, unmatched = qtype.probs_from_elicited(
        q, {"A안": 0.2, "찬성": 0.8})
    assert "A안" in unmatched
    assert unmatched["A안"] == pytest.approx(0.2)
    assert probs["A안 / B안(혼합)"] == pytest.approx(0.0)
    assert probs["A안 / B안(기타)"] == pytest.approx(0.0)
    assert sum(probs.values()) + sum(unmatched.values()) == pytest.approx(1.0)

    # (c) 부분일치 모호: "A안 / B안"은 두 복합 보기의 부분 문자열이다.
    probs2, unmatched2 = qtype.probs_from_elicited(
        q, {"A안 / B안": 0.2, "찬성": 0.8})
    assert "A안 / B안" in unmatched2
    assert unmatched2["A안 / B안"] == pytest.approx(0.2)
    assert probs2["A안 / B안(혼합)"] == pytest.approx(0.0)
    assert probs2["A안 / B안(기타)"] == pytest.approx(0.0)


# ----------------------------------------------------------------------------
# 부록: prompt_fragment — 줄바꿈·번호 목록 + 그대로 복사 지시
# ----------------------------------------------------------------------------
def test_prompt_fragment_lists_options_numbered(q_holdout):
    """보기를 '/' 한 줄로 붙이지 않고 줄바꿈·번호 목록으로 제시하며,
    보기 문자열을 목록 그대로 복사하라는 지시가 있다."""
    frag = get_question_type("single").prompt_fragment(q_holdout)
    assert "\n" in frag
    assert "1) 찬성" in frag
    assert "3) " + HOLDOUT in frag
    assert "그대로 복사" in frag
    assert "찬성/반대" not in frag
