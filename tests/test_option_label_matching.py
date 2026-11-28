# -*- coding: utf-8 -*-
"""유보·무당층 보기 라벨 매칭 회귀 테스트 (pickle/<이슈번호> 브랜치용 초안).

현재 d3b3d61e에서는 대부분 RED다(버그를 잡는 테스트). C1 배포 후 GREEN이 되어야 한다.
저장소에는 `prepanal/tests/test_option_label_matching.py` 로 들어간다.

검증하는 계약
1. `_build_dist_prompt()`는 보기 라벨을 그대로 보존한다 — 라벨 안의 '/', ' / '가
   보기 구분자와 섞이지 않는다.
2. `probs_from_elicited()`는 라벨을 정확히 일치시키지 못해도 질량을 잃지 않는다:
   (a) 복합 라벨을 두 키로 쪼개 보낸 경우 두 확률을 합쳐 원래 보기에 붙인다
   (b) 공백·NFKC 변형을 흡수한다
   (c) 괄호 부가어 생략("잘 모르겠다")을 흡수한다
3. 유보·무당 보기는 0으로 붕괴하지 않는다 — 300명 샘플에서 실제로 출현한다.
"""
from __future__ import annotations

import random

import pytest

from core.question_types import get_question_type
from core.strategies import _build_dist_prompt

Q1 = {
    "id": "Q1",
    "type": "single",
    "text": "이재명 대통령 직무 수행 평가",
    "options": ["잘하고 있다", "잘못하고 있다",
                "어느 쪽도 아니다 / 잘 모르겠다(의견 유보)"],
}
Q2 = {
    "id": "Q2",
    "type": "single",
    "text": "정당 지지도",
    "options": ["더불어민주당", "국민의힘", "조국혁신당", "개혁신당", "진보당",
                "이외 정당/단체", "지지하는 정당 없다 / 잘 모르겠다"],
}
Q3 = {
    "id": "Q3",
    "type": "single",
    "text": "12.3 비상계엄 내란 인식",
    "options": ["내란이다", "내란이 아니다", "잘 모르겠다(의견 유보)"],
}


# -- 1. 프롬프트가 라벨을 보존하는가 -----------------------------------------
_QUOTES = ("\"", "'", "「", "」", "“", "”")


def _label_is_standalone(prompt: str, label: str) -> bool:
    """라벨이 보기 하나로 '떼어 읽히게' 제시됐는지.

    구현 방식은 자유지만(줄바꿈, 번호 매김, 따옴표), 구분자로 붙인 한 줄 안에
    섞여 있으면 안 된다 — 모델이 라벨을 그대로 복사할 수 있어야 하기 때문.
    """
    import re
    # (a) 줄 시작(앞에 번호·불릿 허용)
    if re.search(r"(^|\n)[ \t]*(?:[-*]|[0-9]+[.)]|[①-⑳])?[ \t]*"
                 + re.escape(label), prompt):
        return True
    # (b) 따옴표로 감쌈
    return any(f"{a}{label}{b}" in prompt for a in _QUOTES for b in _QUOTES)


@pytest.mark.parametrize("q", [Q1, Q2, Q3])
def test_prompt_keeps_option_labels_verbatim(q):
    prompt = _build_dist_prompt([q], ["성별 여성, 연령대 40대, 지역 수도권"])
    for opt in q["options"]:
        assert opt in prompt, f"보기 라벨이 프롬프트에서 변형됐습니다: {opt!r}"
        assert _label_is_standalone(prompt, opt), (
            f"보기 라벨이 구분자에 붙어 있어 그대로 복사할 수 없습니다: {opt!r}")


def test_compound_label_not_joined_with_selected_delimiter():
    """보기 안에 ' / '가 있어도 문항 줄에서 보기 개수가 3개로 유지돼야 한다."""
    prompt = _build_dist_prompt([Q1], ["성별 여성, 연령대 40대"])
    assert _label_is_standalone(prompt, "어느 쪽도 아니다 / 잘 모르겠다(의견 유보)")


# -- 2. probs_from_elicited 가 질량을 잃지 않는가 ----------------------------
def _pct(q, raw, target, n=4000, seed=684):
    qtype = get_question_type(q["type"])
    probs = qtype.probs_from_elicited(q, raw)
    rng = random.Random(seed)
    hits = sum(1 for _ in range(n) if qtype.sample_from_elicited(q, probs, rng) == target)
    return hits / n * 100


def test_split_keys_are_merged_into_compound_option_q1():
    """모델이 복합 라벨을 두 키로 쪼개 보낸 경우(현재 0% 붕괴의 원인)."""
    raw = {"잘하고 있다": 0.47, "잘못하고 있다": 0.44,
           "어느 쪽도 아니다": 0.04, "잘 모르겠다(의견 유보)": 0.05}
    share = _pct(Q1, raw, "어느 쪽도 아니다 / 잘 모르겠다(의견 유보)")
    assert share > 5.0, f"유보 보기가 여전히 붕괴했습니다: {share:.1f}%"


def test_split_keys_are_merged_into_compound_option_q2():
    raw = {"더불어민주당": 0.32, "국민의힘": 0.30, "조국혁신당": 0.05,
           "개혁신당": 0.03, "진보당": 0.02, "이외 정당/단체": 0.01,
           "지지하는 정당 없다": 0.18, "잘 모르겠다": 0.09}
    share = _pct(Q2, raw, "지지하는 정당 없다 / 잘 모르겠다")
    assert share > 12.0, f"무당 보기가 여전히 붕괴했습니다: {share:.1f}%"


def test_whitespace_variant_is_matched():
    raw = {"내란이다": 0.5, "내란이  아니다": 0.3, "잘 모르겠다(의견 유보)": 0.2}
    share = _pct(Q3, raw, "잘 모르겠다(의견 유보)")
    assert share > 10.0


def test_parenthetical_omitted_is_matched():
    """모델이 '잘 모르겠다'만 보내도 괄호 부가어가 붙은 보기에 붙어야 한다."""
    raw = {"내란이다": 0.5, "내란이 아니다": 0.3, "잘 모르겠다": 0.2}
    share = _pct(Q3, raw, "잘 모르겠다(의견 유보)")
    assert share > 10.0


def test_slash_without_spaces_still_matched():
    """'이외 정당/단체' 처럼 공백 없는 '/'는 지금도 매칭된다(회귀 방지)."""
    raw = {"더불어민주당": 0.4, "국민의힘": 0.4, "조국혁신당": 0.05,
           "개혁신당": 0.05, "진보당": 0.03, "이외 정당/단체": 0.05,
           "지지하는 정당 없다 / 잘 모르겠다": 0.02}
    share = _pct(Q2, raw, "이외 정당/단체")
    assert share > 2.0


def test_unknown_key_alone_does_not_silently_shift_mass():
    """옵션과 전혀 다른 키만 오면 0으로 버리지 말고 실패로 알려야 한다(재시도 대상).

    C1에서 미매칭 질량이 임계(2%)를 넘으면 ValueError(재시도)로 처리하기로 했으므로,
    전량 미매칭은 예외 또는 균등분포 중 하나로 '명시적'이어야 한다. 조용히
    잘못된 보기로 몰리는 것만 금지한다.
    """
    qtype = get_question_type(Q3["type"])
    try:
        probs = qtype.probs_from_elicited(Q3, {"모름": 1.0})
    except ValueError:
        return  # 재시도 경로 — 정상
    total = sum(probs.values())
    assert total < 1e-9 or all(abs(v - 1 / 3) < 1e-6 for v in probs.values()), (
        f"미매칭 질량이 조용히 다른 보기로 재분배됐습니다: {probs}")


# -- 3. 90% 이상 미매칭인 배치를 재시도하는가 (C1 (c)) ------------------------
def test_low_match_batch_is_retried(monkeypatch):
    """첫 호출이 전부 미매칭이면 _call_with_retry가 다시 부르는지 확인한다."""
    from core import strategies

    calls = {"n": 0}

    def fake_generate(provider_id, prompt, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return '[{"Q3": {"모름": 1.0}}]'
        return '[{"Q3": {"내란이다": 0.5, "내란이 아니다": 0.3, "잘 모르겠다(의견 유보)": 0.2}}]'

    ctx = strategies.GenerationContext(provider_id="stub", model="stub",
                                       generate_fn=fake_generate)
    ctx.cache.clear()
    out = strategies.DistributionElicitationStrategy().generate(
        ctx, [Q3], [{"성별": "여성"}], random.Random(1))
    assert calls["n"] >= 2, "미매칭 배치를 재시도하지 않았습니다"
    assert out[0]["Q3"] in Q3["options"]
