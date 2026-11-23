"""core/variance.py 단위 테스트 (LLM 호출 없음 — 순수 함수만 검증)."""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.variance import (  # noqa: E402
    DEFAULT_TENDENCY_WEIGHTS,
    TENDENCY_EXTREME,
    TENDENCY_MAJORITY,
    TENDENCY_MINORITY,
    assign_tendency,
    calibrate_flattening,
    calibrate_temperature,
    flatten_distribution,
    jsd,
    mixture_adjust,
    mixture_sample,
    normalize_probs,
    sample_from_probs,
    temperature_scale,
)


def test_normalize_sums_to_one():
    d = normalize_probs({"a": 2.0, "b": 3.0, "c": 0.0})
    assert abs(sum(d.values()) - 1.0) < 1e-9
    assert abs(d["a"] - 0.4) < 1e-9


def test_normalize_all_zero_is_uniform():
    d = normalize_probs({"a": 0.0, "b": 0.0})
    assert d == {"a": 0.5, "b": 0.5}


def test_normalize_negative_clipped():
    d = normalize_probs({"a": -1.0, "b": 1.0})
    assert d["a"] == 0.0 and d["b"] == 1.0


def test_jsd_identical_is_zero():
    p = {"a": 0.7, "b": 0.3}
    assert jsd(p, p) < 1e-12


def test_jsd_disjoint_is_one():
    assert abs(jsd({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}) - 1.0) < 1e-9


def test_jsd_symmetric():
    p, q = {"a": 0.8, "b": 0.2}, {"a": 0.4, "b": 0.6}
    assert abs(jsd(p, q) - jsd(q, p)) < 1e-12


def test_flatten_alpha_zero_identity():
    p = {"a": 0.8, "b": 0.1, "c": 0.1}
    assert flatten_distribution(p, 0.0) == normalize_probs(p)


def test_flatten_alpha_one_uniform():
    p = {"a": 0.8, "b": 0.1, "c": 0.1}
    d = flatten_distribution(p, 1.0)
    assert all(abs(v - 1 / 3) < 1e-9 for v in d.values())


def test_flatten_reduces_modal_share():
    p = {"a": 0.9, "b": 0.05, "c": 0.05}
    d = flatten_distribution(p, 0.3)
    assert d["a"] < p["a"] and d["b"] > p["b"]
    assert abs(sum(d.values()) - 1.0) < 1e-9


def test_flatten_alpha_out_of_range():
    try:
        flatten_distribution({"a": 1.0}, 1.5)
    except ValueError:
        return
    raise AssertionError("alpha 범위 초과 시 ValueError가 나야 한다")


def test_temperature_scale_one_identity():
    p = {"a": 0.8, "b": 0.2}
    assert temperature_scale(p, 1.0) == normalize_probs(p)


def test_temperature_scale_high_flattens():
    p = {"a": 0.9, "b": 0.1}
    d = temperature_scale(p, 2.0)
    assert d["a"] < p["a"] and d["b"] > p["b"]
    assert abs(sum(d.values()) - 1.0) < 1e-9


def test_temperature_scale_low_sharpens():
    p = {"a": 0.6, "b": 0.4}
    d = temperature_scale(p, 0.5)
    assert d["a"] > p["a"]


def test_temperature_scale_invalid():
    try:
        temperature_scale({"a": 1.0}, 0.0)
    except ValueError:
        return
    raise AssertionError("temp<=0 시 ValueError가 나야 한다")


def test_calibrate_flattening_picks_smaller_jsd():
    # 학습 분포가 평탄하면 alpha가 커지는 쪽이 JSD가 작아진다.
    elicited = {"a": 0.9, "b": 0.05, "c": 0.05}
    learned = {"a": 0.4, "b": 0.3, "c": 0.3}
    alpha, d = calibrate_flattening(elicited, learned)
    assert alpha > 0.0
    assert d < jsd(elicited, learned)


def test_calibrate_flattening_exact_match_alpha_zero():
    p = {"a": 0.5, "b": 0.5}
    alpha, d = calibrate_flattening(p, p)
    assert alpha == 0.0 and d < 1e-12


def test_calibrate_temperature_picks_valid():
    elicited = {"a": 0.9, "b": 0.05, "c": 0.05}
    learned = {"a": 0.4, "b": 0.3, "c": 0.3}
    t, d = calibrate_temperature(elicited, learned)
    assert t > 1.0
    assert d < jsd(elicited, learned)


def test_assign_tendency_returns_known():
    rng = random.Random(0)
    for _ in range(50):
        assert assign_tendency(rng) in (
            TENDENCY_MAJORITY, TENDENCY_MINORITY, TENDENCY_EXTREME)


def test_assign_tendency_respects_weights():
    rng = random.Random(1)
    counts = {t: 0 for t in (TENDENCY_MAJORITY, TENDENCY_MINORITY, TENDENCY_EXTREME)}
    for _ in range(3000):
        counts[assign_tendency(rng)] += 1
    # 가중치 0.6/0.3/0.1 근방 (표본오차 허용 ±0.05)
    assert abs(counts[TENDENCY_MAJORITY] / 3000 - 0.6) < 0.05
    assert abs(counts[TENDENCY_MINORITY] / 3000 - 0.3) < 0.05
    assert abs(counts[TENDENCY_EXTREME] / 3000 - 0.1) < 0.05


def test_mixture_majority_boosts_modal():
    p = {"a": 0.6, "b": 0.25, "c": 0.15}
    d = mixture_adjust(p, TENDENCY_MAJORITY, strength=1.0)
    assert d["a"] > p["a"]


def test_mixture_minority_boosts_non_modal():
    p = {"a": 0.6, "b": 0.25, "c": 0.15}
    d = mixture_adjust(p, TENDENCY_MINORITY, strength=1.0)
    assert d["b"] > p["b"] and d["c"] > p["c"] and d["a"] < p["a"]


def test_mixture_extreme_boosts_endpoints():
    # 서수 문항 가정: options 순서의 양 끝단
    p = {"1": 0.1, "2": 0.3, "3": 0.4, "4": 0.1, "5": 0.1}
    d = mixture_adjust(p, TENDENCY_EXTREME, strength=1.0,
                       options=["1", "2", "3", "4", "5"])
    assert d["1"] > p["1"] and d["5"] > p["5"]


def test_mixture_strength_zero_identity():
    p = {"a": 0.6, "b": 0.4}
    assert mixture_adjust(p, TENDENCY_MINORITY, strength=0.0) == p


def test_mixture_unknown_tendency():
    try:
        mixture_adjust({"a": 1.0}, "weird", strength=0.5)
    except ValueError:
        return
    raise AssertionError("알 수 없는 성향은 ValueError")


def test_mixture_sample_returns_option():
    rng = random.Random(42)
    p = {"a": 0.7, "b": 0.3}
    for tend in (TENDENCY_MAJORITY, TENDENCY_MINORITY, TENDENCY_EXTREME):
        assert mixture_sample(p, tend, rng) in p


def test_sample_from_probs_deterministic_seed():
    p = {"a": 0.5, "b": 0.5}
    r1 = random.Random(7)
    r2 = random.Random(7)
    assert [sample_from_probs(p, r1) for _ in range(20)] == \
        [sample_from_probs(p, r2) for _ in range(20)]


def test_default_tendency_weights_sum_to_one():
    assert abs(sum(DEFAULT_TENDENCY_WEIGHTS.values()) - 1.0) < 1e-9
    assert set(DEFAULT_TENDENCY_WEIGHTS) == {
        TENDENCY_MAJORITY, TENDENCY_MINORITY, TENDENCY_EXTREME}
