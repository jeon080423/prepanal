"""분산 복원 순수 함수 (core/variance.py)

고도화 계획 축 A (분산 복원)의 실험용 순수 함수 모음.
LLM·파일 I/O·전역 상태에 의존하지 않으며, 기존 전략 클래스에 직접 끼워넣지
않고 여기서 검증한 뒤 연결한다 (연결 방법은 benchmarks/variance-experiments.md
"연결 방법" 절 참조).

함수 목록:
- normalize_probs: 확률 딕셔너리 정규화 (합 1, 음수 제거)
- jsd: 두 분포 간 Jensen-Shannon Divergence (밑 2, 0~1)
- flatten_distribution: 균등분포 쪽으로 선형 보간 (A-3, 확률 평탄화)
- temperature_scale: p^(1/T) 거듭제곱 보정 (A-3, temperature scaling)
- calibrate_flattening: 학습 분포와 대조해 보정 계수 선택 (A-3)
- assign_tendency: 페르소나별 응답 성향 추첨 (A-2, 혼합 샘플링)
- mixture_adjust: 성향에 따라 확률분포를 재가중 (A-2)
- sample_from_probs: 정규화된 분포에서 1건 샘플링
"""

from __future__ import annotations

import math
import random

# A-2 혼합 샘플링의 성향 종류
TENDENCY_MAJORITY = "majority"   # 다수 의견형: 최빈값을 더 강화
TENDENCY_MINORITY = "minority"   # 소수 의견형: 비(非)최빈값을 부스트
TENDENCY_EXTREME = "extreme"     # 극단형: 양 끝단(서수 문항) 또는 최소 확률 보기를 부스트

TENDENCIES = (TENDENCY_MAJORITY, TENDENCY_MINORITY, TENDENCY_EXTREME)

# 실측 분포에서 추정한 기본 성향 비율 (고도화 계획 A-2: "성향 비율은 실측 분포에서 추정").
# 아래 기본값은 임시 벤치마크(샘플 200건)의 최빈값 점유율에서 역산한 출발점이며,
# 벤치마크가 바뀌면 다시 추정해야 한다. (측정 근거: benchmarks/variance-experiments.md)
DEFAULT_TENDENCY_WEIGHTS = {
    TENDENCY_MAJORITY: 0.6,
    TENDENCY_MINORITY: 0.3,
    TENDENCY_EXTREME: 0.1,
}


# ----------------------------------------------------------------------------
# 기본 유틸리티
# ----------------------------------------------------------------------------
def normalize_probs(probs: dict[str, float],
                    options: list[str] | None = None) -> dict[str, float]:
    """확률 딕셔너리를 정규화한다. 음수는 0으로, 합이 0이면 균등분포."""
    opts = list(options) if options is not None else list(probs.keys())
    norm = {opt: max(0.0, float(probs.get(opt, 0) or 0)) for opt in opts}
    total = sum(norm.values())
    if total <= 0 or not opts:
        return {opt: 1.0 / len(opts) for opt in opts} if opts else {}
    return {opt: v / total for opt, v in norm.items()}


def jsd(p: dict[str, float], q: dict[str, float],
        options: list[str] | None = None) -> float:
    """두 분포 간 Jensen-Shannon Divergence (밑 2 로그, 0~1).

    p, q는 {보기: 확률} 딕셔너리. options가 주어지면 그 보기를 기준으로 맞춘다.
    """
    opts = list(options) if options is not None \
        else sorted(set(p) | set(q))
    pn = normalize_probs(p, opts)
    qn = normalize_probs(q, opts)
    total = 0.0
    for opt in opts:
        pi, qi = pn[opt], qn[opt]
        m = 0.5 * (pi + qi)
        for v in (pi, qi):
            if v > 0:
                total += 0.5 * v * math.log2(v / m)
    return total


def sample_from_probs(probs: dict[str, float], rng: random.Random,
                      options: list[str] | None = None) -> str:
    """정규화된 분포에서 가중 샘플링으로 1건을 뽑는다."""
    dist = normalize_probs(probs, options)
    total = sum(dist.values())
    r = rng.uniform(0, total)
    upto = 0.0
    for k, v in dist.items():
        upto += v
        if r <= upto:
            return k
    return next(reversed(dist))


# ----------------------------------------------------------------------------
# A-3 반최빈값 보정
# ----------------------------------------------------------------------------
def flatten_distribution(probs: dict[str, float], alpha: float,
                         options: list[str] | None = None) -> dict[str, float]:
    """확률 평탄화: p' = (1 - alpha) * p + alpha * uniform.

    alpha=0이면 원본 그대로, alpha=1이면 균등분포. alpha in [0, 1].
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha는 0~1 사이여야 합니다: {alpha}")
    dist = normalize_probs(probs, options)
    n = len(dist)
    if n == 0:
        return {}
    return {opt: (1.0 - alpha) * p + alpha / n for opt, p in dist.items()}


def temperature_scale(probs: dict[str, float], temp: float,
                      options: list[str] | None = None) -> dict[str, float]:
    """Temperature scaling: p' ∝ p^(1/T).

    T=1이면 원본 그대로, T>1이면 평탄해지고(분산 복원 방향),
    T<1이면 뾰족해진다. temp > 0.
    """
    if temp <= 0:
        raise ValueError(f"temp는 0보다 커야 합니다: {temp}")
    dist = normalize_probs(probs, options)
    if temp == 1.0:
        return dict(dist)
    powered = {opt: p ** (1.0 / temp) for opt, p in dist.items()}
    total = sum(powered.values())
    if total <= 0:
        return {opt: 1.0 / len(dist) for opt in dist}
    return {opt: v / total for opt, v in powered.items()}


def calibrate_flattening(elicited: dict[str, float],
                         learned: dict[str, float],
                         alphas: list[float] | None = None,
                         options: list[str] | None = None
                         ) -> tuple[float, float]:
    """학습 분포(learned)와 대조해 평탄화 계수 alpha를 고른다.

    alpha 그리드를 훑어 보정 후 분포와 학습 분포 사이의 JSD가 가장 작아지는
    (alpha, JSD)를 반환한다. 학습 분포가 없는 문항에는 쓰지 않는다.

    주의: 이 함수는 보정 계수를 '정하는' 용도다. 학습 분포에 직접 맞추는
    것이므로, 벤치마크 문항과 보정 대상 문항이 같으면 과적합(벤치마크에 맞춘
    계수)이 된다. 실전에서는 별도 문항·별도 데이터로 계수를 정한 뒤 적용한다.
    """
    grid = alphas if alphas is not None else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    best_alpha, best_jsd = grid[0], float("inf")
    for alpha in grid:
        corrected = flatten_distribution(elicited, alpha, options)
        d = jsd(corrected, learned, options)
        if d < best_jsd:
            best_alpha, best_jsd = alpha, d
    return best_alpha, best_jsd


def calibrate_temperature(elicited: dict[str, float],
                          learned: dict[str, float],
                          temps: list[float] | None = None,
                          options: list[str] | None = None
                          ) -> tuple[float, float]:
    """학습 분포와 대조해 temperature scaling 계수 T를 고른다.

    calibrate_flattening과 같은 용도 (T 그리드 버전). (best_T, JSD) 반환.
    """
    grid = temps if temps is not None else [1.0, 1.5, 2.0, 2.5, 3.0]
    best_t, best_jsd = grid[0], float("inf")
    for t in grid:
        corrected = temperature_scale(elicited, t, options)
        d = jsd(corrected, learned, options)
        if d < best_jsd:
            best_t, best_jsd = t, d
    return best_t, best_jsd


# ----------------------------------------------------------------------------
# A-2 혼합 샘플링
# ----------------------------------------------------------------------------
def assign_tendency(rng: random.Random,
                    weights: dict[str, float] | None = None) -> str:
    """페르소나 1명의 응답 성향을 가중 추첨한다."""
    w = dict(weights) if weights is not None else dict(DEFAULT_TENDENCY_WEIGHTS)
    total = sum(max(0.0, float(v)) for v in w.values())
    if total <= 0:
        raise ValueError("성향 가중치 합이 0입니다.")
    r = rng.uniform(0, total)
    upto = 0.0
    for tend in TENDENCIES:
        upto += max(0.0, float(w.get(tend, 0) or 0))
        if r <= upto:
            return tend
    return TENDENCIES[-1]


def mixture_adjust(probs: dict[str, float], tendency: str,
                   strength: float = 1.0,
                   options: list[str] | None = None) -> dict[str, float]:
    """성향에 따라 확률분포를 재가중한다. strength in [0, 1] (0=변화 없음).

    - majority: 최빈값 보기의 확률을 (1 + strength)배로 키운 뒤 정규화.
    - minority: 최빈값을 제외한 나머지 보기를 (1 + strength)배로 키운다.
    - extreme: 양 끝단 보기(서수 문항 가정: options 순서의 첫·마지막)를
      (1 + strength)배로 키운다. 보기 순서가 서수가 아닌 문항(명목형)에서는
      확률이 가장 작은 보기를 키우는 것으로 대체한다.
    """
    if tendency not in TENDENCIES:
        raise ValueError(f"알 수 없는 성향: {tendency}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength는 0~1 사이여야 합니다: {strength}")
    dist = normalize_probs(probs, options)
    if strength == 0.0 or not dist:
        return dict(dist)
    opts = list(dist.keys())
    boost = 1.0 + strength
    adjusted = dict(dist)
    if tendency == TENDENCY_MAJORITY:
        top = max(dist, key=lambda o: dist[o])
        adjusted[top] = dist[top] * boost
    elif tendency == TENDENCY_MINORITY:
        top = max(dist, key=lambda o: dist[o])
        for opt in opts:
            if opt != top:
                adjusted[opt] = dist[opt] * boost
    else:  # extreme
        if len(opts) >= 2:
            # 서수 문항이면 양 끝단, 명목형이면 최소 확률 보기 2개를 키운다.
            # 여기서는 options 순서의 양 끝단을 쓴다 (리커트·Q6 등 서수 문항용).
            for opt in (opts[0], opts[-1]):
                adjusted[opt] = dist[opt] * boost
        else:
            adjusted[opts[0]] = dist[opts[0]] * boost
    total = sum(adjusted.values())
    return {opt: v / total for opt, v in adjusted.items()}


def mixture_sample(probs: dict[str, float], tendency: str,
                   rng: random.Random, strength: float = 1.0,
                   options: list[str] | None = None) -> str:
    """성향을 적용한 뒤 분포에서 1건을 샘플링한다 (A-2 한 번에)."""
    adjusted = mixture_adjust(probs, tendency, strength, options)
    return sample_from_probs(adjusted, rng, options)
