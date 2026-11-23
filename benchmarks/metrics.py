"""벤치마크 지표 계산 (benchmarks/metrics.py)

고도화 계획 §3 축 B·§6 지표 정의에 따른 순수 계산 모듈.
실측 분포와 합성 분포(문항별 {보기: 비율})를 받아 지표를 계산한다.

- 평균 오차: 서수 문항만 (|합성 평균 − 실측 평균|의 평균)
- SD 비율: 합성 표준편차 / 실측 표준편차 (서수 문항 평균)
- 문항별 변이율 VR: 1 − 최빈 선택지 비율 (문항별 지표)
- 극단 동질화 문항 비율: VR < 0.05인 문항(한 선택지에 95% 초과로 쏠린 문항)이
  단일·리커트 문항에서 차지하는 비율. 다중선택 문항은 보기별 선택률의 합이
  1이 아니라 최빈 점유율 해석이 달라 극단 동질화 분모에서 제외한다
  (문항별 VR 자체는 다중선택에도 계산한다).
  ※ 피주임 검토 반영 (2026-11-23): 문항별 VR과 극단 동질화 문항 비율은
  다른 지표이므로 구분해서 표기한다.
- JSD: 합성 분포와 실측 분포 간 Jensen-Shannon Divergence (문항 평균)
- 소수 의견 재현율: 실측 10% 이상 선택지 중 합성 5% 이상인 비율
"""

from __future__ import annotations

import math
import statistics

from core.variance import jsd

# 서수 코딩 (평균·표준편차 계산용)
LIKERT_CODES = {"전혀 그렇지 않다": 1, "그렇지 않다": 2, "보통이다": 3,
                "그렇다": 4, "매우 그렇다": 5}
Q6_CODES = {"1회 이하": 1, "2~3회": 2, "4~5회": 3, "6회 이상": 4}

MULTI_SEP = ";"


def ordinal_codes(q: dict) -> dict[str, float] | None:
    """문항의 서수 코딩. 명목형(single Q1 등)이면 None."""
    if q["type"] == "likert":
        return dict(LIKERT_CODES)
    if q["id"] == "Q6":
        return dict(Q6_CODES)
    return None


def dist_from_values(values: list[str], q: dict) -> dict[str, float]:
    """응답값 목록에서 문항 분포를 만든다.

    single/likert: 보기별 비율. multi: 보기별 선택률(응답자 기준, 합이 1이 아님).
    """
    from config import LIKERT_OPTIONS
    if q["type"] == "multi":
        opts = [str(o) for o in q.get("options") or []]
        n = len(values)
        counts = {opt: 0 for opt in opts}
        for v in values:
            for part in str(v).split(MULTI_SEP):
                part = part.strip()
                if part in counts:
                    counts[part] += 1
        return {opt: (counts[opt] / n if n else 0.0) for opt in opts}
    opts = list(LIKERT_OPTIONS) if q["type"] == "likert" \
        else [str(o) for o in q.get("options") or []]
    n = len(values)
    counts = {opt: 0 for opt in opts}
    for v in values:
        v = str(v).strip()
        if v in counts:
            counts[v] += 1
    return {opt: (counts[opt] / n if n else 0.0) for opt in opts}


def _bernoulli_jsd(p: float, q: float) -> float:
    """두 Bernoulli 분포(p, 1-p)와 (q, 1-q) 사이의 JSD."""
    return jsd({"yes": p, "no": 1 - p}, {"yes": q, "no": 1 - q})


def _mean_sd(dist: dict[str, float], codes: dict[str, float]) -> tuple[float, float]:
    """분포와 서수 코딩에서 평균·표준편차를 계산한다."""
    mean = sum(codes[o] * p for o, p in dist.items() if o in codes)
    var = sum(p * (codes[o] - mean) ** 2 for o, p in dist.items() if o in codes)
    return mean, math.sqrt(var)


def compute_metrics(real_dists: dict[str, dict[str, float]],
                    synth_dists: dict[str, dict[str, float]],
                    questions: list[dict]) -> dict:
    """실측/합성 분포 딕셔너리에서 전체 지표를 계산한다."""
    per_q: dict[str, dict] = {}
    mean_errs: list[float] = []
    sd_ratios: list[float] = []
    vr_synth_list: list[float] = []
    vr_real_list: list[float] = []
    extreme_flags: list[bool] = []
    extreme_real_flags: list[bool] = []
    jsds: list[float] = []
    minority_hits = 0
    minority_total = 0

    for q in questions:
        qid = q["id"]
        if qid not in real_dists or qid not in synth_dists:
            continue
        real = real_dists[qid]
        synth = synth_dists[qid]
        row: dict = {"real": real, "synth": synth}
        codes = ordinal_codes(q)
        if q["type"] == "multi":
            # 보기별 선택률 기준
            errs = [abs(synth[o] - real[o]) for o in real]
            row["mean_abs_diff"] = statistics.mean(errs) if errs else 0.0
            ratios = []
            for o in real:
                rs = math.sqrt(real[o] * (1 - real[o])) if 0 < real[o] < 1 else 0.0
                ss = math.sqrt(synth[o] * (1 - synth[o])) if 0 < synth[o] < 1 else 0.0
                ratios.append(ss / rs if rs > 0 else (1.0 if ss == 0 else float("inf")))
            row["sd_ratio"] = statistics.mean(ratios) if ratios else 1.0
            row["jsd"] = statistics.mean(
                [_bernoulli_jsd(real[o], synth[o]) for o in real]) if real else 0.0
            row["max_share_synth"] = max(synth.values()) if synth else 0.0
            row["max_share_real"] = max(real.values()) if real else 0.0
            row["vr_synth"] = 1.0 - row["max_share_synth"]
            row["vr_real"] = 1.0 - row["max_share_real"]
            vr_synth_list.append(row["vr_synth"])
            vr_real_list.append(row["vr_real"])
            sd_ratios.append(row["sd_ratio"])
        else:
            if codes:
                rm, rs = _mean_sd(real, codes)
                sm, ss = _mean_sd(synth, codes)
                row["mean_real"] = rm
                row["mean_synth"] = sm
                row["mean_err"] = abs(sm - rm)
                row["sd_real"] = rs
                row["sd_synth"] = ss
                row["sd_ratio"] = (ss / rs) if rs > 0 else 1.0
                mean_errs.append(row["mean_err"])
                sd_ratios.append(row["sd_ratio"])
            row["jsd"] = jsd(real, synth)
            row["max_share_synth"] = max(synth.values()) if synth else 0.0
            row["max_share_real"] = max(real.values()) if real else 0.0
            row["vr_synth"] = 1.0 - row["max_share_synth"]
            row["vr_real"] = 1.0 - row["max_share_real"]
            vr_synth_list.append(row["vr_synth"])
            vr_real_list.append(row["vr_real"])
            # 극단 동질화: VR < 0.05 ⟺ 최빈 점유율 > 0.95 (단일·리커트만)
            extreme_flags.append(row["vr_synth"] < 0.05)
            extreme_real_flags.append(row["vr_real"] < 0.05)
        jsds.append(row["jsd"])
        for opt, rp in real.items():
            if rp >= 0.10:
                minority_total += 1
                if synth.get(opt, 0.0) >= 0.05:
                    minority_hits += 1
        per_q[qid] = row

    summary = {
        "mean_error": statistics.mean(mean_errs) if mean_errs else None,
        "mean_error_n": len(mean_errs),
        "sd_ratio": statistics.mean(sd_ratios) if sd_ratios else None,
        "sd_ratio_n": len(sd_ratios),
        # 문항별 VR 평균 (전 문항 — 다중선택 포함)
        "vr_mean_synth": (statistics.mean(vr_synth_list)
                          if vr_synth_list else None),
        "vr_mean_real": (statistics.mean(vr_real_list)
                         if vr_real_list else None),
        # 극단 동질화 문항 비율 (VR < 0.05인 문항 비중 — 단일·리커트만)
        "extreme_homog_rate_synth": (
            (sum(extreme_flags) / len(extreme_flags)) if extreme_flags else 0.0),
        "extreme_homog_rate_real": (
            (sum(extreme_real_flags) / len(extreme_real_flags))
            if extreme_real_flags else 0.0),
        "extreme_homog_n": len(extreme_flags),
        "jsd": statistics.mean(jsds) if jsds else None,
        "minority_recall": (minority_hits / minority_total) if minority_total else None,
        "minority_hits": minority_hits,
        "minority_total": minority_total,
    }
    return {"per_question": per_q, "summary": summary}
