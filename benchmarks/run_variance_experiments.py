"""Phase B 오프라인 실험 (benchmarks/run_variance_experiments.py)

베이스라인에서 포획한 LLM 추출 확률분포(elicited_baseline.json)를 재사용해
추가 LLM 호출 없이 A-3(반최빈값 보정)·A-2(혼합 샘플링)를 실험한다.
각 조건은 동일 시드로 재샘플링해 베이스라인 리플레이와 비교한다.

실행: cd ~/workspace/prepanel && python3 benchmarks/run_variance_experiments.py
산출물: benchmarks/variance_experiments.json (수치), 리포트는 variance-experiments.md
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LIKERT_OPTIONS  # noqa: E402
from core.variance import (  # noqa: E402
    DEFAULT_TENDENCY_WEIGHTS,
    assign_tendency,
    calibrate_flattening,
    flatten_distribution,
    mixture_adjust,
    normalize_probs,
    sample_from_probs,
    temperature_scale,
)
from metrics import MULTI_SEP, compute_metrics, dist_from_values  # noqa: E402

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
RESAMPLE_SEED = 99


def load_inputs():
    questions = json.loads((ROOT / "samples" / "sample_questionnaire.json").read_text())
    choice_qs = [q for q in questions if q["type"] != "open"]
    real_dists = json.loads((BASE / "real_dists.json").read_text())
    elicited = json.loads((BASE / "elicited_baseline.json").read_text())["rows"]
    return choice_qs, real_dists, elicited


def sample_answers(choice_qs, elicited, transform, seed=RESAMPLE_SEED,
                   tendency_weights=None):
    """추출 분포에 transform(qid, probs, rng, tendency)을 적용해 1건씩 샘플링한다.

    transform은 단일/리커트 문항에는 simplex 분포를, 다중선택 문항에는
    보기별 독립 선택확률 딕셔너리를 받아 같은 형태를 반환한다.
    tendency는 페르소나당 1회만 추첨해 전 문항에 동일하게 적용한다 (A-2).
    """
    rng = random.Random(seed)
    rows = []
    for probs in elicited:
        tendency = assign_tendency(rng, tendency_weights)
        row = {}
        for q in choice_qs:
            qid = q["id"]
            p = transform(qid, probs[qid], rng, tendency) if transform else probs[qid]
            if q["type"] == "multi":
                opts = [str(o) for o in q["options"]]
                picks = [o for o in opts if rng.random() < min(1.0, max(0.0, p.get(o, 0)))]
                if not picks:
                    picks = [max(p, key=lambda o: p[o])]
                row[qid] = MULTI_SEP.join(sorted(picks))
            else:
                opts = (list(LIKERT_OPTIONS) if q["type"] == "likert"
                        else [str(o) for o in q["options"]])
                row[qid] = sample_from_probs(p, rng, opts)
        rows.append(row)
    return rows


def evaluate(name, choice_qs, real_dists, rows):
    synth_dists = {q["id"]: dist_from_values([r[q["id"]] for r in rows], q)
                   for q in choice_qs}
    m = compute_metrics(real_dists, synth_dists, choice_qs)
    return {"name": name, "summary": m["summary"], "per_question": m["per_question"]}


# -- 변환 함수들 -------------------------------------------------------------
def _multi_flatten(alpha):
    def f(qid, p, rng):
        return {o: (1 - alpha) * v + alpha * 0.5 for o, v in p.items()}
    return f


def _multi_temp(t):
    def f(qid, p, rng):
        out = {}
        for o, v in p.items():
            out[o] = temperature_scale({"yes": v, "no": 1 - v}, t)["yes"]
        return out
    return f


def _multi_mixture(weights, strength):
    def f(qid, p, rng, tendency):
        if tendency == "majority":
            top = max(p, key=lambda o: p[o])
            return {o: min(0.95, v * (1 + strength)) if o == top else v
                    for o, v in p.items()}
        if tendency == "minority":
            top = max(p, key=lambda o: p[o])
            return {o: v if o == top else min(0.95, v * (1 + strength))
                    for o, v in p.items()}
        # extreme: 선택률이 가장 낮은 보기 2개를 부스트
        low = sorted(p, key=lambda o: p[o])[:2]
        return {o: min(0.95, v * (1 + strength)) if o in low else v
                for o, v in p.items()}
    return f


def _multi_calibrated_alpha(p, real):
    """다중선택용 보정 계수: 보기별 Bernoulli JSD 평균이 최소가 되는 alpha."""
    from core.variance import jsd as _jsd
    best_alpha, best_d = 0.0, float("inf")
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        flat = {o: (1 - alpha) * v + alpha * 0.5 for o, v in p.items()}
        d = sum(_jsd({"yes": flat[o], "no": 1 - flat[o]},
                     {"yes": real[o], "no": 1 - real[o]}) for o in real) / len(real)
        if d < best_d:
            best_alpha, best_d = alpha, d
    return best_alpha


def make_transform(choice_qs, kind, param, real_dists):
    """문항 유형에 따라 simplex/다중선택 변환을 나눠 적용한다."""
    multi_fns = {
        "flatten": _multi_flatten,
        "temp": _multi_temp,
    }

    def transform(qid, p, rng, tendency):
        q = next(q for q in choice_qs if q["id"] == qid)
        if q["type"] == "multi":
            if kind in multi_fns:
                return multi_fns[kind](param)(qid, p, rng)
            if kind == "mixture":
                weights, strength = param
                return _multi_mixture(weights, strength)(qid, p, rng, tendency)
            if kind == "calibrated":
                alpha = _multi_calibrated_alpha(p, real_dists[qid])
                return _multi_flatten(alpha)(qid, p, rng)
            if kind == "mixture_calibrated":
                alpha = _multi_calibrated_alpha(p, real_dists[qid])
                p2 = _multi_flatten(alpha)(qid, p, rng)
                weights, strength = param
                return _multi_mixture(weights, strength)(qid, p2, rng, tendency)
            return p
        # 단일/리커트 (simplex)
        opts = (list(LIKERT_OPTIONS) if q["type"] == "likert"
                else [str(o) for o in q["options"]])
        if kind == "flatten":
            return flatten_distribution(p, param, opts)
        if kind == "temp":
            return temperature_scale(p, param, opts)
        if kind == "mixture":
            weights, strength = param
            return mixture_adjust(p, tendency, strength, opts)
        if kind == "calibrated":
            alpha, _ = calibrate_flattening(p, real_dists[qid], options=opts)
            return flatten_distribution(p, alpha, opts)
        if kind == "mixture_calibrated":
            alpha, _ = calibrate_flattening(p, real_dists[qid], options=opts)
            p2 = flatten_distribution(p, alpha, opts)
            weights, strength = param
            return mixture_adjust(p2, tendency, strength, opts)
        return p

    return transform


def main():
    choice_qs, real_dists, elicited = load_inputs()
    print(f"추출 분포 {len(elicited)}건 로드")
    results = []

    def run(name, kind=None, param=None):
        tr = make_transform(choice_qs, kind, param, real_dists) if kind else None
        tw = param[0] if kind in ("mixture", "mixture_calibrated") and param else None
        rows = sample_answers(choice_qs, elicited, tr, tendency_weights=tw)
        res = evaluate(name, choice_qs, real_dists, rows)
        results.append(res)
        s = res["summary"]
        print(f"{name:38s} mean_err={s['mean_error']:.3f} sd_ratio={s['sd_ratio']:.3f} "
              f"극단동질화={s['extreme_homog_rate_synth']:.2f} JSD={s['jsd']:.4f} "
              f"minority={s['minority_recall']:.2f}")

    run("baseline_replay (추출분포 그대로 재샘플링)")
    for alpha in (0.1, 0.2, 0.3, 0.5):
        run(f"A-3 flatten alpha={alpha}", "flatten", alpha)
    for t in (1.5, 2.0, 3.0):
        run(f"A-3 temp-scale T={t}", "temp", t)
    run("A-3 calibrated alpha (학습분포 대조)", "calibrated", None)
    run("A-2 mixture 0.6/0.3/0.1 strength=1.0", "mixture",
        (DEFAULT_TENDENCY_WEIGHTS, 1.0))
    run("A-2 mixture 0.6/0.3/0.1 strength=0.5", "mixture",
        (DEFAULT_TENDENCY_WEIGHTS, 0.5))
    run("A-2+A-3 calibrated+mixture", "mixture_calibrated",
        (DEFAULT_TENDENCY_WEIGHTS, 1.0))

    (BASE / "variance_experiments.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장: benchmarks/variance_experiments.json")


if __name__ == "__main__":
    main()
