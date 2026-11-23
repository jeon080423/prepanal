"""Phase A 베이스라인 측정 (benchmarks/measure_baseline.py)

임시 벤치마크(샘플 설문 + 실측 200건)에 대해 현재 모델(분포추출 전략,
Gemini gemini-2.5-flash-lite, temperature 기본값 0.7)로 합성 응답 200건을
생성하고 지표를 잰다.

실행: cd ~/workspace/prepanel && python3 benchmarks/measure_baseline.py
산출물 (benchmarks/):
- synthetic_baseline.csv — 합성 응답 200건
- elicited_baseline.json — 배치별 LLM 추출 확률분포 (Phase B 오프라인 실험용)
- real_dists.json — 실측 문항별 분포
- baseline_metrics.json — 지표 결과
- llm_calls_baseline.jsonl — LLM 호출 프롬프트·응답 기록
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from config import DEFAULT_PERSONA  # noqa: E402
from core import providers as providers_mod  # noqa: E402
from core import strategies as strat  # noqa: E402
from core.question_types import get_question_type  # noqa: E402
from core.services import GenerationService  # noqa: E402
from metrics import compute_metrics, dist_from_values  # noqa: E402

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
N = 200
BATCH_SIZE = 10
SEED = 20261123
PROVIDER_ID = "gemini"
MODEL = "gemini-2.5-flash-lite"


def load_real() -> tuple[list[dict], pd.DataFrame]:
    questions = json.loads((ROOT / "samples" / "sample_questionnaire.json").read_text())
    df = pd.read_excel(ROOT / "samples" / "sample_real_data.xlsx")
    # xlsx 컬럼 순서는 설문 문항 순서와 동일 (헤더 텍스트 대조로 확인됨)
    assert len(df.columns) == len(questions), "실측 컬럼 수와 문항 수가 다릅니다"
    rows = []
    for _, r in df.iterrows():
        rows.append({q["id"]: str(r.iloc[i]).strip()
                     for i, q in enumerate(questions)})
    return questions, rows


def main() -> None:
    questions, real_rows = load_real()
    choice_qs = [q for q in questions if q["type"] != "open"]
    print(f"실측 {len(real_rows)}건, 선택형 문항 {[q['id'] for q in choice_qs]}")

    real_dists = {q["id"]: dist_from_values([r[q["id"]] for r in real_rows], q)
                  for q in choice_qs}
    (BASE / "real_dists.json").write_text(
        json.dumps(real_dists, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- LLM 호출 기록 + 429 재시도 래퍼 -------------------------------------
    calls_path = BASE / "llm_calls_baseline.jsonl"
    calls_f = calls_path.open("w", encoding="utf-8")

    def recording_generate(provider_id: str, prompt: str, model: str) -> str:
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                out = providers_mod.generate(provider_id, prompt, model)
                calls_f.write(json.dumps(
                    {"prompt": prompt, "response": out},
                    ensure_ascii=False) + "\n")
                calls_f.flush()
                return out
            except RuntimeError as e:
                last_err = e
                if "429" in str(e).lower():
                    wait = 30 * (attempt + 1)
                    print(f"  429 발생 — {wait}초 대기 후 재시도 ({attempt + 1}/6)")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"LLM 호출 6회 실패: {last_err}")

    svc = GenerationService()
    t0 = time.time()
    df = svc.generate_responses(
        choice_qs, DEFAULT_PERSONA, N,
        provider_id=PROVIDER_ID, model=MODEL,
        batch_size=BATCH_SIZE, seed=SEED,
        generate_fn=recording_generate,
    )
    calls_f.close()
    elapsed = time.time() - t0
    print(f"생성 완료: {len(df)}건, {elapsed / 60:.1f}분 소요")
    df.to_csv(BASE / "synthetic_baseline.csv", index=False, encoding="utf-8-sig")

    # -- 기록된 호출에서 추출 확률분포 복원 ----------------------------------
    elicited: list[dict] = []
    with calls_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "[문항]" not in rec["prompt"]:
                continue
            parsed = strat.parse_json_array(rec["response"])
            # 전략의 _check_batch와 동일하게 배치 크기만큼만 취한다
            # (LLM이 더 많이 반환하면 앞부분만 사용).
            parsed = parsed[:BATCH_SIZE]
            for prow in parsed:
                entry = {}
                for q in choice_qs:
                    qtype = get_question_type(q["type"])
                    raw = prow.get(q["id"], {}) if isinstance(prow, dict) else {}
                    entry[q["id"]] = qtype.probs_from_elicited(
                        q, raw if isinstance(raw, dict) else {})
                elicited.append(entry)
    print(f"추출 분포 복원: {len(elicited)}건 (기대 {N}건)")
    (BASE / "elicited_baseline.json").write_text(
        json.dumps({"n": len(elicited), "seed": SEED, "model": MODEL,
                    "batch_size": BATCH_SIZE, "rows": elicited},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # -- 지표 -----------------------------------------------------------------
    synth_rows = [{q["id"]: str(v) for q, v in
                   zip(choice_qs, [row[q["id"]] for q in choice_qs])}
                  for _, row in df.iterrows()]
    synth_dists = {q["id"]: dist_from_values([r[q["id"]] for r in synth_rows], q)
                   for q in choice_qs}
    metrics = compute_metrics(real_dists, synth_dists, choice_qs)
    metrics["meta"] = {"n_real": len(real_rows), "n_synth": len(df),
                       "seed": SEED, "model": MODEL, "provider": PROVIDER_ID,
                       "batch_size": BATCH_SIZE,
                       "elapsed_minutes": round(elapsed / 60, 1),
                       "strategy": "elicitation (분포추출, 기본값 temperature 0.7)"}
    (BASE / "baseline_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
