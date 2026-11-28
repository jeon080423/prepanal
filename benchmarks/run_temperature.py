"""Phase B A-1 온도 실험 — 라이브 실행 (benchmarks/run_temperature.py)

로컬 Gemini CLI는 temperature를 0.7로 하드코딩하고 있어 온도 실험이 안 된다.
그래서 Gemini CLI와 동일한 승인 인증 경로( vault 서로게이트, dynamic_credentials
헬퍼, 허용 호스트 generativelanguage.googleapis.com만)로 Gemini REST를 직접
호출하되, generationConfig.temperature만 바꿔 호출한다. 원본 키는 이 프로세스에
닿지 않는다 (서로게이트를 URL 쿼리 파라미터로만 전달, Sentinel이 교체).

GenerationService에 generate_fn으로 주입하므로, 페르소나·시드·배치·전략은
베이스라인과 완전히 동일하고 LLM temperature만 달라진다.

실행: cd ~/workspace/prepanel && python3 benchmarks/run_temperature.py 1.3
산출물: benchmarks/synthetic_temp_1.3.csv, elicited_temp_1.3.json,
        temp_1.3_metrics.json, llm_calls_temp_1.3.jsonl
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynamic_credentials import url_with_surrogate_query_param  # noqa: E402

import pandas as pd  # noqa: E402

from config import DEFAULT_PERSONA  # noqa: E402
from core import strategies as strat  # noqa: E402
from core.question_types import get_question_type  # noqa: E402
from core.services import GenerationService  # noqa: E402
from metrics import compute_metrics, dist_from_values  # noqa: E402

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
N = 200
BATCH_SIZE = 10
SEED = 20261123
MODEL = "gemini-2.5-flash-lite"
ALLOWED_HOSTS = ["generativelanguage.googleapis.com"]
API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def call_gemini_rest(prompt: str, model: str, temperature: float,
                     timeout: int = 180) -> str:
    """Gemini REST 직접 호출 (CLI와 동일한 서로게이트 인증 경로)."""
    url = f"{API_BASE}/models/{model}:generateContent"
    url = url_with_surrogate_query_param(
        url, "custom.gemini", allowed_hosts=ALLOWED_HOSTS)
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini 응답 파싱 실패: {json.dumps(data)[:300]}")


def main() -> None:
    if len(sys.argv) < 2:
        print("사용법: python3 benchmarks/run_temperature.py <temperature>")
        sys.exit(1)
    temp = float(sys.argv[1])
    tag = str(temp).replace(".", "_")
    questions = json.loads((ROOT / "samples" / "sample_questionnaire.json").read_text())
    choice_qs = [q for q in questions if q["type"] != "open"]
    real_dists = json.loads((BASE / "real_dists.json").read_text())

    calls_path = BASE / f"llm_calls_temp_{tag}.jsonl"
    calls_f = calls_path.open("w", encoding="utf-8")

    def rest_generate(provider_id: str, prompt: str, model: str) -> str:
        # 무료 등급 RPM 제한 회피용 호출 간격 (베이스라인 CLI 경로는 호출당
        # ~13초가 자연히 걸렸으나, REST 직접 호출은 더 빨라 연타 시 429가 남)
        time.sleep(8)
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                out = call_gemini_rest(prompt, model, temp)
                calls_f.write(json.dumps(
                    {"temperature": temp, "prompt": prompt, "response": out},
                    ensure_ascii=False) + "\n")
                calls_f.flush()
                return out
            except Exception as e:  # noqa: BLE001 — 429/503/타임아웃 재시도용
                last_err = e
                msg = str(e).lower()
                if ("429" in msg or "503" in msg or "timed out" in msg
                        or "timeout" in msg):
                    wait = 30 * (attempt + 1)
                    print(f"  {type(e).__name__} 발생 — {wait}초 대기 후 재시도 "
                          f"({attempt + 1}/6)")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"LLM 호출 6회 실패: {last_err}")

    svc = GenerationService()
    t0 = time.time()
    df = svc.generate_responses(
        choice_qs, DEFAULT_PERSONA, N,
        provider_id="gemini", model=MODEL,
        batch_size=BATCH_SIZE, seed=SEED,
        generate_fn=rest_generate,
    )
    calls_f.close()
    elapsed = time.time() - t0
    print(f"생성 완료: {len(df)}건, {elapsed / 60:.1f}분 소요 (temperature={temp})")
    df.to_csv(BASE / f"synthetic_temp_{tag}.csv", index=False, encoding="utf-8-sig")

    elicited: list[dict] = []
    with calls_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "[문항]" not in rec["prompt"]:
                continue
            parsed = strat.parse_json_array(rec["response"])
            # 전략의 _check_batch와 동일하게 배치 크기만큼만 취한다.
            parsed = parsed[:BATCH_SIZE]
            for prow in parsed:
                entry = {}
                for q in choice_qs:
                    qtype = get_question_type(q["type"])
                    raw = prow.get(q["id"], {}) if isinstance(prow, dict) else {}
                    _probs, _unmatched = qtype.probs_from_elicited(
                        q, raw if isinstance(raw, dict) else {})
                    entry[q["id"]] = _probs
                elicited.append(entry)
    print(f"추출 분포 복원: {len(elicited)}건")
    (BASE / f"elicited_temp_{tag}.json").write_text(
        json.dumps({"n": len(elicited), "seed": SEED, "model": MODEL,
                    "batch_size": BATCH_SIZE, "temperature": temp,
                    "rows": elicited}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    synth_rows = [{q["id"]: str(row[q["id"]]) for q in choice_qs}
                  for _, row in df.iterrows()]
    synth_dists = {q["id"]: dist_from_values([r[q["id"]] for r in synth_rows], q)
                   for q in choice_qs}
    metrics = compute_metrics(real_dists, synth_dists, choice_qs)
    metrics["meta"] = {"n_real": 200, "n_synth": len(df), "seed": SEED,
                       "model": MODEL, "batch_size": BATCH_SIZE,
                       "temperature": temp,
                       "elapsed_minutes": round(elapsed / 60, 1),
                       "strategy": "elicitation (분포추출)"}
    (BASE / f"temp_{tag}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
