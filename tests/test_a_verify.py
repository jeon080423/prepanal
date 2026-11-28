"""Bundle-3 워커 A 검증 스크립트 (mock만 사용, 실제 API 호출 없음).

실행: ~/workspace/.venv-ahptest/bin/python test_a_verify.py
검증 항목:
  T1  파싱 재시도 — 첫 호출 깨진 JSON → 두 번째 정상 JSON이면 성공
      (DistributionElicitation / Precision / OpenEnded 전부)
  T1b 3회 전부 실패 시 오류 발생
  T2  (b) 앵커 일관성 — 2차 호출 프롬프트에 앵커 텍스트 포함,
      조건부 확률이 무조건부 확률과 다름, 앵커 답변과 모순 없는 답변
  T2b 캐시 키에 앵커 답변 포함 — 앵커가 다르면 캐시 미스
  T3  학습된 분포 문항은 LLM 호출 없이 조건부 prior로 처리
  T4  GenerationService 경유 end-to-end (strategy_order에 conditional 포함)
  T5  앵커가 없는 설문은 일반 분포추출과 동일 동작
"""
import json
import random
import sys

sys.path.insert(0, ".")

from core.strategies import (
    ConditionalElicitationStrategy,
    DistributionElicitationStrategy,
    GenerationContext,
    OpenEndedGenerationStrategy,
    PrecisionGenerationStrategy,
    _build_dist_prompt,
    _cache_key,
    _is_anchor_question,
    get_strategy,
)
from core.services import GenerationService

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


PERSONAS = [
    {"성별": "남성", "연령대": "20대"},
    {"성별": "여성", "연령대": "30대"},
]

Q_ANCHOR_PARTY = {"id": "Q1", "text": "지지하는 정당이 있습니까?",
                  "type": "single",
                  "options": ["더불어민주당", "국민의힘", "조국혁신당",
                              "개혁신당", "지지하는 정당 없음"]}
Q_ANCHOR_IDEO = {"id": "Q2", "text": "자신의 이념 성향은 어디에 가깝습니까?",
                 "type": "single", "options": ["진보", "중도", "보수"]}
Q_EVAL = {"id": "Q3", "text": "대통령의 국정운영을 어떻게 평가하십니까?",
          "type": "single",
          "options": ["매우 잘하고 있다", "대체로 잘하고 있다",
                      "대체로 잘못하고 있다", "매우 잘못하고 있다"]}
Q_ECON = {"id": "Q4", "text": "앞으로 우리나라 경제는 어떻게 될 것 같습니까?",
          "type": "single",
          "options": ["좋아질 것이다", "비슷할 것이다", "나빠질 것이다"]}
Q_OPEN = {"id": "Q5", "text": "정부에 바라는 점을 자유롭게 적어주세요.",
          "type": "open", "options": []}

GOOD_DIST = json.dumps(
    [{"Q3": {"매우 잘하고 있다": 0.25, "대체로 잘하고 있다": 0.25,
             "대체로 잘못하고 있다": 0.25, "매우 잘못하고 있다": 0.25}}
     for _ in range(2)], ensure_ascii=False)
GOOD_OPEN = json.dumps([{"Q5": "물가 안정이 필요합니다."} for _ in range(2)],
                       ensure_ascii=False)


def ctx_with(fn, **kw):
    return GenerationContext(provider_id="dummy", model="dummy-1",
                             generate_fn=fn, **kw)


# ---------------------------------------------------------------- T1
def test_t1_elicitation_retry():
    calls = []

    def fn(pid, prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            return "깨진 JSON {{{ 응답 없음"
        return GOOD_DIST

    out = DistributionElicitationStrategy().generate(
        ctx_with(fn), [Q_EVAL], PERSONAS, random.Random(7))
    check("T1a elicitation: 1회 파싱 실패 후 재시도 성공",
          len(out) == 2 and all(r["Q3"] in Q_EVAL["options"] for r in out)
          and len(calls) == 2, f"calls={len(calls)}")


def test_t1_precision_retry():
    calls = []

    def fn(pid, prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            return "[{Q3: broken"
        return GOOD_DIST

    ctx = ctx_with(fn, use_precision=True,
                   transcripts=[{"name": "t1", "text": "인터뷰 내용 " * 50}])
    out = PrecisionGenerationStrategy().generate(
        ctx, [Q_EVAL], PERSONAS, random.Random(7))
    check("T1b precision: 1회 파싱 실패 후 재시도 성공",
          len(out) == 2 and len(calls) == 2, f"calls={len(calls)}")


def test_t1_open_retry():
    calls = []

    def fn(pid, prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            return "not json at all"
        return GOOD_OPEN

    out = OpenEndedGenerationStrategy().generate(
        ctx_with(fn), [Q_OPEN], PERSONAS, random.Random(7))
    check("T1c open_ended: 1회 파싱 실패 후 재시도 성공",
          len(out) == 2 and out[0]["Q5"] == "물가 안정이 필요합니다."
          and len(calls) == 2, f"calls={len(calls)}")


def test_t1b_all_fail_raises():
    def fn(pid, prompt, model):
        return "깨진 JSON {{{"

    try:
        DistributionElicitationStrategy().generate(
            ctx_with(fn), [Q_EVAL], PERSONAS, random.Random(7))
        check("T1d 3회 전부 파싱 실패 시 오류 발생", False, "예외 없음")
    except ValueError as e:
        check("T1d 3회 전부 파싱 실패 시 오류 발생", True, f"ValueError: {e}")


# ---------------------------------------------------------------- T2
def test_t2_anchor_consistency():
    prompts = []

    def fn(pid, prompt, model):
        prompts.append(prompt)
        if "[이 응답자의 앵커 답변:" in prompt:
            # 조건부 분포: 더불어민주당·진보 앵커와 일관된 평가에 확률 집중
            return json.dumps(
                [{"Q3": {"매우 잘하고 있다": 1.0, "대체로 잘하고 있다": 0.0,
                         "대체로 잘못하고 있다": 0.0, "매우 잘못하고 있다": 0.0},
                  "Q4": {"좋아질 것이다": 1.0, "비슷할 것이다": 0.0,
                         "나빠질 것이다": 0.0}} for _ in range(2)],
                ensure_ascii=False)
        if "지지하는 정당" in prompt:
            # 1단계 앵커 추출: 더불어민주당 1.0 / 진보 1.0 (결정적)
            return json.dumps(
                [{"Q1": {"더불어민주당": 1.0, "국민의힘": 0.0,
                         "조국혁신당": 0.0, "개혁신당": 0.0,
                         "지지하는 정당 없음": 0.0},
                  "Q2": {"진보": 1.0, "중도": 0.0, "보수": 0.0}}
                 for _ in range(2)], ensure_ascii=False)
        # 앵커 텍스트가 없는 무조건부 프롬프트 → 반대 방향 분포
        return json.dumps(
            [{"Q3": {"매우 잘하고 있다": 0.0, "대체로 잘하고 있다": 0.0,
                     "대체로 잘못하고 있다": 0.0, "매우 잘못하고 있다": 1.0},
              "Q4": {"좋아질 것이다": 0.0, "비슷할 것이다": 0.0,
                     "나빠질 것이다": 1.0}} for _ in range(2)],
            ensure_ascii=False)

    strat = ConditionalElicitationStrategy()
    out = strat.generate(ctx_with(fn),
                         [Q_ANCHOR_PARTY, Q_ANCHOR_IDEO, Q_EVAL, Q_ECON],
                         PERSONAS, random.Random(7))
    phase2 = [p for p in prompts if "[이 응답자의 앵커 답변:" in p]
    check("T2a 앵커 판별 (id/내용)",
          _is_anchor_question(Q_ANCHOR_PARTY)
          and _is_anchor_question(Q_ANCHOR_IDEO)
          and not _is_anchor_question(Q_EVAL)
          and not _is_anchor_question(Q_ECON))
    check("T2b 2차 호출 프롬프트에 앵커 텍스트 포함",
          len(phase2) == 1 and "더불어민주당" in phase2[0]
          and "진보" in phase2[0]
          and "Q1(지지하는 정당이 있습니까?)=더불어민주당" in phase2[0],
          f"phase2 prompts={len(phase2)}")
    check("T2c 앵커 답변과 후속 답변이 모순되지 않음",
          all(r["Q1"] == "더불어민주당" and r["Q2"] == "진보"
              and r["Q3"] == "매우 잘하고 있다"
              and r["Q4"] == "좋아질 것이다" for r in out),
          f"out={out}")

    # 조건부 확률이 무조건부 확률과 달라지는지: 같은 mock에 무조건부
    # 프롬프트(앵커 텍스트 없음)를 주면 다른 분포를 반환하는지 직접 확인
    uncond = json.loads(fn("d", _build_dist_prompt([Q_EVAL], ["프로필"]), "m"))
    cond = json.loads(fn("d", _build_dist_prompt(
        [Q_EVAL], ["프로필\n[이 응답자의 앵커 답변: Q1=더불어민주당]"]), "m"))
    check("T2d 조건부 확률이 무조건부 확률과 다름",
          uncond != cond, f"uncond={uncond} cond={cond}")


def test_t2b_cache_key_includes_anchor():
    calls = []

    def fn(pid, prompt, model):
        calls.append(prompt)
        if "[이 응답자의 앵커 답변:" in prompt:
            return json.dumps(
                [{"Q3": {"매우 잘하고 있다": 0.5, "대체로 잘하고 있다": 0.5,
                         "대체로 잘못하고 있다": 0.0, "매우 잘못하고 있다": 0.0}}
                 for _ in range(2)], ensure_ascii=False)
        return json.dumps(
            [{"Q1": {"더불어민주당": 1.0, "국민의힘": 0.0,
                     "조국혁신당": 0.0, "개혁신당": 0.0,
                     "지지하는 정당 없음": 0.0},
              "Q2": {"진보": 1.0, "중도": 0.0, "보수": 0.0}}
             for _ in range(2)], ensure_ascii=False)

    strat = ConditionalElicitationStrategy()
    ctx = ctx_with(fn)
    strat.generate(ctx, [Q_ANCHOR_PARTY, Q_ANCHOR_IDEO, Q_EVAL],
                   PERSONAS, random.Random(7))
    n1 = len(calls)
    # 같은 ctx·같은 페르소나로 다시 실행 → 캐시 적중으로 추가 호출 0건
    strat.generate(ctx, [Q_ANCHOR_PARTY, Q_ANCHOR_IDEO, Q_EVAL],
                   PERSONAS, random.Random(7))
    check("T2e 동일 앵커면 2차 실행은 캐시 적중 (추가 호출 0건)",
          len(calls) == n1, f"calls {n1} -> {len(calls)}")
    # 앵커 답변만 다른 프롬프트는 캐시 키가 다름 (키 자체를 직접 대조)
    k1 = _cache_key("dummy", "dummy-1", "prompt 앵커=더불어민주당")
    k2 = _cache_key("dummy", "dummy-1", "prompt 앵커=국민의힘")
    check("T2f 캐시 키가 앵커 답변에 따라 달라짐", k1 != k2)


def test_t2_fallback_prompt_expansion():
    """2차 호출이 3회 전부 실패하면 (a) 확장 프롬프트로 폴백."""
    prompts = []
    state = {"phase2_calls": 0}

    def fn(pid, prompt, model):
        prompts.append(prompt)
        if "[앵커 조건]" in prompt:
            return json.dumps(
                [{"Q3": {"매우 잘하고 있다": 1.0, "대체로 잘하고 있다": 0.0,
                         "대체로 잘못하고 있다": 0.0, "매우 잘못하고 있다": 0.0}}
                 for _ in range(2)], ensure_ascii=False)
        if "[이 응답자의 앵커 답변:" in prompt:
            state["phase2_calls"] += 1
            return "깨진 JSON"
        return json.dumps(
            [{"Q1": {"더불어민주당": 1.0, "국민의힘": 0.0,
                     "조국혁신당": 0.0, "개혁신당": 0.0,
                     "지지하는 정당 없음": 0.0}} for _ in range(2)],
            ensure_ascii=False)

    out = ConditionalElicitationStrategy().generate(
        ctx_with(fn), [Q_ANCHOR_PARTY, Q_EVAL], PERSONAS, random.Random(7))
    check("T2g 2차 호출 3회 실패 시 (a) 확장 프롬프트 폴백으로 성공",
          len(out) == 2 and out[0]["Q3"] == "매우 잘하고 있다"
          and state["phase2_calls"] == 3
          and any("[앵커 조건]" in p for p in prompts),
          f"phase2_calls={state['phase2_calls']}")


# ---------------------------------------------------------------- T3
def test_t3_learned_prior_no_llm():
    prompts = []

    def fn(pid, prompt, model):
        prompts.append(prompt)
        if "[이 응답자의 앵커 답변:" in prompt:
            # 2차 호출 프롬프트에 학습 문항(Q4)이 들어있으면 안 됨
            assert "Q4" not in prompt, "학습 문항이 2차 호출에 포함됨"
            return json.dumps(
                [{"Q3": {"매우 잘하고 있다": 1.0, "대체로 잘하고 있다": 0.0,
                         "대체로 잘못하고 있다": 0.0, "매우 잘못하고 있다": 0.0}}
                 for _ in range(2)], ensure_ascii=False)
        return json.dumps(
            [{"Q1": {"더불어민주당": 1.0, "국민의힘": 0.0,
                     "조국혁신당": 0.0, "개혁신당": 0.0,
                     "지지하는 정당 없음": 0.0}} for _ in range(2)],
            ensure_ascii=False)

    learned = {"Q4": {"matched_col": "c", "type": "single",
                      "dist": {"좋아질 것이다": 1.0, "비슷할 것이다": 0.0,
                               "나빠질 것이다": 0.0}, "n": 50}}
    ctx = ctx_with(fn, learned=learned)
    out = ConditionalElicitationStrategy().generate(
        ctx, [Q_ANCHOR_PARTY, Q_EVAL, Q_ECON], PERSONAS, random.Random(7))
    check("T3a 학습 문항은 LLM 호출 없이 조건부 prior로 처리",
          len(prompts) == 2  # 1단계 앵커 + 2차(Q3만) = 2건
          and all(r["Q4"] == "좋아질 것이다" for r in out),
          f"calls={len(prompts)} out={out}")
    check("T3b 2차 호출 프롬프트에 학습 문항 미포함",
          all("경제는 어떻게" not in p for p in prompts))


# ---------------------------------------------------------------- T4/T5
def test_t4_service_end_to_end():
    def fn(pid, prompt, model):
        if "[이 응답자의 앵커 답변:" in prompt:
            return json.dumps(
                [{"Q3": {"매우 잘하고 있다": 1.0, "대체로 잘하고 있다": 0.0,
                         "대체로 잘못하고 있다": 0.0, "매우 잘못하고 있다": 0.0}}
                 for _ in range(4)], ensure_ascii=False)
        return json.dumps(
            [{"Q1": {"더불어민주당": 1.0, "국민의힘": 0.0,
                     "조국혁신당": 0.0, "개혁신당": 0.0,
                     "지지하는 정당 없음": 0.0},
              "Q2": {"진보": 1.0, "중도": 0.0, "보수": 0.0}}
             for _ in range(4)], ensure_ascii=False)

    learned = {"Q4": {"matched_col": "c", "type": "single",
                      "dist": {"좋아질 것이다": 1.0, "비슷할 것이다": 0.0,
                               "나빠질 것이다": 0.0}, "n": 50}}
    svc = GenerationService(strategy_order=[
        "conditional", "learned", "precision", "elicitation", "open_ended"])
    df = svc.generate_responses(
        [Q_ANCHOR_PARTY, Q_ANCHOR_IDEO, Q_EVAL, Q_ECON],
        {"성별": {"남성": 50.0, "여성": 50.0},
         "연령대": {"20대": 50.0, "30대": 50.0}},
        4, provider_id="dummy", model="dummy-1", learned=learned,
        batch_size=4, seed=7, generate_fn=fn)
    check("T4 GenerationService 경유 end-to-end",
          len(df) == 4 and (df["Q1"] == "더불어민주당").all()
          and (df["Q2"] == "진보").all()
          and (df["Q3"] == "매우 잘하고 있다").all()
          and (df["Q4"] == "좋아질 것이다").all(),
          f"cols={list(df.columns)}")


def test_t5_no_anchor_degenerates():
    def fn(pid, prompt, model):
        return json.dumps(
            [{"Q3": {"매우 잘하고 있다": 0.5, "대체로 잘하고 있다": 0.5,
                     "대체로 잘못하고 있다": 0.0, "매우 잘못하고 있다": 0.0},
              "Q4": {"좋아질 것이다": 1.0, "비슷할 것이다": 0.0,
                     "나빠질 것이다": 0.0}} for _ in range(2)],
            ensure_ascii=False)

    out = ConditionalElicitationStrategy().generate(
        ctx_with(fn), [Q_EVAL, Q_ECON], PERSONAS, random.Random(7))
    check("T5 앵커 없음 → 일반 분포추출과 동일 동작",
          len(out) == 2
          and all(r["Q3"] in Q_EVAL["options"] for r in out)
          and all(r["Q4"] == "좋아질 것이다" for r in out))


def test_registry():
    check("T6 레지스트리에 conditional 등록",
          get_strategy("conditional").name == "conditional"
          and get_strategy("conditional").label.startswith("조건부"))


if __name__ == "__main__":
    test_t1_elicitation_retry()
    test_t1_precision_retry()
    test_t1_open_retry()
    test_t1b_all_fail_raises()
    test_t2_anchor_consistency()
    test_t2b_cache_key_includes_anchor()
    test_t2_fallback_prompt_expansion()
    test_t3_learned_prior_no_llm()
    test_t4_service_end_to_end()
    test_t5_no_anchor_degenerates()
    test_registry()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
