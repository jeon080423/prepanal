"""배치 중간 저장 / 재개 (체크포인트) 테스트 — 워커 D.

실제 LLM 호출 없음 — generate_fn 주입(mock)만 쓴다. 비밀값 없음.

검증 포인트:
  1. k개 배치 후 의도적 중단(예외) → 재개 시 처음부터 끝까지 한 번에
     돌린 결과와 행 단위로 동일 (manual / Nemotron / 고정밀+다중선택 /
     분기 문항 포함).
  2. 재개 시 "N배치 완료, 이어서 진행" 상태가 표시됨
     (format_checkpoint_status + last_resume_info + last_gen_meta).
  3. 세션 캐시가 완료 배치를 커버하면 재개 시 완료 배치에 LLM 재호출이
     없고 (cache_misses == 0), 캐시가 비어도 결과는 동일하고
     cache_misses > 0으로 기록됨.
  4. 지문 불일치·seed 없음 체크포인트는 ValueError.
  5. 이미 완료된 체크포인트는 LLM 호출 없이 저장된 행을 그대로 반환.
"""
import json
import re

import pandas as pd
import pytest

from core.services import (GenerationService, build_checkpoint_fingerprint,
                           checkpoint_matches, format_checkpoint_status)


def _profile_count(prompt: str) -> int:
    return len(re.findall(r"(?m)^\d+\.\s", prompt))


def fake_generate(provider_id, prompt, model):
    """test_core.py의 fake_generate와 동일한 동작 (독립 정의)."""
    n = _profile_count(prompt)
    assert n > 0
    if "[주관식 문항]" in prompt:
        return json.dumps([{"Q3": "배송이 빨라서 좋아요"} for _ in range(n)],
                          ensure_ascii=False)
    return json.dumps([{"Q2": {"예": 0.7, "아니오": 0.3}} for _ in range(n)],
                      ensure_ascii=False)


@pytest.fixture
def persona_dims():
    return {"성별": {"남성": 50.0, "여성": 50.0},
            "연령대": {"20대": 50.0, "30대": 50.0}}


@pytest.fixture
def mixed_questions():
    return [
        {"id": "Q1", "text": "학습된 문항", "type": "single",
         "options": ["A", "B"]},
        {"id": "Q2", "text": "미학습 문항", "type": "single",
         "options": ["예", "아니오"]},
        {"id": "Q3", "text": "주관식 문항", "type": "open", "options": []},
    ]


@pytest.fixture
def learned():
    return {"Q1": {"matched_col": "c", "type": "single",
                   "dist": {"A": 0.5, "B": 0.5}, "n": 50}}


class _FailAfter:
    """fail_at번째 호출에서 1회만 예외를 내고 이후에는 정상 동작한다
    (중단은 1회성 사건이고 재개 시 LLM은 다시 살아 있다는 가정)."""

    def __init__(self, fn, fail_at: int):
        self.fn = fn
        self.fail_at = fail_at
        self.calls: list[str] = []

    def __call__(self, provider_id, prompt, model):
        self.calls.append(prompt)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("simulated interruption (LLM 실패)")
        return self.fn(provider_id, prompt, model)


def _run_kwargs(questions, persona_dims, n, *, seed=42, batch_size=4,
                generate_fn=fake_generate, cache=None, **extra):
    kw = dict(provider_id="gemini", model="dummy", batch_size=batch_size,
              seed=seed, generate_fn=generate_fn)
    if cache is not None:
        kw["cache"] = cache
    kw.update(extra)
    return kw


# ----------------------------------------------------------------------------
# 1. 중단 → 재개 = 중단 없이 한 번에 돌린 결과와 행 단위 동일 (manual)
# ----------------------------------------------------------------------------
def test_resume_row_identical_manual(mixed_questions, persona_dims, learned):
    n, batch_size, seed = 12, 4, 42
    # (a) 중단 없이 한 번에
    cache_full: dict = {}
    df_full = GenerationService().generate_responses(
        mixed_questions, persona_dims, n,
        **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                      batch_size=batch_size, learned=learned,
                      cache=cache_full))
    calls_full = 6  # 배치당 분포추출 1 + 주관식 1 = 2회 × 3배치
    assert len(cache_full) == calls_full

    # (b) 1배치 후 의도적 중단 (배치당 2회 호출 → 3번째 호출에서 실패)
    cache: dict = {}
    failing = _FailAfter(fake_generate, fail_at=3)
    svc = GenerationService()
    snaps: list[dict] = []
    with pytest.raises(RuntimeError, match="simulated interruption"):
        svc.generate_responses(
            mixed_questions, persona_dims, n,
            **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                          batch_size=batch_size, learned=learned,
                          cache=cache, generate_fn=failing,
                          checkpoint_cb=snaps.append))
    assert len(failing.calls) == 3  # 2회 성공 + 1회 실패
    assert svc.last_checkpoint is not None
    assert svc.last_checkpoint["completed_batches"] == 1
    assert len(svc.last_checkpoint["rows"]) == 4
    assert len(snaps) == 1 and snaps[0]["completed_batches"] == 1
    assert format_checkpoint_status(svc.last_checkpoint).startswith(
        "1배치 완료, 이어서 진행")

    # (c) 같은 시드·같은 캐시·체크포인트로 재개
    svc2 = GenerationService()
    df_resumed = svc2.generate_responses(
        mixed_questions, persona_dims, n,
        **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                      batch_size=batch_size, learned=learned,
                      cache=cache, generate_fn=failing,
                      checkpoint=svc.last_checkpoint))
    pd.testing.assert_frame_equal(df_resumed, df_full)
    info = svc2.last_resume_info
    assert info["resumed_batches"] == 1
    assert info["cache_misses"] == 0
    assert info["cache_covered"] is True
    assert info["status"] == "resumed"
    # 재개가 끝나면 last_checkpoint는 최종(3배치) 스냅샷이라 상태 문구도
    # 최종 완료를 가리킨다. 재개 시작 시점은 gen_meta에 남는다.
    assert info["status_text"].startswith("3배치 완료, 이어서 진행")
    assert "1배치 완료 상태에서 이어서 진행" in svc2.last_gen_meta["이어서 진행"]
    # 재개 시 완료 배치에 LLM 재호출이 없어 전체 호출 수가 한 번에 돌린
    # 경우보다 실패한 1회만 많다 (중단 시 실패한 3번째 호출은 결과를
    # 남기지 않았으므로 재개 시 정상 호출로 1회 다시 나감).
    assert len(failing.calls) == calls_full + 1
    assert svc2.last_checkpoint["completed_batches"] == 3
    assert len(svc2.last_checkpoint["rows"]) == 12


# ----------------------------------------------------------------------------
# 2. 고정밀 + 다중선택 + 분기 문항 포함 시에도 행 단위 동일
# ----------------------------------------------------------------------------
def test_resume_row_identical_precision_multi_branch(persona_dims, learned):
    questions = [
        {"id": "Q1", "text": "학습된 문항", "type": "single",
         "options": ["A", "B"]},
        {"id": "Q2", "text": "미학습 문항", "type": "single",
         "options": ["예", "아니오"]},
        {"id": "Q3", "text": "주관식 문항", "type": "open", "options": []},
        {"id": "Q4", "text": "다중선택 문항", "type": "multi",
         "options": ["X", "Y", "Z"]},
        {"id": "Q5", "text": "동반 유형", "type": "single",
         "options": ["X", "Y"], "branching": "문1의 A 응답자만 응답"},
    ]
    transcripts = [{"name": "t1", "text": "인터뷰 내용 " * 100},
                   {"name": "t2", "text": "다른 인터뷰 내용 " * 100}]
    n, batch_size, seed = 10, 5, 7
    extra = dict(learned=learned, transcripts=transcripts,
                 use_precision=True)
    svc0 = GenerationService()
    df_full = svc0.generate_responses(
        questions, persona_dims, n,
        **_run_kwargs(questions, persona_dims, n, seed=seed,
                      batch_size=batch_size, cache={}, **extra))
    assert svc0.last_branch_meta["공란 처리"].get("Q5", 0) > 0

    cache: dict = {}
    # 배치당 LLM 호출: 분포추출(Q2·Q4·Q5, 고정밀 그룹) 1 + 주관식(Q3) 1
    # = 2회. 1배치 후 중단하려면 2회까지 허용하고 3번째에서 실패시킨다.
    failing = _FailAfter(fake_generate, fail_at=3)
    svc = GenerationService()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        svc.generate_responses(
            questions, persona_dims, n,
            **_run_kwargs(questions, persona_dims, n, seed=seed,
                          batch_size=batch_size, cache=cache,
                          generate_fn=failing, **extra))
    assert svc.last_checkpoint["completed_batches"] == 1

    svc2 = GenerationService()
    df_resumed = svc2.generate_responses(
        questions, persona_dims, n,
        **_run_kwargs(questions, persona_dims, n, seed=seed,
                      batch_size=batch_size, cache=cache,
                      generate_fn=failing,
                      checkpoint=svc.last_checkpoint, **extra))
    pd.testing.assert_frame_equal(df_resumed, df_full)
    assert svc2.last_branch_meta == svc0.last_branch_meta
    assert svc2.last_resume_info["cache_misses"] == 0


# ----------------------------------------------------------------------------
# 3. Nemotron 모드 (발췌본은 합성 레코드로 monkeypatch — data/ 없음)
# ----------------------------------------------------------------------------
def _synthetic_nemotron_records():
    recs = []
    pid = 0
    for sex in ("남성", "여성"):
        for age_band in ("20대", "30대", "40대"):
            for region, province in (("수도권", "서울"), ("영남권", "부산"),
                                     ("호남권", "광주")):
                for _ in range(2):
                    pid += 1
                    recs.append({
                        "pid": f"P{pid:04d}", "sex": sex, "age": 25,
                        "age_band": age_band, "province": province,
                        "district": f"{province} 어딘가", "region": region,
                        "education_level": "대학교 졸업",
                        "occupation": "직장인", "marital_status": "미혼",
                        "persona": "평범한 직장인",
                        "professional_persona": "사무직 직장인",
                    })
    return recs


def test_resume_row_identical_nemotron(mixed_questions, monkeypatch):
    monkeypatch.setattr("core.services.load_nemotron_extract",
                        lambda: _synthetic_nemotron_records())
    persona_dims = {"성별": {"남성": 50.0, "여성": 50.0},
                    "연령대": {"20대": 50.0, "30대": 50.0}}
    n, batch_size, seed = 10, 5, 42
    extra = dict(persona_source="nemotron", nemotron_quotas=None)
    svc0 = GenerationService()
    df_full = svc0.generate_responses(
        mixed_questions, persona_dims, n,
        **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                      batch_size=batch_size, cache={}, **extra))

    cache: dict = {}
    failing = _FailAfter(fake_generate, fail_at=3)
    svc = GenerationService()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        svc.generate_responses(
            mixed_questions, persona_dims, n,
            **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                          batch_size=batch_size, cache=cache,
                          generate_fn=failing, **extra))
    assert svc.last_checkpoint["completed_batches"] == 1

    svc2 = GenerationService()
    df_resumed = svc2.generate_responses(
        mixed_questions, persona_dims, n,
        **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                      batch_size=batch_size, cache=cache,
                      generate_fn=failing,
                      checkpoint=svc.last_checkpoint, **extra))
    pd.testing.assert_frame_equal(df_resumed, df_full)
    assert svc2.last_nemotron_meta["n_sampled"] == 10
    assert svc2.last_gen_meta["쿼터 달성"] == svc0.last_gen_meta["쿼터 달성"]
    assert svc2.last_resume_info["cache_misses"] == 0


# ----------------------------------------------------------------------------
# 4. 캐시가 비어도 결과는 동일하고 cache_misses에 기록됨
# ----------------------------------------------------------------------------
def test_resume_cache_miss_still_identical(mixed_questions, persona_dims,
                                           learned):
    n, batch_size, seed = 12, 4, 42
    df_full = GenerationService().generate_responses(
        mixed_questions, persona_dims, n,
        **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                      batch_size=batch_size, learned=learned, cache={}))

    cache: dict = {}
    failing = _FailAfter(fake_generate, fail_at=3)
    svc = GenerationService()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        svc.generate_responses(
            mixed_questions, persona_dims, n,
            **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                          batch_size=batch_size, learned=learned,
                          cache=cache, generate_fn=failing))
    ckpt = svc.last_checkpoint
    assert ckpt["completed_batches"] == 1

    # 캐시를 비우고 재개 — 완료 배치 재생성에 LLM 재호출이 발생하지만
    # 재생성 행은 버리고 체크포인트 행을 쓰므로 최종 결과는 동일하다.
    svc2 = GenerationService()
    df_resumed = svc2.generate_responses(
        mixed_questions, persona_dims, n,
        **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                      batch_size=batch_size, learned=learned,
                      cache={}, generate_fn=failing, checkpoint=ckpt))
    pd.testing.assert_frame_equal(df_resumed, df_full)
    info = svc2.last_resume_info
    assert info["cache_misses"] == 2  # 완료 배치 1개분의 프롬프트 2건
    assert info["cache_covered"] is False
    assert "캐시 미커버 2건" in svc2.last_gen_meta["이어서 진행"]


# ----------------------------------------------------------------------------
# 5. 상태 표시 · 지문 검증 · seed 없음 · 이미 완료된 체크포인트
# ----------------------------------------------------------------------------
def test_checkpoint_status_and_matching(mixed_questions, persona_dims,
                                        learned):
    n, batch_size, seed = 12, 4, 42
    fp = build_checkpoint_fingerprint(
        mixed_questions, n, batch_size=batch_size, seed=seed,
        provider_id="gemini", model="dummy", persona_source="manual",
        learned=learned)
    ckpt = {"version": 1, "fingerprint": fp, "completed_batches": 2,
            "total_batches": 3, "rows": [{}] * 8,
            "branch_blanked": {}, "cache_misses": 0, "status": "partial"}
    assert checkpoint_matches(ckpt, fp) is True
    assert format_checkpoint_status(ckpt) == \
        "2배치 완료, 이어서 진행 (전체 3배치 중 2배치 · 8건 저장됨)"

    fp_other_seed = build_checkpoint_fingerprint(
        mixed_questions, n, batch_size=batch_size, seed=43,
        provider_id="gemini", model="dummy", persona_source="manual",
        learned=learned)
    assert checkpoint_matches(ckpt, fp_other_seed) is False
    fp_other_n = build_checkpoint_fingerprint(
        mixed_questions, 8, batch_size=batch_size, seed=seed,
        provider_id="gemini", model="dummy", persona_source="manual",
        learned=learned)
    assert checkpoint_matches(ckpt, fp_other_n) is False

    svc = GenerationService()
    with pytest.raises(ValueError, match="이어서 진행할 수 없습니다"):
        svc.generate_responses(
            mixed_questions, persona_dims, n,
            **_run_kwargs(mixed_questions, persona_dims, n, seed=43,
                          batch_size=batch_size, learned=learned,
                          cache={}, checkpoint=ckpt))
    with pytest.raises(ValueError, match="seed가 필요"):
        svc.generate_responses(
            mixed_questions, persona_dims, n,
            **_run_kwargs(mixed_questions, persona_dims, n, seed=None,
                          batch_size=batch_size, learned=learned,
                          cache={}, checkpoint=ckpt))


def test_checkpoint_fingerprint_covers_all_inputs(mixed_questions,
                                                  persona_dims, learned):
    """C-1 회귀: 지문은 persona_dims·학습 분포 값·트랜스크립트 본문·
    Nemotron facets 변경을 모두 잡아야 한다. 하나라도 바뀌면
    checkpoint_matches()는 False, generate_responses(checkpoint=...)는
    ValueError를 낸다 (조용한 데이터 오염 방지)."""
    n, batch_size, seed = 12, 4, 42
    transcripts = [{"name": "T1", "text": "원본 인터뷰 내용"}]
    facets = ["age", "gender"]
    fp = build_checkpoint_fingerprint(
        mixed_questions, n, batch_size=batch_size, seed=seed,
        provider_id="gemini", model="dummy", persona_source="manual",
        persona_dims=persona_dims, learned=learned,
        transcripts=transcripts, nemotron_facets=facets)
    ckpt = {"version": 1, "fingerprint": fp, "completed_batches": 2,
            "total_batches": 3, "rows": [{}] * 8,
            "branch_blanked": {}, "cache_misses": 0, "status": "partial"}
    assert checkpoint_matches(ckpt, fp) is True

    def _fp(**over):
        kw = dict(questions=mixed_questions, n=n, batch_size=batch_size,
                  seed=seed, provider_id="gemini", model="dummy",
                  persona_source="manual", persona_dims=persona_dims,
                  learned=learned, transcripts=transcripts,
                  nemotron_facets=facets)
        kw.update(over)
        return build_checkpoint_fingerprint(**kw)

    # 1) persona_dims 변경 (분포 값만 바꿔도 달라야 함)
    dims_changed = {"성별": {"남성": 70.0, "여성": 30.0},
                    "연령대": {"20대": 50.0, "30대": 50.0}}
    assert checkpoint_matches(ckpt, _fp(persona_dims=dims_changed)) is False
    # 2) 학습 분포 값 변경 (문항 id는 그대로)
    learned_changed = {"Q1": {"matched_col": "c", "type": "single",
                             "dist": {"A": 0.9, "B": 0.1}, "n": 50}}
    assert checkpoint_matches(ckpt, _fp(learned=learned_changed)) is False
    # 3) 트랜스크립트 본문 변경 (이름은 그대로)
    transcripts_changed = [{"name": "T1", "text": "전혀 다른 인터뷰 내용"}]
    assert checkpoint_matches(ckpt,
                              _fp(transcripts=transcripts_changed)) is False
    # 4) Nemotron facets 변경
    assert checkpoint_matches(ckpt,
                              _fp(nemotron_facets=["age"])) is False

    # generate_responses도 지문 불일치 체크포인트를 거부한다
    svc = GenerationService()
    with pytest.raises(ValueError, match="이어서 진행할 수 없습니다"):
        svc.generate_responses(
            mixed_questions, dims_changed, n,
            **_run_kwargs(mixed_questions, dims_changed, n, seed=seed,
                          batch_size=batch_size, learned=learned,
                          cache={}, checkpoint=ckpt))


def test_resume_already_complete_no_llm_calls(mixed_questions, persona_dims,
                                              learned):
    n, batch_size, seed = 8, 4, 42
    snaps: list[dict] = []
    svc0 = GenerationService()
    df_full = svc0.generate_responses(
        mixed_questions, persona_dims, n,
        **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                      batch_size=batch_size, learned=learned,
                      cache={}, checkpoint_cb=snaps.append))
    final_ckpt = snaps[-1]
    assert final_ckpt["completed_batches"] == 2
    assert len(final_ckpt["rows"]) == 8

    def _boom(pid, prompt, model):
        raise AssertionError("이미 완료된 체크포인트에서 LLM을 호출하면 안 됨")

    svc = GenerationService()
    df = svc.generate_responses(
        mixed_questions, persona_dims, n,
        **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                      batch_size=batch_size, learned=learned,
                      cache={}, generate_fn=_boom, checkpoint=final_ckpt))
    pd.testing.assert_frame_equal(df, df_full)
    assert svc.last_resume_info["status"] == "already_complete"
    assert svc.last_resume_info["resumed_batches"] == 2


def test_checkpoint_cb_called_each_batch(mixed_questions, persona_dims,
                                         learned):
    n, batch_size, seed = 12, 4, 42
    snaps: list[dict] = []
    progress: list[float] = []
    svc = GenerationService()
    df = svc.generate_responses(
        mixed_questions, persona_dims, n,
        **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                      batch_size=batch_size, learned=learned,
                      cache={}, checkpoint_cb=snaps.append,
                      progress_cb=progress.append))
    assert len(df) == 12
    assert [s["completed_batches"] for s in snaps] == [1, 2, 3]
    assert [len(s["rows"]) for s in snaps] == [4, 8, 12]
    assert progress == [1 / 3, 2 / 3, 1.0]
    assert svc.last_checkpoint["completed_batches"] == 3
    assert svc.last_resume_info["status"] == "fresh"
    assert svc.last_resume_info["resumed_batches"] == 0
    # 스냅샷은 이후 append에 오염되지 않는다 (복사본).
    assert len(snaps[0]["rows"]) == 4
