"""재개 체크포인트 durable 저장 테스트 — 결함 1 (워커 A).

결함: 번들-3의 배치 중간 저장/재개가 st.session_state에만 살아 있어
새로고침하면 복구가 안 된다. 피주임 재개 테스트 실패
(`재개테스트_ad793782_20명` 20명 생성 중 새로고침 강제 중단 3회 →
재로그인 후 4단계에 "N배치 완료, 이어서 진행"이 전혀 안 뜸).

검증 포인트:
  1. (결함 재현) 세션 전용 체크포인트는 새로고침 시뮬레이션
     (새 MemoryStore로 교체)에서 사라진다 — 수정 전후 모두 이
     메커니즘 자체는 참이다. durable 경로가 없으면 재개 불가.
  2. durable 저장 후 세션 소실 → 체크포인트 복원·재개, 중단 없이
     한 번에 돌린 결과와 행 단위 동일 (gen_seed 포함).
  3. 지문 불일치 시 재개 차단 (checkpoint_matches False +
     generate_responses ValueError).
  4. 완료 후 체크포인트 삭제 (양쪽 백엔드 모두).
  5. Sheets 저장 실패 시 로컬 폴백 후 복구 시 동기화.
  6. 체크포인트 저장은 가벼운 전용 경로 — 프로젝트 전체 상태
     (save_project)를 다시 쓰지 않는다.
  7. 주 저장소에 체크포인트 메서드가 없는 구형 백엔드에서도
     로컬 폴백으로 동작한다 (하위 호환).

실제 LLM 호출 없음 — generate_fn 주입(mock)만 쓴다. 실제 Google
Sheets에는 접속하지 않는다. 비밀값 없음.
"""
import json
import re
import uuid
from datetime import datetime

import pandas as pd
import pytest

from core.hybrid_projects import HybridProjectStore
from core.projects import ProjectStore
from core.services import (GenerationService, build_checkpoint_fingerprint,
                           checkpoint_matches, format_checkpoint_status)
from core.store import MemoryStore


def _profile_count(prompt: str) -> int:
    return len(re.findall(r"(?m)^\d+\.\s", prompt))


def fake_generate(provider_id, prompt, model):
    """test_checkpoint_resume.py의 fake_generate와 동일한 동작."""
    n = _profile_count(prompt)
    assert n > 0
    if "[주관식 문항]" in prompt:
        return json.dumps([{"Q3": "배송이 빨라서 좋아요"} for _ in range(n)],
                          ensure_ascii=False)
    return json.dumps([{"Q2": {"예": 0.7, "아니오": 0.3}} for _ in range(n)],
                      ensure_ascii=False)


class _FailAfter:
    """fail_at번째 호출에서 1회만 예외를 내고 이후에는 정상 동작한다."""

    def __init__(self, fn, fail_at: int):
        self.fn = fn
        self.fail_at = fail_at
        self.calls: list[str] = []

    def __call__(self, provider_id, prompt, model):
        self.calls.append(prompt)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("simulated interruption (LLM 실패)")
        return self.fn(provider_id, prompt, model)


class SheetsDownError(Exception):
    """gspread 429(APIError)를 흉내 낸 예외."""


class FakeSheetsStore:
    """SheetsProjectStore와 같은 인터페이스 + 체크포인트 메서드.

    fail=True면 모든 연산이 429처럼 실패한다.
    """

    def __init__(self):
        self.fail = False
        self.rows: dict[str, dict] = {}
        self.checkpoints: dict[str, dict] = {}
        self.save_project_calls = 0

    def _check(self):
        if self.fail:
            raise SheetsDownError(
                "APIError 429: Quota exceeded for quota metric "
                "'Read requests'")

    def list_projects(self):
        self._check()
        projects = [{
            "id": pid,
            "name": row["name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        } for pid, row in self.rows.items()]
        projects.sort(key=lambda p: p["updated_at"], reverse=True)
        return projects

    def create_project(self, name):
        clean = name.strip()
        if not clean:
            raise ValueError("프로젝트 이름을 입력해 주세요.")
        self._check()
        pid = uuid.uuid4().hex[:12]
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.rows[pid] = {"name": clean, "created_at": now,
                          "updated_at": now, "state": {}}
        return pid

    def save_project(self, project_id, state):
        self._check()
        self.save_project_calls += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        if project_id not in self.rows:
            self.rows[project_id] = {"name": "새 프로젝트",
                                     "created_at": now, "updated_at": now,
                                     "state": {}}
        self.rows[project_id]["state"] = dict(state)
        self.rows[project_id]["updated_at"] = now

    def load_project(self, project_id):
        self._check()
        row = self.rows.get(project_id)
        return dict(row["state"]) if row else {}

    def get_meta(self, project_id):
        self._check()
        row = self.rows.get(project_id)
        if not row:
            return {}
        return {"id": project_id, "name": row["name"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]}

    def rename_project(self, project_id, new_name):
        self._check()
        if project_id in self.rows:
            self.rows[project_id]["name"] = new_name.strip()

    def delete_project(self, project_id):
        self._check()
        self.rows.pop(project_id, None)

    # --- 체크포인트 전용 경로 (가벼운 쓰기: 프로젝트 상태와 분리) ---
    def save_checkpoint(self, project_id, checkpoint, gen_seed=None):
        self._check()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.checkpoints[project_id] = {
            "checkpoint": checkpoint, "gen_seed": gen_seed,
            "updated_at": now}

    def load_checkpoint(self, project_id):
        self._check()
        payload = self.checkpoints.get(project_id)
        return dict(payload) if payload else None

    def delete_checkpoint(self, project_id):
        self._check()
        self.checkpoints.pop(project_id, None)


class LegacyFakeSheetsStore(FakeSheetsStore):
    """체크포인트 메서드가 없는 구형 주 백엔드 (하위 호환 검증용)."""

    def __init__(self):
        super().__init__()
        # 인스턴스 속성을 None으로 가려 클래스 메서드를 타지 않게 한다 —
        # Hybrid가 getattr()으로 메서드를 찾을 때 None이 나와
        # "메서드 없음"으로 취급해야 한다.
        self.save_checkpoint = None
        self.load_checkpoint = None
        self.delete_checkpoint = None


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


@pytest.fixture
def sheets():
    return FakeSheetsStore()


@pytest.fixture
def hybrid(sheets, tmp_path):
    return HybridProjectStore(primary_factory=lambda: sheets,
                              fallback=ProjectStore(tmp_path / "projects"))


def fresh_hybrid(sheets, tmp_path):
    """rerun 시뮬레이션: 소유권 맵이 비어 있는 새 하이브리드 인스턴스."""
    return HybridProjectStore(primary_factory=lambda: sheets,
                              fallback=ProjectStore(tmp_path / "projects"))


def _run_kwargs(questions, persona_dims, n, *, seed=42, batch_size=4,
                generate_fn=fake_generate, cache=None, **extra):
    kw = dict(provider_id="gemini", model="dummy", batch_size=batch_size,
              seed=seed, generate_fn=generate_fn)
    if cache is not None:
        kw["cache"] = cache
    kw.update(extra)
    return kw


def _fingerprint(questions, n, *, seed=42, batch_size=4, learned=None,
                 persona_dims=None):
    return build_checkpoint_fingerprint(
        questions, n, batch_size=batch_size, seed=seed,
        provider_id="gemini", model="dummy", persona_source="manual",
        persona_dims=persona_dims, learned=learned)


def _interrupted_checkpoint(questions, persona_dims, learned, *, n=12,
                            batch_size=4, seed=42):
    """1배치 완료 후 중단된 실행의 체크포인트를 만든다."""
    cache: dict = {}
    failing = _FailAfter(fake_generate, fail_at=3)
    svc = GenerationService()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        svc.generate_responses(
            questions, persona_dims, n,
            **_run_kwargs(questions, persona_dims, n, seed=seed,
                          batch_size=batch_size, learned=learned,
                          cache=cache, generate_fn=failing))
    assert svc.last_checkpoint["completed_batches"] == 1
    assert len(svc.last_checkpoint["rows"]) == 4
    return svc.last_checkpoint


# ----------------------------------------------------------------------------
# 1. 결함 재현: 세션 전용 체크포인트는 새로고침 시 사라진다
# ----------------------------------------------------------------------------
def test_session_only_checkpoint_lost_after_refresh(mixed_questions,
                                                    persona_dims, learned):
    """결함 메커니즘 증명. 세션 상태(MemoryStore)에만 둔 체크포인트는
    새로고침(= 새 Store로 교체) 후 None이 되어 4단계 재개 UI가 뜰 수
    없다. durable 경로가 없으면 이 상태가 곧 재개 불가다."""
    ckpt = _interrupted_checkpoint(mixed_questions, persona_dims, learned)
    session_before = MemoryStore({"gen_checkpoint": ckpt, "gen_seed": 42})
    assert session_before.get("gen_checkpoint") is not None
    # 새로고침: Streamlit 세션 상태가 통째로 사라지고 새 Store가 생긴다.
    session_after_refresh = MemoryStore()
    assert session_after_refresh.get("gen_checkpoint") is None
    assert session_after_refresh.get("gen_seed") is None


# ----------------------------------------------------------------------------
# 2. durable 저장 후 세션 소실 → 복원·재개, 결과 행 단위 동일
# ----------------------------------------------------------------------------
def test_durable_checkpoint_survives_session_loss_and_resumes(
        mixed_questions, persona_dims, learned, hybrid, sheets, tmp_path):
    n, batch_size, seed = 12, 4, 42
    # (a) 중단 없이 한 번에 돌린 기준 결과
    df_full = GenerationService().generate_responses(
        mixed_questions, persona_dims, n,
        **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                      batch_size=batch_size, learned=learned, cache={}))

    # (b) 1배치 후 중단 → 체크포인트를 durable 저장소에 저장
    ckpt = _interrupted_checkpoint(mixed_questions, persona_dims, learned,
                                   n=n, batch_size=batch_size, seed=seed)
    pid = hybrid.create_project("재개테스트")
    assert pid in sheets.rows  # Sheets에 만든 프로젝트
    hybrid.save_checkpoint(pid, ckpt, gen_seed=seed)

    # (c) 새로고침 시뮬레이션: 새 세션 Store + 새 하이브리드 인스턴스
    #     (rerun). gen_cache도 세션 전용이라 함께 사라진다.
    fresh_session = MemoryStore()
    assert fresh_session.get("gen_checkpoint") is None
    reloaded = fresh_hybrid(sheets, tmp_path)
    payload = reloaded.load_checkpoint(pid)
    assert payload is not None
    assert payload["gen_seed"] == seed  # seed도 함께 durable 저장됨
    restored_cp = payload["checkpoint"]
    # 세션으로 복원 (app.py _load_project_to_store가 하는 일)
    fresh_session.set("gen_checkpoint", restored_cp)
    fresh_session.set("gen_seed", payload["gen_seed"])
    fp = _fingerprint(mixed_questions, n, seed=seed, batch_size=batch_size,
                      learned=learned, persona_dims=persona_dims)
    assert checkpoint_matches(restored_cp, fp) is True
    assert format_checkpoint_status(restored_cp).startswith(
        "1배치 완료, 이어서 진행")

    # (d) 복원된 체크포인트로 재개 — 캐시가 비어도 결과는 동일하다
    svc2 = GenerationService()
    df_resumed = svc2.generate_responses(
        mixed_questions, persona_dims, n,
        **_run_kwargs(mixed_questions, persona_dims, n, seed=seed,
                      batch_size=batch_size, learned=learned,
                      cache={}, generate_fn=fake_generate,
                      checkpoint=fresh_session.get("gen_checkpoint")))
    pd.testing.assert_frame_equal(df_resumed, df_full)
    assert svc2.last_resume_info["resumed_batches"] == 1
    assert svc2.last_resume_info["status"] == "resumed"


# ----------------------------------------------------------------------------
# 3. 지문 불일치 시 재개 차단
# ----------------------------------------------------------------------------
def test_durable_checkpoint_fingerprint_mismatch_blocks_resume(
        mixed_questions, persona_dims, learned, hybrid, sheets):
    ckpt = _interrupted_checkpoint(mixed_questions, persona_dims, learned)
    pid = hybrid.create_project("지문테스트")
    hybrid.save_checkpoint(pid, ckpt, gen_seed=42)
    payload = hybrid.load_checkpoint(pid)
    assert payload is not None

    # seed가 바뀐 조건으로는 재개 불가
    fp_other_seed = _fingerprint(mixed_questions, 12, seed=43,
                                 batch_size=4, learned=learned,
                                 persona_dims=persona_dims)
    assert checkpoint_matches(payload["checkpoint"], fp_other_seed) is False
    svc = GenerationService()
    with pytest.raises(ValueError, match="이어서 진행할 수 없습니다"):
        svc.generate_responses(
            mixed_questions, persona_dims, 12,
            **_run_kwargs(mixed_questions, persona_dims, 12, seed=43,
                          batch_size=4, learned=learned, cache={},
                          checkpoint=payload["checkpoint"]))

    # 응답자 수가 바뀐 조건으로도 재개 불가
    fp_other_n = _fingerprint(mixed_questions, 8, seed=42,
                              batch_size=4, learned=learned,
                              persona_dims=persona_dims)
    assert checkpoint_matches(payload["checkpoint"], fp_other_n) is False


# ----------------------------------------------------------------------------
# 4. 완료 후 체크포인트 삭제
# ----------------------------------------------------------------------------
def test_durable_checkpoint_deleted_after_completion(
        mixed_questions, persona_dims, learned, hybrid, sheets, tmp_path):
    ckpt = _interrupted_checkpoint(mixed_questions, persona_dims, learned)
    pid = hybrid.create_project("삭제테스트")
    hybrid.save_checkpoint(pid, ckpt, gen_seed=42)
    assert hybrid.load_checkpoint(pid) is not None
    assert pid in sheets.checkpoints

    hybrid.delete_checkpoint(pid)
    assert hybrid.load_checkpoint(pid) is None
    assert pid not in sheets.checkpoints  # Sheets에서도 삭제됨
    # 로컬 폴백 파일도 남아 있지 않다
    assert list((tmp_path / "projects").glob("*.checkpoint")) == []
    # 없는 체크포인트 삭제는 no-op (예외 없음)
    hybrid.delete_checkpoint(pid)
    assert hybrid.load_checkpoint(pid) is None


# ----------------------------------------------------------------------------
# 5. Sheets 저장 실패 시 로컬 폴백 후 복구 시 동기화
# ----------------------------------------------------------------------------
def test_sheets_failure_falls_back_to_local_then_syncs(
        mixed_questions, persona_dims, learned, hybrid, sheets, tmp_path):
    ckpt = _interrupted_checkpoint(mixed_questions, persona_dims, learned)
    pid = hybrid.create_project("폴백테스트")
    hybrid.save_project(pid, {"questions": ["Q1"], "provider": "gemini"})
    state_before = dict(sheets.rows[pid]["state"])

    # (a) 429 장애 중 저장 → Sheets에는 안 남고 로컬에만 남는다
    sheets.fail = True
    hybrid.save_checkpoint(pid, ckpt, gen_seed=42)
    assert pid not in sheets.checkpoints
    local_files = list((tmp_path / "projects").glob("*.checkpoint"))
    assert len(local_files) == 1

    # (b) 장애 중에도 새 인스턴스(rerun)에서 로컬 폴백으로 읽힌다
    outage = fresh_hybrid(sheets, tmp_path)
    payload = outage.load_checkpoint(pid)
    assert payload is not None
    assert payload["checkpoint"]["completed_batches"] == 1
    assert payload["gen_seed"] == 42

    # (c) 복구 후 읽으면 Sheets로 동기화된다
    sheets.fail = False
    recovered = fresh_hybrid(sheets, tmp_path)
    payload2 = recovered.load_checkpoint(pid)
    assert payload2 is not None
    assert payload2["checkpoint"]["completed_batches"] == 1
    assert pid in sheets.checkpoints  # 동기화로 Sheets에 생김

    # (d) 동기화 뒤에는 로컬 파일이 사라져도(리부트) Sheets에서 읽힌다
    for f in (tmp_path / "projects").glob("*.checkpoint"):
        f.unlink()
    healed = fresh_hybrid(sheets, tmp_path)
    payload3 = healed.load_checkpoint(pid)
    assert payload3 is not None
    assert payload3["gen_seed"] == 42

    # (e) 체크포인트 저장·동기화가 프로젝트 전체 상태를 건드리지 않았다
    assert sheets.rows[pid]["state"] == state_before


# ----------------------------------------------------------------------------
# 6. 가벼운 전용 경로 — 프로젝트 전체 상태를 다시 쓰지 않는다
# ----------------------------------------------------------------------------
def test_checkpoint_save_does_not_rewrite_project_state(
        mixed_questions, persona_dims, learned, hybrid, sheets, tmp_path):
    ckpt = _interrupted_checkpoint(mixed_questions, persona_dims, learned)
    pid = hybrid.create_project("경량테스트")
    hybrid.save_project(pid, {"questions": ["Q1"], "provider": "gemini"})
    local_path = tmp_path / "projects" / f"{pid}.json"
    assert local_path.exists() is False  # Sheets 소유라 로컬 본이 없음
    state_before = dict(sheets.rows[pid]["state"])
    calls_before = sheets.save_project_calls

    # 체크포인트 3회 저장(배치 3개분) — 프로젝트 저장은 0회여야 한다
    for completed in (1, 2, 3):
        snap = dict(ckpt, completed_batches=completed)
        hybrid.save_checkpoint(pid, snap, gen_seed=42)
    assert sheets.save_project_calls == calls_before
    assert sheets.rows[pid]["state"] == state_before
    assert local_path.exists() is False


# ----------------------------------------------------------------------------
# 7. 구형 주 백엔드(체크포인트 메서드 없음)에서도 로컬 폴백으로 동작
# ----------------------------------------------------------------------------
def test_legacy_primary_without_checkpoint_methods_falls_back_to_local(
        mixed_questions, persona_dims, learned, tmp_path):
    legacy = LegacyFakeSheetsStore()
    store = HybridProjectStore(primary_factory=lambda: legacy,
                               fallback=ProjectStore(tmp_path / "projects"))
    ckpt = _interrupted_checkpoint(mixed_questions, persona_dims, learned)
    pid = store.create_project("구형테스트")
    # 예외 없이 로컬 폴백에 저장되고, 다시 읽힌다
    store.save_checkpoint(pid, ckpt, gen_seed=42)
    payload = store.load_checkpoint(pid)
    assert payload is not None
    assert payload["checkpoint"]["completed_batches"] == 1
    assert payload["gen_seed"] == 42
    store.delete_checkpoint(pid)
    assert store.load_checkpoint(pid) is None


# ----------------------------------------------------------------------------
# 8. 로컬 전용 모드(primary 없음) + 저장된 기록이 없을 때
# ----------------------------------------------------------------------------
def test_local_only_mode_checkpoint_roundtrip(
        mixed_questions, persona_dims, learned, tmp_path):
    store = HybridProjectStore(
        primary_factory=None, fallback=ProjectStore(tmp_path / "projects"))
    pid = store.create_project("로컬전용")
    assert store.load_checkpoint(pid) is None  # 저장된 기록 없음
    ckpt = _interrupted_checkpoint(mixed_questions, persona_dims, learned)
    store.save_checkpoint(pid, ckpt, gen_seed=42)
    reloaded = HybridProjectStore(
        primary_factory=None, fallback=ProjectStore(tmp_path / "projects"))
    payload = reloaded.load_checkpoint(pid)
    assert payload is not None
    assert payload["gen_seed"] == 42
    assert payload["checkpoint"]["completed_batches"] == 1
