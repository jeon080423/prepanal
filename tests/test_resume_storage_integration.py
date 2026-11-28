"""통합 테스트 — 결함 2×3 경계: 429 중 재로그인 시 세션 프로젝트 유지.

배경: 워커 B의 목록 불완전 가드는 일반 선택 경로(`elif projects and
current_pid not in project_ids`)에만 있었고, 워커 C의
`_handle_fresh_after_verify()`는 목록 불완전 신호를 보지 않았다.
그대로 두면 429 중에 재로그인했을 때 불완전한 목록의 첫 항목을
불러오거나 빈 프로젝트를 새로 만들어 [554] 결함 2 증상(다른 프로젝트
로드 → 덮어쓰기 위험)이 재로그인 경로에서 재현된다. 통합 수정으로
핸들러가 degraded 신호를 먼저 보고 세션 프로젝트를 유지한다.

실제 Sheets·Streamlit에는 접속하지 않는다. FakeSheetsStore로 흉내 낸다.
"""

import pytest

from core.hybrid_projects import HybridProjectStore
from core.projects import ProjectStore
from core.store import MemoryStore
from tests.test_no_auto_create import FakeSheetsStore, _seed_sheets_project


def test_fresh_after_verify_keeps_session_project_when_list_degraded(
        tmp_path, monkeypatch):
    """429로 목록이 불완전하면 재로그인 시 세션 프로젝트를 유지한다.

    시트에는 세션 프로젝트(재개테스트)와 다른 프로젝트(비교실험)가 있다.
    목록 조회를 1회 429로 실패시켜 degraded 신호를 세팅한 뒤 핸들러를
    호출하면, 세션 프로젝트를 그대로 돌려주고 다른 프로젝트를 불러오지도
    빈 프로젝트를 만들지도 않아야 한다.
    """
    import app as app_module

    sheets = FakeSheetsStore()
    session_pid = _seed_sheets_project(
        sheets, "재개테스트", state={"questions": ["Q1"]},
        updated_at="2026-11-28 10:00")
    _seed_sheets_project(
        sheets, "갤럽_684호_비교실험", state={"questions": ["Q9"]},
        updated_at="2026-11-28 09:00")
    hybrid = HybridProjectStore(
        primary_factory=lambda: sheets,
        fallback=ProjectStore(tmp_path / "projects"))
    # 목록 조회 429를 한 번 일으켜 degraded 신호를 세팅한다.
    sheets.fail_list_calls = 1
    hybrid.list_projects()
    assert hybrid.is_list_degraded() is True

    monkeypatch.setattr(app_module, "_project_store", hybrid)
    monkeypatch.setattr(
        app_module, "store", MemoryStore({"project_id": session_pid}))
    result = app_module._handle_fresh_after_verify()

    assert result == session_pid
    assert sheets.create_calls == 0
    assert len(sheets.rows) == 2


def test_fresh_after_verify_still_resolves_when_list_healthy(
        tmp_path, monkeypatch):
    """목록이 건강하면 기존 동작 유지 — 세션 ID 없음 시 최근 프로젝트로 간다."""
    import app as app_module

    sheets = FakeSheetsStore()
    _seed_sheets_project(
        sheets, "오래된 프로젝트", state={"questions": ["Q0"]},
        updated_at="2026-11-27 09:00")
    recent_pid = _seed_sheets_project(
        sheets, "최근 프로젝트", state={"questions": ["Q1"]},
        updated_at="2026-11-28 10:00")
    hybrid = HybridProjectStore(
        primary_factory=lambda: sheets,
        fallback=ProjectStore(tmp_path / "projects"))
    assert hybrid.is_list_degraded() is False

    monkeypatch.setattr(app_module, "_project_store", hybrid)
    monkeypatch.setattr(app_module, "store", MemoryStore({"project_id": None}))
    result = app_module._handle_fresh_after_verify()

    assert result == recent_pid
    assert sheets.create_calls == 0
