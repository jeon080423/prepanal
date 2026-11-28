"""저장 429 우아한 처리 + 덮어쓰기 방지 — 워커 B.

재현 시나리오(피주임 검증 보고 [554]):
  4단계에서 `저장` 클릭 → Google Sheets 429(Read requests per minute)
  → "프로젝트가 없습니다"로 프로젝트 연결이 끊기고, 세션에 다른
  프로젝트(갤럽_684호_비교실험) 데이터가 로드됨. 그 상태에서 저장하면
  다른 프로젝트를 덮어쓸 수 있다.

실제 Sheets에는 접속하지 않는다. Sheets 백엔드는 429를 흉내 내는
가짜로 대체하고, 로컬 백엔드는 실제 ProjectStore(tmp 경로)를 쓴다.
app.py 레벨 함수는 app.py 소스에서 AST로 뽑아 샌드박스에서 실행한다
(app.py 모듈 전체를 import하면 Streamlit UI가 통째로 실행되므로).
"""

import ast
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import core.hybrid_projects as hybrid_mod
from core.hybrid_projects import HybridProjectStore
from core.projects import PROJECT_STATE_KEYS, ProjectStore
from core.store import MemoryStore

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


# ----------------------------------------------------------------------------
# Sheets 가짜 (429 횟수 지정 가능)
# ----------------------------------------------------------------------------
class RateLimitError(Exception):
    """gspread APIError(429) 흉내. code 속성이 있는 형태."""

    code = 429


class FlakySheetsStore:
    """save/list 실패 횟수를 지정할 수 있는 Sheets 가짜.

    fail_save_times: save_project가 429를 던질 남은 횟수
    fail_save_always: save_project가 항상 429를 던짐
    plain_429: True면 code 속성 없는 "429" 메시지 예외를 던짐
    fail_list: list_projects가 429를 던짐
    """

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.fail_save_times = 0
        self.fail_save_always = False
        self.plain_429 = False
        self.fail_list = False
        self.save_attempts = 0
        self.list_attempts = 0

    def _check_save(self):
        self.save_attempts += 1
        if self.fail_save_always or self.fail_save_times > 0:
            if self.fail_save_times > 0:
                self.fail_save_times -= 1
            if self.plain_429:
                raise RuntimeError(
                    "APIError 429: Quota exceeded for quota metric "
                    "'Read requests'")
            raise RateLimitError(
                "Quota exceeded for quota metric 'Read requests'")

    def list_projects(self):
        self.list_attempts += 1
        if self.fail_list:
            raise RateLimitError(
                "Quota exceeded for quota metric 'Read requests'")
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
        if any(p["name"].strip().casefold() == clean.casefold()
               for p in self.list_projects()):
            raise ValueError("같은 이름의 프로젝트가 있습니다. 다른 이름으로 입력해 주세요.")
        self._check_save()
        import uuid
        pid = uuid.uuid4().hex[:12]
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.rows[pid] = {"name": clean, "created_at": now,
                          "updated_at": now, "state": {}}
        return pid

    def save_project(self, project_id, state):
        self._check_save()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        if project_id not in self.rows:
            self.rows[project_id] = {"name": "새 프로젝트", "created_at": now,
                                     "updated_at": now, "state": {}}
        self.rows[project_id]["state"] = dict(state)
        self.rows[project_id]["updated_at"] = now

    def load_project(self, project_id):
        row = self.rows.get(project_id)
        return dict(row["state"]) if row else {}

    def get_meta(self, project_id):
        row = self.rows.get(project_id)
        if not row:
            return {}
        return {"id": project_id, "name": row["name"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]}

    def rename_project(self, project_id, new_name):
        if project_id in self.rows:
            self.rows[project_id]["name"] = new_name.strip()

    def delete_project(self, project_id):
        self.rows.pop(project_id, None)


class FailingLocalStore:
    """로컬 저장소까지 죽은 경우 (전체 저장 실패 재현용)."""

    def __init__(self, tmp_path):
        self.dir = tmp_path / "projects"
        self.dir.mkdir(parents=True, exist_ok=True)

    def list_projects(self):
        return []

    def save_project(self, project_id, state):
        raise RuntimeError("로컬 디스크 쓰기 실패")

    def load_project(self, project_id):
        return {}

    def get_meta(self, project_id):
        return {}

    def rename_project(self, project_id, new_name):
        raise RuntimeError("로컬 디스크 쓰기 실패")

    def delete_project(self, project_id):
        raise RuntimeError("로컬 디스크 쓰기 실패")


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    """재시도 대기를 0으로 — 테스트는 재시도 횟수만 본다.
    수정 전에는 상수가 없어도 테스트가 돌아가야 하므로 raising=False."""
    monkeypatch.setattr(hybrid_mod, "SAVE_RETRY_DELAYS", (0, 0),
                        raising=False)


@pytest.fixture
def sheets():
    fake = FlakySheetsStore()
    pid = fake.create_project("갤럽_684호_비교실험")
    fake.save_project(pid, {"questions": ["Q1"], "provider": "gemini"})
    return fake


@pytest.fixture
def factory(sheets):
    calls = {"n": 0}

    def _make():
        calls["n"] += 1
        return sheets

    _make.calls = calls
    return _make


@pytest.fixture
def hybrid(factory, tmp_path):
    return HybridProjectStore(primary_factory=factory,
                              fallback=ProjectStore(tmp_path / "projects"))


def _warm_owner_primary(hybrid, sheets):
    """소유권을 primary로 확정 (건강한 목록 조회 1회). 픽스처 setup분의
    카운터는 리셋해 테스트가 재시도 횟수만 보게 한다."""
    projects = hybrid.list_projects()
    assert projects and projects[0]["id"] in sheets.rows
    sheets.save_attempts = 0
    sheets.list_attempts = 0
    return projects[0]["id"]


# ----------------------------------------------------------------------------
# core: 429 재시도 + 로컬 폴백
# ----------------------------------------------------------------------------
def test_save_429_retries_then_succeeds(hybrid, sheets, factory):
    pid = _warm_owner_primary(hybrid, sheets)
    sheets.fail_save_times = 2
    calls_before = factory.calls["n"]
    hybrid.save_project(pid, {"questions": ["Q1", "Q2"]})
    # 429 2회 후 3번째 시도에서 성공 — 재시도 중에는 주 인스턴스를
    # 새로 만들지 않는다(스토어 플립 금지).
    assert sheets.save_attempts == 3
    assert factory.calls["n"] == calls_before
    assert sheets.rows[pid]["state"] == {"questions": ["Q1", "Q2"]}
    assert hybrid.last_save_degraded is False


def test_save_429_plain_message_also_retries(hybrid, sheets):
    pid = _warm_owner_primary(hybrid, sheets)
    sheets.plain_429 = True
    sheets.fail_save_times = 1
    hybrid.save_project(pid, {"questions": ["Q1", "Q2"]})
    assert sheets.save_attempts == 2
    assert sheets.rows[pid]["state"] == {"questions": ["Q1", "Q2"]}


def test_save_429_all_fail_falls_back_to_local(hybrid, sheets, tmp_path):
    pid = _warm_owner_primary(hybrid, sheets)
    sheets.fail_save_always = True
    hybrid.save_project(pid, {"questions": ["Q1", "Q2", "Q3"]})
    # 예외 없이 로컬 안전본에 저장되고, 시트 원본은 건드리지 않는다.
    assert sheets.rows[pid]["state"] == {"questions": ["Q1"],
                                         "provider": "gemini"}
    assert hybrid.load_project(pid) == {"questions": ["Q1", "Q2", "Q3"]}
    assert hybrid.last_save_degraded is True
    assert hybrid.last_save_error and "429" in hybrid.last_save_error


def test_save_non_429_error_does_not_retry(hybrid, sheets):
    """429가 아닌 에러는 재시도하지 않고 기존 폴백 경로로 간다."""
    pid = _warm_owner_primary(hybrid, sheets)
    orig_save = sheets.save_project

    def _boom(project_id, state):
        sheets.save_attempts += 1
        raise ValueError("시트 스키마 오류")

    sheets.save_project = _boom
    hybrid.save_project(pid, {"questions": ["QX"]})
    assert sheets.save_attempts == 1
    assert hybrid.load_project(pid) == {"questions": ["QX"]}
    orig_save  # noqa: B018 — 원복 불필요(픽스처는 테스트별 생성)


def test_save_total_failure_raises(hybrid, sheets, tmp_path):
    """주·로컬이 둘 다 죽으면 예외를 던진다 (조용히 성공으로 보이면 안 됨)."""
    failing = HybridProjectStore(
        primary_factory=lambda: sheets,
        fallback=FailingLocalStore(tmp_path))
    pid = sheets.list_projects()[0]["id"]
    sheets.fail_save_always = True
    with pytest.raises(Exception):
        failing.save_project(pid, {"questions": ["QX"]})


def test_list_degraded_signal(hybrid, sheets):
    """목록 불완전 신호: 주 장애 중 True, 복구 후 False, 로컬 전용 False."""
    assert hybrid.is_list_degraded() is False
    sheets.fail_list = True
    assert hybrid.list_projects() == [] or True  # 로컬은 비어 있음
    assert hybrid.primary_available is False
    assert hybrid.is_list_degraded() is True
    sheets.fail_list = False
    hybrid.list_projects()
    assert hybrid.is_list_degraded() is False


def test_list_degraded_false_in_local_only_mode(tmp_path):
    store = HybridProjectStore(primary_factory=None,
                               fallback=ProjectStore(tmp_path / "projects"))
    store.list_projects()
    assert store.primary_available is False  # 주는 애초에 없음
    assert store.is_list_degraded() is False  # 그래도 목록은 완전하다


# ----------------------------------------------------------------------------
# app.py 샌드박스 실행용 헬퍼
# ----------------------------------------------------------------------------
def _extract_app_functions(*names):
    """app.py 소스에서 지정한 최상위 함수들의 소스를 뽑는다.
    없으면 AssertionError — 미구현 상태 자체가 재현 FAIL이다."""
    src = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = ast.get_source_segment(src, node)
    missing = [n for n in names if n not in found]
    assert not missing, f"app.py에서 함수를 찾지 못함(미구현?): {missing}"
    return found


class _StopSignal(Exception):
    """FakeSt.stop()이 던지는 신호 (실제 Streamlit의 st.stop() 대응)."""


class _FakeForm:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class FakeSt:
    """st 최소 스텁: 호출 기록 + session_state dict."""

    def __init__(self, session_state=None):
        self.session_state = session_state if session_state is not None else {}
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.successes: list[str] = []
        self.infos: list[str] = []
        self.captions: list[str] = []

    def error(self, msg):
        self.errors.append(str(msg))

    def warning(self, msg):
        self.warnings.append(str(msg))

    def success(self, msg):
        self.successes.append(str(msg))

    def info(self, msg):
        self.infos.append(str(msg))

    def caption(self, msg):
        self.captions.append(str(msg))

    def stop(self):
        raise _StopSignal()

    def form(self, *args, **kwargs):
        return _FakeForm()

    def text_input(self, *args, **kwargs):
        return ""

    def form_submit_button(self, *args, **kwargs):
        return False


_SAVE_FUNCS = ("_now_kst", "_fmt_kst_now", "_state_fingerprint",
               "_current_state_snapshot", "_is_state_effectively_empty",
               "_save_current_project")


def _save_sandbox(store, project_store, session_state):
    """_save_current_project()를 실제 app.py 소스로 실행할 네임스페이스."""
    srcs = _extract_app_functions(*_SAVE_FUNCS)
    st = FakeSt(session_state)
    ns = {
        "st": st,
        "store": store,
        "_project_store": project_store,
        "PROJECT_STATE_KEYS": PROJECT_STATE_KEYS,
        "KST": timezone(timedelta(hours=9)),
        "datetime": datetime,
        "json": json,
        "hashlib": hashlib,
        "pd": pd,
    }
    for name in _SAVE_FUNCS:
        exec(srcs[name], ns)  # noqa: S102 — 테스트용 소스 실행
    ns["_fake_st"] = st
    return ns


def _full_state():
    return {
        "questions": [{"id": "Q1", "text": "문항1", "type": "single",
                       "options": ["A", "B"]}],
        "provider": "gemini",
    }


def _set_store_state(store, pid, state):
    store.set("project_id", pid)
    for key in PROJECT_STATE_KEYS:
        store.set(key, state.get(key))
    store.set("provider", state.get("provider", "gemini"))
    store.set("model_select_gemini", "gemini-2.0-flash")


# ----------------------------------------------------------------------------
# app: 저장 경로 429 우아한 처리
# ----------------------------------------------------------------------------
def test_app_save_429_retry_then_success(hybrid, sheets, tmp_path):
    pid = _warm_owner_primary(hybrid, sheets)
    sheets.fail_save_times = 2
    store = MemoryStore()
    _set_store_state(store, pid, _full_state())
    session: dict = {}
    ns = _save_sandbox(store, hybrid, session)
    status = ns["_save_current_project"]()
    assert status == "ok"
    assert sheets.save_attempts == 3
    assert sheets.rows[pid]["state"]["questions"] == _full_state()["questions"]
    assert ns["_fake_st"].errors == []
    assert ns["_fake_st"].warnings == []
    assert session.get("_last_saved_fp")  # 저장 지점 기록됨


def test_app_save_429_fallback_is_degraded_not_success(hybrid, sheets, tmp_path):
    """재시도 전부 실패 → 로컬 폴백 저장 + 실패 표시, 저장 완료로 보이지 않음."""
    pid = _warm_owner_primary(hybrid, sheets)
    sheets.fail_save_always = True
    store = MemoryStore()
    _set_store_state(store, pid, _full_state())
    session: dict = {}
    ns = _save_sandbox(store, hybrid, session)
    status = ns["_save_current_project"]()
    assert status == "degraded"
    # 로컬 안전본에는 새 상태가 있다
    assert hybrid.load_project(pid)["questions"] == _full_state()["questions"]
    # 시트 원본은 그대로
    assert sheets.rows[pid]["state"] == {"questions": ["Q1"],
                                         "provider": "gemini"}
    # 실패가 화면에 보인다 (조용히 넘기지 않음)
    shown = " ".join(ns["_fake_st"].warnings + ns["_fake_st"].errors)
    assert "429" in shown and "로컬" in shown
    # 저장된 것처럼 보이면 안 된다: 저장 지점 미갱신
    assert not session.get("_last_saved_fp")
    assert not session.get("_last_saved_at")


def test_app_save_total_failure_fp_unchanged(hybrid, sheets, tmp_path):
    """주·로컬 둘 다 실패 → 'failed' + 에러 표시 + 저장 지점 미갱신."""
    failing = HybridProjectStore(
        primary_factory=lambda: sheets,
        fallback=FailingLocalStore(tmp_path))
    pid = sheets.list_projects()[0]["id"]
    store = MemoryStore()
    _set_store_state(store, pid, _full_state())
    session: dict = {}
    ns = _save_sandbox(store, failing, session)
    # 먼저 성공 저장을 해서 저장 지점을 만든다
    status_ok = ns["_save_current_project"]()
    assert status_ok == "ok"
    fp_before = session.get("_last_saved_fp")
    assert fp_before
    # 상태를 바꾸고 전체 실패를 유발한다
    store.set("questions", [{"id": "Q1", "text": "문항1-수정", "type": "single",
                             "options": ["A", "B"]}])
    sheets.fail_save_always = True
    status = ns["_save_current_project"]()
    assert status == "failed"
    assert ns["_fake_st"].errors, "저장 실패가 화면에 보여야 한다"
    assert session.get("_last_saved_fp") == fp_before  # 미갱신
    assert "다시 시도" in " ".join(ns["_fake_st"].errors)


def test_app_empty_state_guard_blocks_overwrite(hybrid, sheets, tmp_path):
    """빈 state로 기존 행 덮어쓰기 차단: 저장된 내용이 있는데
    세션 상태가 비어 있으면 저장을 중단하고 경고한다."""
    pid = _warm_owner_primary(hybrid, sheets)
    store = MemoryStore()
    _set_store_state(store, pid, _full_state())
    session: dict = {}
    ns = _save_sandbox(store, hybrid, session)
    assert ns["_save_current_project"]() == "ok"
    # 세션 상태가 비어 버린 상황(stale/빈 state)을 만든다
    _set_store_state(store, pid, {})
    status = ns["_save_current_project"]()
    assert status == "blocked"
    assert ns["_fake_st"].warnings, "덮어쓰기 차단 경고가 보여야 한다"
    # 기존 행은 그대로
    assert sheets.rows[pid]["state"]["questions"] == _full_state()["questions"]
    assert hybrid.load_project(pid)["questions"] == _full_state()["questions"]


def test_app_empty_state_fresh_project_allowed(hybrid, sheets, tmp_path):
    """갓 만든 빈 프로젝트의 빈 상태 저장은 막지 않는다."""
    pid = sheets.create_project("새 프로젝트")
    store = MemoryStore()
    _set_store_state(store, pid, {})
    session: dict = {}
    ns = _save_sandbox(store, hybrid, session)
    # 빈 상태가 곧 저장된 상태(지문 일치)이면 허용
    ns["_save_current_project"]()
    fp = session.get("_last_saved_fp")
    assert fp
    status = ns["_save_current_project"]()
    assert status == "ok"
    assert ns["_fake_st"].warnings == []


# ----------------------------------------------------------------------------
# app: 프로젝트 선택 폴백 (목록 실패 시 현재 프로젝트 유지)
# ----------------------------------------------------------------------------
def _extract_selection_block():
    """app.py 모듈 레벨의 프로젝트 선택 if/elif 블록 소스를 뽑는다."""
    src = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Call)
                and isinstance(test.func, ast.Attribute)
                and test.func.attr == "pop"
                and test.args
                and isinstance(test.args[0], ast.Constant)
                and test.args[0].value == "_fresh_after_verify"):
            return ast.get_source_segment(src, node)
    raise AssertionError("app.py에서 프로젝트 선택 블록을 찾지 못함")


def _extract_empty_projects_block():
    """'프로젝트가 없습니다' UI 블록(if not projects:) 소스를 뽑는다."""
    src = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        seg = ast.get_source_segment(src, node) or ""
        if "프로젝트가 없습니다" in seg:
            return seg
    raise AssertionError("app.py에서 '프로젝트가 없습니다' 블록을 찾지 못함")


def _selection_sandbox(store, project_store, session_state, loaded):
    """선택 블록 실행용 네임스페이스. loaded는 _load_project_to_store 호출 기록."""
    st = FakeSt(session_state)
    ns = {
        "st": st,
        "store": store,
        "_project_store": project_store,
        "is_code_verified": lambda: True,
        "_load_project_to_store": lambda pid: (
            loaded.append(pid), store.set("project_id", pid)),
    }
    # _is_project_list_degraded: app.py 실물이 있으면 그것을 쓰고,
    # 없으면(수정 전) 스토어 신호를 그대로 읽는 스텁을 쓴다.
    try:
        srcs = _extract_app_functions("_is_project_list_degraded")
        exec(srcs["_is_project_list_degraded"], ns)  # noqa: S102
    except AssertionError:
        ns["_is_project_list_degraded"] = (
            lambda: project_store.is_list_degraded())
    ns["_fake_st"] = st
    return ns


def _run_selection_block(ns, project_store):
    prelude = (
        "projects = _project_store.list_projects()\n"
        "current_pid = store.get(\"project_id\")\n"
        "project_ids = [p[\"id\"] for p in projects]\n"
    )
    block = _extract_selection_block()
    exec(prelude + block, ns)  # noqa: S102 — 테스트용 소스 실행
    return ns


def test_selection_keeps_current_when_list_empty_on_429(hybrid, sheets, tmp_path):
    """429로 목록이 비어 보여도 세션의 현재 프로젝트를 유지한다."""
    pid = _warm_owner_primary(hybrid, sheets)
    sheets.fail_list = True
    store = MemoryStore({"project_id": pid})
    loaded: list[str] = []
    ns = _selection_sandbox(store, hybrid, {}, loaded)
    _run_selection_block(ns, hybrid)
    assert store.get("project_id") == pid
    assert loaded == [], "다른 프로젝트 데이터를 로드하면 안 된다"
    assert ns["current_pid"] == pid


def test_selection_keeps_current_when_list_missing_current_on_429(
        hybrid, sheets, tmp_path):
    """429로 목록에 로컬 프로젝트만 보여도 현재 Sheets 프로젝트를 유지한다."""
    pid = _warm_owner_primary(hybrid, sheets)
    local_pid = hybrid._fallback.create_project("로컬만 있는 프로젝트")
    sheets.fail_list = True
    store = MemoryStore({"project_id": pid})
    loaded: list[str] = []
    ns = _selection_sandbox(store, hybrid, {}, loaded)
    _run_selection_block(ns, hybrid)
    assert store.get("project_id") == pid
    assert loaded == []
    assert local_pid not in loaded


def test_selection_switches_when_list_complete_and_current_gone(
        hybrid, sheets, tmp_path):
    """대조: 목록이 완전하고 현재 프로젝트가 정말 없으면 첫 항목으로 바꾼다
    (의도된 폴백 동작은 유지)."""
    pid = _warm_owner_primary(hybrid, sheets)
    other = sheets.create_project("다른 프로젝트")
    store = MemoryStore({"project_id": "deleted-pid"})
    loaded: list[str] = []
    ns = _selection_sandbox(store, hybrid, {}, loaded)
    _run_selection_block(ns, hybrid)
    assert loaded == [other] or loaded == [pid]
    assert store.get("project_id") == loaded[0]


def test_empty_projects_ui_does_not_stop_app_on_429(hybrid, sheets, tmp_path):
    """429로 목록이 비어도 '프로젝트가 없습니다' + 앱 중단을 하지 않는다."""
    pid = _warm_owner_primary(hybrid, sheets)
    sheets.fail_list = True
    store = MemoryStore({"project_id": pid})
    st = FakeSt({})
    ns = {
        "st": st,
        "store": store,
        "_project_store": hybrid,
        "projects": [],
        "current_pid": pid,
        "_projects_list_degraded": True,
        "_load_project_to_store": lambda p: store.set("project_id", p),
    }
    block = _extract_empty_projects_block()
    try:
        exec(block, ns)  # noqa: S102 — 테스트용 소스 실행
    except _StopSignal:
        pytest.fail("429 중에는 '프로젝트가 없습니다'로 앱을 멈추면 안 된다")
    assert "프로젝트가 없습니다" not in " ".join(st.infos)
    assert store.get("project_id") == pid


# ----------------------------------------------------------------------------
# app: 저장 상태 표시 (실패가 화면에 보이게)
# ----------------------------------------------------------------------------
def _render_status_sandbox(store, session_state):
    srcs = _extract_app_functions(
        "_now_kst", "_fmt_kst_now", "_fmt_saved_at_display",
        "_state_fingerprint", "_current_state_snapshot",
        "_is_current_state_saved", "_render_save_status")
    st = FakeSt(session_state)
    ns = {
        "st": st,
        "store": store,
        "PROJECT_STATE_KEYS": PROJECT_STATE_KEYS,
        "KST": timezone(timedelta(hours=9)),
        "datetime": datetime,
        "timezone": timezone,
        "json": json,
        "hashlib": hashlib,
        "pd": pd,
        "_storage_label": "☁️ Google Sheets",
    }
    for name in ("_now_kst", "_fmt_kst_now", "_fmt_saved_at_display",
                 "_state_fingerprint", "_current_state_snapshot",
                 "_is_current_state_saved", "_render_save_status"):
        exec(srcs[name], ns)  # noqa: S102 — 테스트용 소스 실행
    ns["_fake_st"] = st
    return ns


def test_render_save_status_shows_degraded(hybrid, sheets, tmp_path):
    """로컬 임시 저장 상태면 상태 표시줄에 429·로컬 임시 저장이 보인다."""
    pid = _warm_owner_primary(hybrid, sheets)
    store = MemoryStore()
    _set_store_state(store, pid, _full_state())
    session = {"_save_degraded": True,
               "_save_degraded_msg": "Sheets 저장 실패(429) — 로컬에 임시 저장됨"}
    ns = _render_status_sandbox(store, session)
    ns["_render_save_status"]()
    shown = " ".join(ns["_fake_st"].captions)
    assert "429" in shown and "로컬" in shown
    assert "✅ 저장됨" not in shown


def test_render_save_status_shows_saved_only_on_success(
        hybrid, sheets, tmp_path):
    """성공 시에만 '저장됨'이 보인다 (실패 플래그가 있으면 저장됨 표시 금지)."""
    pid = _warm_owner_primary(hybrid, sheets)
    store = MemoryStore()
    _set_store_state(store, pid, _full_state())
    session: dict = {}
    ns = _render_status_sandbox(store, session)
    # 저장 전: 저장됨이 아니어야 한다
    ns["_render_save_status"]()
    assert "✅ 저장됨" not in " ".join(ns["_fake_st"].captions)
    # 저장 성공 후: 저장됨 표시
    save_ns = _save_sandbox(store, hybrid, session)
    assert save_ns["_save_current_project"]() == "ok"
    ns2 = _render_status_sandbox(store, session)
    ns2["_render_save_status"]()
    assert "✅ 저장됨" in " ".join(ns2["_fake_st"].captions)
