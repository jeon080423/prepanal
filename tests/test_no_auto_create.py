"""결함 3 재현·회귀 테스트 — 새로고침 후 빈 '새 프로젝트' 자동 생성 중단.

배경: 피주임 검증 보고 [554] — 새로고침 후 재로그인할 때마다 빈
'새 프로젝트'가 자동 생성·저장되어 프로젝트 목록에 쌓임.

원인 (수정 전 코드):
- HybridProjectStore.create_project()가 주(Sheets) create의 ValueError
  (중복 이름)를 일시 장애로 오인해 `except Exception`으로 삼키고 로컬
  폴백으로 같은 이름의 프로젝트를 새로 만든다. 주 목록 조회가 429로
  잠시 실패하면 상단 중복 검사도 시트 행을 못 보므로, 마지막 방어선인
  주의 ValueError마저 삼켜지는 순간 동명 행이 하나 더 생긴다.
- app.py `_fresh_after_verify` 블록은 코드 입력(재로그인) 때마다 무조건
  create_project("새 프로젝트")를 먼저 시도해, 세션 프로젝트가 유효해도
  새 행 만들기를 시도한다.

실제 Sheets에는 접속하지 않는다. FakeSheetsStore로 흉내 낸다.
비밀값 없음.
"""

import uuid
from datetime import datetime

import pytest

from core.hybrid_projects import HybridProjectStore
from core.projects import ProjectStore


class SheetsDownError(Exception):
    """gspread 429(APIError)를 흉내 낸 예외."""


class FakeSheetsStore:
    """SheetsProjectStore와 같은 인터페이스의 인메모리 가짜.

    fail_list_calls=N이면 list_projects()가 처음 N번 429처럼 실패한다
    (목록 조회 순간의 일시 장애를 재현). create_project()는 실제와 같이
    list_projects()로 중복을 검사한 뒤 ValueError를 던진다.
    """

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.fail_list_calls = 0
        self.create_calls = 0
        self.save_calls = 0

    def _check_list(self):
        if self.fail_list_calls > 0:
            self.fail_list_calls -= 1
            raise SheetsDownError(
                "APIError 429: Quota exceeded for quota metric "
                "'Read requests'")

    def list_projects(self):
        self._check_list()
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
        self.create_calls += 1
        pid = uuid.uuid4().hex[:12]
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.rows[pid] = {"name": clean, "created_at": now,
                          "updated_at": now, "state": {}}
        return pid

    def save_project(self, project_id, state):
        self._check_list()
        self.save_calls += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        if project_id not in self.rows:
            self.rows[project_id] = {"name": "새 프로젝트", "created_at": now,
                                     "updated_at": now, "state": {}}
        self.rows[project_id]["state"] = dict(state)
        self.rows[project_id]["updated_at"] = now

    def load_project(self, project_id):
        self._check_list()
        row = self.rows.get(project_id)
        return dict(row["state"]) if row else {}

    def get_meta(self, project_id):
        self._check_list()
        row = self.rows.get(project_id)
        if not row:
            return {}
        return {"id": project_id, "name": row["name"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]}

    def rename_project(self, project_id, new_name):
        clean = new_name.strip()
        if not clean:
            raise ValueError("프로젝트 이름을 입력해 주세요.")
        if any(p["id"] != project_id
               and p["name"].strip().casefold() == clean.casefold()
               for p in self.list_projects()):
            raise ValueError("같은 이름의 프로젝트가 있습니다. 다른 이름으로 입력해 주세요.")
        self._check_list()
        if project_id in self.rows:
            self.rows[project_id]["name"] = clean

    def delete_project(self, project_id):
        self._check_list()
        self.rows.pop(project_id, None)


def _seed_sheets_project(sheets, name, state=None, updated_at=None):
    """가짜 시트에 행을 직접 심는다 (create 경로를 타지 않음)."""
    pid = uuid.uuid4().hex[:12]
    now = updated_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    sheets.rows[pid] = {"name": name, "created_at": now,
                        "updated_at": now, "state": dict(state or {})}
    return pid


@pytest.fixture
def sheets_with_new_project():
    """시트에 저장된 내용이 있는 '새 프로젝트' 1건이 있는 상태."""
    fake = FakeSheetsStore()
    pid = _seed_sheets_project(
        fake, "새 프로젝트",
        state={"questions": ["Q1", "Q2"], "provider": "gemini"},
        updated_at="2026-11-28 10:00")
    fake.seeded_pid = pid
    return fake


@pytest.fixture
def hybrid(sheets_with_new_project, tmp_path):
    return HybridProjectStore(
        primary_factory=lambda: sheets_with_new_project,
        fallback=ProjectStore(tmp_path / "projects"))


def _names(store):
    return [p["name"] for p in store.list_projects()]


def _local_files(tmp_path):
    return list((tmp_path / "projects").glob("*.json"))


# ----------------------------------------------------------------------------
# 재현 1: Hybrid.create_project가 중복 ValueError를 삼키고 로컬에 동명 생성
# ----------------------------------------------------------------------------
def test_duplicate_create_raises_when_sheets_list_flaky(
        sheets_with_new_project, tmp_path):
    """결함 3 핵심 재현.

    시트에 '새 프로젝트'가 있는데 목록 조회가 1회만 429로 실패하면,
    수정 전 코드는 주의 ValueError(중복)를 일시 장애로 삼켜 로컬에
    같은 이름의 빈 프로젝트를 새로 만든다 → 목록에 동명 2건.
    수정 후에는 ValueError가 그대로 전파되고 동명 행이 생기지 않는다.
    """
    sheets_with_new_project.fail_list_calls = 1  # 상단 중복 검사용 목록만 실패
    store = HybridProjectStore(
        primary_factory=lambda: sheets_with_new_project,
        fallback=ProjectStore(tmp_path / "projects"))
    with pytest.raises(ValueError, match="같은 이름의 프로젝트가 있습니다"):
        store.create_project("새 프로젝트")
    assert _names(store).count("새 프로젝트") == 1
    assert _local_files(tmp_path) == []


def test_duplicate_create_raises_when_healthy(
        sheets_with_new_project, tmp_path):
    """건강한 상태에서도 중복 생성은 ValueError (기존 동작 유지)."""
    store = HybridProjectStore(
        primary_factory=lambda: sheets_with_new_project,
        fallback=ProjectStore(tmp_path / "projects"))
    with pytest.raises(ValueError, match="같은 이름의 프로젝트가 있습니다"):
        store.create_project("새 프로젝트")
    assert _names(store).count("새 프로젝트") == 1
    assert _local_files(tmp_path) == []


def test_duplicate_create_never_falls_back_on_repeated_flaky_logins(
        sheets_with_new_project, tmp_path):
    """재로그인 반복 시뮬레이션: 목록이 매번 1회씩 실패해도 동명 행은 안 쌓인다."""
    for _ in range(3):
        sheets_with_new_project.fail_list_calls = 1
        store = HybridProjectStore(
            primary_factory=lambda: sheets_with_new_project,
            fallback=ProjectStore(tmp_path / "projects"))
        with pytest.raises(ValueError):
            store.create_project("새 프로젝트")
    assert _names(store).count("새 프로젝트") == 1
    assert _local_files(tmp_path) == []
    assert len(sheets_with_new_project.rows) == 1


# ----------------------------------------------------------------------------
# 로그인/새로고침 선택 로직: app._resolve_project_after_login
# (a) 세션 project_id 유효 시 유지 (b) 없으면 최근 수정 프로젝트
# (c) 0개일 때만 최초 1개 생성 — 자동 생성 없음
# ----------------------------------------------------------------------------
def test_resolve_keeps_valid_session_project(sheets_with_new_project,
                                             tmp_path):
    import app as app_module
    store = HybridProjectStore(
        primary_factory=lambda: sheets_with_new_project,
        fallback=ProjectStore(tmp_path / "projects"))
    pid, create_first = app_module._resolve_project_after_login(
        store, sheets_with_new_project.seeded_pid)
    assert pid == sheets_with_new_project.seeded_pid
    assert create_first is False
    assert sheets_with_new_project.create_calls == 0
    assert _local_files(tmp_path) == []


def test_resolve_falls_back_to_most_recently_updated(tmp_path):
    import app as app_module
    sheets = FakeSheetsStore()
    _seed_sheets_project(sheets, "오래된 프로젝트",
                         updated_at="2026-11-27 09:00")
    recent_pid = _seed_sheets_project(sheets, "최근 프로젝트",
                                      updated_at="2026-11-28 10:00")
    store = HybridProjectStore(
        primary_factory=lambda: sheets,
        fallback=ProjectStore(tmp_path / "projects"))
    # 세션 ID 없음(새로고침 후) + 유효하지 않은 ID 둘 다 최근 프로젝트로 간다.
    for session_pid in (None, "없는-id"):
        pid, create_first = app_module._resolve_project_after_login(
            store, session_pid)
        assert pid == recent_pid
        assert create_first is False
    assert sheets.create_calls == 0


def test_resolve_creates_only_when_zero_projects(tmp_path):
    import app as app_module
    sheets = FakeSheetsStore()
    store = HybridProjectStore(
        primary_factory=lambda: sheets,
        fallback=ProjectStore(tmp_path / "projects"))
    pid, create_first = app_module._resolve_project_after_login(store, None)
    assert pid is None
    assert create_first is True
    # 최초 1개만 생성되고, 이후 resolve는 그 프로젝트를 유지한다.
    first_pid, is_new = app_module._create_first_project(store)
    assert is_new is True
    assert len(store.list_projects()) == 1
    pid2, create_first2 = app_module._resolve_project_after_login(
        store, first_pid)
    assert pid2 == first_pid
    assert create_first2 is False
    assert len(store.list_projects()) == 1


def test_create_first_project_reuses_on_race(sheets_with_new_project,
                                             tmp_path):
    """최초 생성 경합: create가 ValueError를 던지면 새로 만들지 않고 기존 행 재사용."""
    import app as app_module
    sheets_with_new_project.fail_list_calls = 1
    store = HybridProjectStore(
        primary_factory=lambda: sheets_with_new_project,
        fallback=ProjectStore(tmp_path / "projects"))
    pid, is_new = app_module._create_first_project(store)
    assert pid == sheets_with_new_project.seeded_pid
    assert is_new is False
    assert _names(store).count("새 프로젝트") == 1
    assert _local_files(tmp_path) == []


def test_create_first_project_reuses_name_match_not_most_recent(tmp_path):
    """C-1: 경합 재사용은 이름이 맞는 행만 고른다. 더 최근에 수정된
    다른 프로젝트가 있어도 그 프로젝트를 반환하지 않는다."""
    import app as app_module
    sheets = FakeSheetsStore()
    old_pid = _seed_sheets_project(
        sheets, "새 프로젝트",
        state={"questions": ["Q1", "Q2"], "provider": "gemini"},
        updated_at="2026-11-27 09:00")
    _seed_sheets_project(
        sheets, "X 프로젝트",
        state={"questions": ["Q9"], "provider": "gemini"},
        updated_at="2026-11-28 10:00")
    sheets.fail_list_calls = 1
    store = HybridProjectStore(
        primary_factory=lambda: sheets,
        fallback=ProjectStore(tmp_path / "projects"))
    pid, is_new = app_module._create_first_project(store)
    assert pid == old_pid
    assert is_new is False
    assert _names(store).count("새 프로젝트") == 1
    assert len(sheets.rows) == 2
    assert _local_files(tmp_path) == []


# ----------------------------------------------------------------------------
# 재사용 시 저장 내용 보존 (B-1): 빈 값 덮어쓰기 없음
# ----------------------------------------------------------------------------
def test_reuse_preserves_saved_content(sheets_with_new_project, tmp_path,
                                       monkeypatch):
    """기존 행 재사용 경로는 저장된 내용을 그대로 불러오고, 빈 상태로
    저장하지 않는다. _load_project_to_store() 사용."""
    import app as app_module
    from core.store import MemoryStore

    mem = MemoryStore()
    monkeypatch.setattr(app_module, "store", mem)
    store = HybridProjectStore(
        primary_factory=lambda: sheets_with_new_project,
        fallback=ProjectStore(tmp_path / "projects"))
    monkeypatch.setattr(app_module, "_project_store", store)

    class _FakeSt:
        def __init__(self):
            self.session_state = {}

    monkeypatch.setattr(app_module, "st", _FakeSt())
    app_module._load_project_to_store(sheets_with_new_project.seeded_pid)
    assert mem.get("questions") == ["Q1", "Q2"]
    assert mem.get("project_id") == sheets_with_new_project.seeded_pid
    assert sheets_with_new_project.save_calls == 0
    assert sheets_with_new_project.rows[
        sheets_with_new_project.seeded_pid]["state"]["questions"] == ["Q1", "Q2"]


def test_fresh_after_verify_repeated_logins_create_nothing(
        sheets_with_new_project, tmp_path, monkeypatch):
    """_fresh_after_verify 핸들러를 재로그인처럼 3회 반복 호출해도
    새 행이 생기지 않고, 저장된 내용이 그대로 불러와진다."""
    import app as app_module
    from core.store import MemoryStore

    mem = MemoryStore()
    monkeypatch.setattr(app_module, "store", mem)
    store = HybridProjectStore(
        primary_factory=lambda: sheets_with_new_project,
        fallback=ProjectStore(tmp_path / "projects"))
    monkeypatch.setattr(app_module, "_project_store", store)

    class _FakeSt:
        def __init__(self):
            self.session_state = {}

    fake_st = _FakeSt()
    monkeypatch.setattr(app_module, "st", fake_st)
    for _ in range(3):
        fake_st.session_state["_fresh_after_verify"] = True
        pid = app_module._handle_fresh_after_verify()
        assert pid == sheets_with_new_project.seeded_pid
        assert mem.get("questions") == ["Q1", "Q2"]
    assert len(store.list_projects()) == 1
    assert _names(store).count("새 프로젝트") == 1
    assert _local_files(tmp_path) == []
    assert sheets_with_new_project.save_calls == 0


# ----------------------------------------------------------------------------
# C-1 회귀: 2건 프로젝트 + 목록 1회 실패 시 다른 프로젝트 덮어쓰기 금지
# (REVIEW_repro_C1.py를 테스트화 — 수정 전에는 "X 프로젝트" id를 반환해
# 빈 초기화를 한 뒤 저장 때 Q9가 "새 문항"으로 덮어씌워졌다.)
# ----------------------------------------------------------------------------
def test_fresh_after_verify_race_two_projects_reuses_name_match(
        tmp_path, monkeypatch):
    """오래된 내용 있는 '새 프로젝트'(Q1·Q2) + 더 최근의 'X 프로젝트'(Q9)
    2건 + 목록 1회 실패 조건에서 핸들러 호출 후
    (a) 새 행이 생기지 않고 (b) X 프로젝트 id가 반환되지 않으며
    (c) 세션에 중복 행의 내용(Q1·Q2)이 불러와지고
    (d) 이후 저장해도 X 프로젝트의 저장 내용이 그대로 보존됨을 검증한다."""
    import app as app_module
    from core.store import MemoryStore

    sheets = FakeSheetsStore()
    old_pid = _seed_sheets_project(
        sheets, "새 프로젝트",
        state={"questions": ["Q1", "Q2"], "provider": "gemini"},
        updated_at="2026-11-27 09:00")
    x_pid = _seed_sheets_project(
        sheets, "X 프로젝트",
        state={"questions": ["Q9"], "provider": "gemini"},
        updated_at="2026-11-28 10:00")
    sheets.fail_list_calls = 1  # _resolve 안의 첫 목록 조회만 429로 실패

    mem = MemoryStore()
    monkeypatch.setattr(app_module, "store", mem)
    store = HybridProjectStore(
        primary_factory=lambda: sheets,
        fallback=ProjectStore(tmp_path / "projects"))
    monkeypatch.setattr(app_module, "_project_store", store)

    class _FakeSt:
        def __init__(self):
            self.session_state = {}

        def warning(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

        def success(self, *a, **k):
            pass

    monkeypatch.setattr(app_module, "st", _FakeSt())

    pid = app_module._handle_fresh_after_verify()
    # (b) X 프로젝트가 아니라 이름이 맞는 중복 행이다.
    assert pid == old_pid
    assert pid != x_pid
    # (c) 세션에는 중복 행의 저장 내용이 불러와진다 (빈 상태가 아님).
    assert mem.get("questions") == ["Q1", "Q2"]
    assert mem.get("project_id") == old_pid
    # (a) 새 행이 생기지 않는다.
    assert len(sheets.rows) == 2
    assert sheets.create_calls == 0
    assert _local_files(tmp_path) == []

    # 사용자가 문항 1개를 추가해 저장해도 X 프로젝트는 그대로다.
    mem.set("questions", ["새 문항"])
    assert app_module._save_current_project() == "ok"
    # (d) X 프로젝트의 저장 내용(Q9)이 보존된다.
    assert sheets.rows[x_pid]["state"].get("questions") == ["Q9"]
    assert sheets.rows[old_pid]["state"].get("questions") == ["새 문항"]


def test_fresh_after_verify_defers_when_list_stays_degraded(
        tmp_path, monkeypatch):
    """M-1: 목록 장애가 계속되면(429 지속) 세션 project_id가 없을 때
    '0개'로 단정해 최초 생성을 시도하지 않고 None을 반환하며,
    새 행을 만들지 않는다."""
    import app as app_module
    from core.store import MemoryStore

    sheets = FakeSheetsStore()
    _seed_sheets_project(
        sheets, "새 프로젝트",
        state={"questions": ["Q1"], "provider": "gemini"},
        updated_at="2026-11-27 09:00")
    sheets.fail_list_calls = 10  # _resolve·회복 확인 조회가 계속 실패

    mem = MemoryStore()
    monkeypatch.setattr(app_module, "store", mem)
    store = HybridProjectStore(
        primary_factory=lambda: sheets,
        fallback=ProjectStore(tmp_path / "projects"))
    monkeypatch.setattr(app_module, "_project_store", store)

    class _FakeSt:
        def __init__(self):
            self.session_state = {}
            self.warnings = []

        def warning(self, msg, *a, **k):
            self.warnings.append(msg)

        def error(self, *a, **k):
            pass

        def success(self, *a, **k):
            pass

    fake_st = _FakeSt()
    monkeypatch.setattr(app_module, "st", fake_st)

    assert app_module._handle_fresh_after_verify() is None
    assert mem.get("project_id") is None
    assert len(sheets.rows) == 1
    assert sheets.create_calls == 0
    assert _local_files(tmp_path) == []
    assert fake_st.warnings  # 보류 사유를 경고로 표시한다.
