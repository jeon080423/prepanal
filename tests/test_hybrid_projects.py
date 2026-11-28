"""저장소 플립 수정(HybridProjectStore) 테스트 — 워커 C.

실제 Google Sheets에는 접속하지 않는다. Sheets 백엔드는 동작을 흉내 내는
FakeSheetsStore로 대체하고, 로컬 백엔드는 실제 ProjectStore(tmp 경로)를 쓴다.

시나리오: 429 → 로컬 폴백 → 프로젝트 생성 → Sheets 복구 후에도 로컬 생성
프로젝트가 목록에 그대로 남는지 검증한다 (MECHANISM_C.md의 재현 시나리오).
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

    fail=True면 모든 연산이 429처럼 실패한다. rows에 실제로 쓰인 내용을
    그대로 노출해 시트 오기입 여부를 검사할 수 있다.
    """

    def __init__(self):
        self.fail = False
        self.rows: dict[str, dict] = {}  # id -> {"name":..., "state":...}

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
        if any(p["name"].strip().casefold() == clean.casefold()
               for p in self.list_projects()):
            raise ValueError("같은 이름의 프로젝트가 있습니다. 다른 이름으로 입력해 주세요.")
        self._check()
        pid = uuid.uuid4().hex[:12]
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.rows[pid] = {"name": clean, "created_at": now,
                          "updated_at": now, "state": {}}
        return pid

    def save_project(self, project_id, state):
        self._check()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        if project_id not in self.rows:
            # 실제 SheetsProjectStore와 동일: 모르는 ID는 "새 프로젝트" 행 append
            self.rows[project_id] = {"name": "새 프로젝트", "created_at": now,
                                     "updated_at": now, "state": {}}
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
        clean = new_name.strip()
        if not clean:
            raise ValueError("프로젝트 이름을 입력해 주세요.")
        if any(p["id"] != project_id
               and p["name"].strip().casefold() == clean.casefold()
               for p in self.list_projects()):
            raise ValueError("같은 이름의 프로젝트가 있습니다. 다른 이름으로 입력해 주세요.")
        self._check()
        if project_id in self.rows:
            self.rows[project_id]["name"] = clean

    def delete_project(self, project_id):
        self._check()
        self.rows.pop(project_id, None)


@pytest.fixture
def sheets():
    """미리 프로젝트 1건이 있는 Sheets 가짜."""
    fake = FakeSheetsStore()
    pid = fake.create_project("기존 Sheets 프로젝트")
    fake.save_project(pid, {"questions": ["Q1"], "provider": "gemini"})
    return fake


@pytest.fixture
def hybrid(sheets, tmp_path):
    """매번 새 인스턴스 — Streamlit rerun마다 app.py 모듈이 처음부터
    다시 실행되는 것과 같은 조건이다."""
    return HybridProjectStore(primary_factory=lambda: sheets,
                              fallback=ProjectStore(tmp_path / "projects"))


def fresh_hybrid(sheets, tmp_path):
    """rerun 시뮬레이션: 소유권 맵이 비어 있는 새 하이브리드 인스턴스."""
    return HybridProjectStore(primary_factory=lambda: sheets,
                              fallback=ProjectStore(tmp_path / "projects"))


# ----------------------------------------------------------------------------
# 핵심 시나리오: 429 → 로컬 생성 → Sheets 복구
# ----------------------------------------------------------------------------
def test_flip_scenario_local_project_survives_recovery(sheets, tmp_path):
    # 1) 로그인 직후 Sheets 429
    sheets.fail = True
    store = fresh_hybrid(sheets, tmp_path)
    assert store.list_projects() == []  # 예외 없이 빈 목록(로컬은 비어 있음)
    assert store.primary_available is False

    # 2) 429 상태에서 프로젝트 생성 + 저장 → 로컬에만 간다
    local_pid = store.create_project("새 프로젝트")
    store.save_project(local_pid, {"questions": ["Q1", "Q2"]})
    assert local_pid not in sheets.rows  # 시트에 쓰이지 않음
    assert store.list_projects()[0]["id"] == local_pid

    # 3) 다음 rerun: Sheets 복구 (새 인스턴스 = 소유권 맵 초기화 상태)
    sheets.fail = False
    recovered = fresh_hybrid(sheets, tmp_path)
    projects = recovered.list_projects()
    ids = [p["id"] for p in projects]
    assert local_pid in ids, "복구 후 목록에서 로컬 생성 프로젝트가 사라짐(플립 재현)"
    assert len(projects) == 2  # Sheets 기존 1건 + 로컬 1건
    assert {p["name"] for p in projects} == {"기존 Sheets 프로젝트", "새 프로젝트"}

    # 4) 로컬 프로젝트 내용이 그대로 읽힌다
    assert recovered.load_project(local_pid) == {"questions": ["Q1", "Q2"]}
    # 5) 시트에는 로컬 프로젝트 행이 생기지 않았다(오기입 없음)
    assert local_pid not in sheets.rows
    assert len(sheets.rows) == 1


def test_flip_scenario_old_code_would_lose_it(sheets, tmp_path):
    """대조 실험: 구 방식(백엔드 통째 교체)이면 같은 시나리오에서
    로컬 프로젝트가 목록에서 사라짐을 보여준다."""
    sheets.fail = True
    local = ProjectStore(tmp_path / "projects")
    local_pid = local.create_project("새 프로젝트")
    # 복구 후 구 코드는 _project_store = SheetsProjectStore()로 갈아낀다.
    sheets.fail = False
    sheets_only_ids = [p["id"] for p in sheets.list_projects()]
    assert local_pid not in sheets_only_ids  # 구 방식이면 여기서 유실로 보임


def test_list_never_raises_while_sheets_down(sheets, tmp_path):
    sheets.fail = True
    store = fresh_hybrid(sheets, tmp_path)
    local_pid = store.create_project("로컬만 있는 프로젝트")
    # 429 중에도 목록 조회는 예외 없이 로컬 목록을 돌려준다
    assert [p["id"] for p in store.list_projects()] == [local_pid]


def test_local_save_never_writes_to_sheets(sheets, tmp_path):
    """소유권이 로컬인 프로젝트의 저장은 Sheets에 행을 만들지 않는다
    (SheetsProjectStore.save_project의 "새 프로젝트" append 함정 회피)."""
    sheets.fail = True
    store = fresh_hybrid(sheets, tmp_path)
    pid = store.create_project("로컬 프로젝트")
    sheets.fail = False
    store.save_project(pid, {"questions": ["Q9"]})
    store.save_project(pid, {"questions": ["Q9", "Q10"]})
    assert pid not in sheets.rows
    assert store.load_project(pid) == {"questions": ["Q9", "Q10"]}


def test_sheets_project_save_falls_back_to_local_on_429(sheets, tmp_path):
    """Sheets 소유 프로젝트를 편집 중 429가 나면 로컬에 안전본이 남고,
    복구 후에도 최신 상태가 읽힌다. 이름도 유지된다. 다음 저장 때는
    Sheets 행으로 되돌려진다(동기화)."""
    # 현실 흐름: 장애 전에 건강한 목록을 한 번 본 인스턴스가 있다.
    warm = fresh_hybrid(sheets, tmp_path)
    pid = warm.list_projects()[0]["id"]
    assert warm.load_project(pid) == {"questions": ["Q1"], "provider": "gemini"}

    # 장애 발생 + 새 rerun(새 인스턴스, 캐시는 디스크에 남아 있음)
    sheets.fail = True
    store = fresh_hybrid(sheets, tmp_path)
    store.save_project(pid, {"questions": ["Q1", "Q2", "Q3"]})
    # 로컬 안전본 존재 + 이름 보존
    assert store.load_project(pid) == {"questions": ["Q1", "Q2", "Q3"]}
    local_names = [p["name"] for p in store.list_projects()]
    assert local_names == ["기존 Sheets 프로젝트"]
    # 시트 원본은 건드리지 않았다
    assert sheets.rows[pid]["state"] == {"questions": ["Q1"],
                                         "provider": "gemini"}

    # 복구 후: 최신본(로컬 안전본)이 읽히고, 다음 저장 때 시트로 되돌아간다
    sheets.fail = False
    recovered = fresh_hybrid(sheets, tmp_path)
    assert recovered.load_project(pid) == {"questions": ["Q1", "Q2", "Q3"]}
    assert [p["name"] for p in recovered.list_projects()] == ["기존 Sheets 프로젝트"]
    recovered.save_project(pid, {"questions": ["Q1", "Q2", "Q3"]})
    assert sheets.rows[pid]["state"] == {"questions": ["Q1", "Q2", "Q3"]}
    # 동기화 뒤에는 다시 Sheets 소유로 돌아간다
    healed = fresh_hybrid(sheets, tmp_path)
    assert healed.load_project(pid) == {"questions": ["Q1", "Q2", "Q3"]}


def test_same_minute_outage_save_stays_diverged_after_recovery(
        sheets, tmp_path):
    """M-1 회귀: 로컬 사본이 이미 있는 Sheets 프로젝트를 장애 중에 편집하면
    첫 저장이 owner=='local' 분기로 가 diverged 기록이 안 된다. Sheets
    최종 저장과 장애 중 첫 로컬 저장이 같은 분이면 복구 후 diverged
    재계산이 동점을 놓쳐 안전본이 고아가 되고 낡은 시트 본이 읽힌다.
    저장 시점에 diverged를 기록해야 복구 후에도 최신본이 읽힌다."""
    # 1) 1차 장애 → 안전본 생성 → 복구·동기화 (로컬 사본이 디스크에 남음)
    warm = fresh_hybrid(sheets, tmp_path)
    pid = warm.list_projects()[0]["id"]
    sheets.fail = True
    store = fresh_hybrid(sheets, tmp_path)
    store.save_project(pid, {"questions": ["Q1", "Q2"]})
    sheets.fail = False
    recovered = fresh_hybrid(sheets, tmp_path)
    recovered.save_project(pid, {"questions": ["Q1", "Q2"]})  # 시트로 동기화
    assert sheets.rows[pid]["state"] == {"questions": ["Q1", "Q2"]}

    # 2) 2차 장애: 로컬 사본이 있는 상태에서 편집·저장
    sheets.fail = True
    outage = fresh_hybrid(sheets, tmp_path)
    outage.list_projects()  # 장애 중 목록 조회가 소유권을 로컬로 떨어뜨림
    outage.save_project(pid, {"questions": ["Q1", "Q2", "Q3"]})
    assert outage.load_project(pid) == {"questions": ["Q1", "Q2", "Q3"]}

    # 3) 복구 직전, 시트 최종 저장 시각을 로컬 저장 시각과 같은 분으로 맞춤
    #    (분 단위 타임스탬프 동점 조건을 결정적으로 재현)
    local_meta = [p for p in outage.list_projects() if p["id"] == pid][0]
    sheets.rows[pid]["updated_at"] = local_meta["updated_at"]

    # 4) 복구 후에도 최신본(로컬 안전본)이 읽혀야 한다
    sheets.fail = False
    healed = fresh_hybrid(sheets, tmp_path)
    healed.list_projects()
    assert healed.load_project(pid) == {"questions": ["Q1", "Q2", "Q3"]}
    healed.save_project(pid, {"questions": ["Q1", "Q2", "Q3"]})
    assert sheets.rows[pid]["state"] == {"questions": ["Q1", "Q2", "Q3"]}


# ----------------------------------------------------------------------------
# 회귀: 정상 경로
# ----------------------------------------------------------------------------
def test_normal_path_uses_sheets(sheets, tmp_path):
    store = fresh_hybrid(sheets, tmp_path)
    pid = store.create_project("정상 프로젝트")
    assert pid in sheets.rows  # Sheets에 생성됨
    assert list((tmp_path / "projects").glob("*.json")) == []  # 로컬은 비어 있음
    store.save_project(pid, {"questions": ["Q1"]})
    assert sheets.rows[pid]["state"] == {"questions": ["Q1"]}
    assert store.load_project(pid) == {"questions": ["Q1"]}
    store.rename_project(pid, "이름 변경")
    assert sheets.rows[pid]["name"] == "이름 변경"
    store.delete_project(pid)
    assert pid not in sheets.rows
    assert store.list_projects()[0]["name"] == "기존 Sheets 프로젝트"


def test_duplicate_name_checked_across_backends(sheets, tmp_path):
    sheets.fail = True
    store = fresh_hybrid(sheets, tmp_path)
    store.create_project("로컬 이름")
    sheets.fail = False
    with pytest.raises(ValueError):
        store.create_project("로컬 이름")  # 로컬에만 있어도 중복
    with pytest.raises(ValueError):
        store.create_project("기존 Sheets 프로젝트")  # Sheets에만 있어도 중복


def test_delete_removes_from_both_backends(sheets, tmp_path):
    store = fresh_hybrid(sheets, tmp_path)
    pid = sheets.list_projects()[0]["id"]
    # 양쪽에 같은 ID가 있는 상태(폴백 저장으로 로컬 본까지 생긴 경우)를 만든다
    sheets.fail = True
    store.save_project(pid, {"questions": ["QX"]})
    sheets.fail = False
    store.delete_project(pid)
    assert pid not in sheets.rows
    assert list((tmp_path / "projects").glob("*.json")) == []
    assert store.list_projects() == []


def test_local_only_mode_without_primary(tmp_path):
    """Sheets 미설정(primary_factory=None)이면 로컬 저장소와 동일하게 동작."""
    store = HybridProjectStore(primary_factory=None,
                               fallback=ProjectStore(tmp_path / "projects"))
    pid = store.create_project("로컬 전용")
    store.save_project(pid, {"questions": ["Q1"]})
    assert store.load_project(pid) == {"questions": ["Q1"]}
    assert [p["name"] for p in store.list_projects()] == ["로컬 전용"]
