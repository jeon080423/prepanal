"""Sheets + 로컬 하이브리드 프로젝트 저장소 (core/hybrid_projects.py)

배경 (저장소 플립 결함, 2026-11-28 조사 — MECHANISM_C.md 참조):
app.py가 매 Streamlit rerun마다 모듈 레벨에서 백엔드를 다시 고른다.
Sheets 프로브(list_projects)가 429를 던진 rerun에는 _project_store가
로컬(ProjectStore)로 떨어지고, 복구된 rerun에는 Sheets로 돌아간다.
로컬에 만든 프로젝트는 Sheets 목록에 없어 app.py:1137-1140이 세션의
project_id를 Sheets 목록 첫 항목으로 바꿔 버려, 사용자에게는 작업이
날아간 것으로 보인다(실제 파일/행은 남아 있음).

이 모듈은 그 플립을 없앤다. 두 백엔드를 하나로 묶어 목록은 합집합으로
보여주고, 각 프로젝트가 어느 백엔드에 속하는지(소유권)를 추적해
읽기·쓰기·삭제가 항상 올바른 백엔드로 간다.

보장 사항:
- list_projects()는 Sheets가 429/장애 중이어도 예외를 던지지 않고,
  최소 로컬 목록을 반환한다. 로컬 생성 프로젝트는 Sheets 복구 후에도
  목록에 그대로 남는다.
- 로컬 전용 프로젝트를 Sheets에 쓰지 않는다. SheetsProjectStore는
  모르는 ID를 저장하면 이름 "새 프로젝트"로 새 행을 append하므로,
  소유 백엔드를 모른 채 저장하는 일은 없어야 한다.
- 시트 기존 행을 수정·삭제하지 않으며, 자동 이관(마이그레이션)도 하지
  않는다. 삭제는 양쪽 모두 시도하고, 이름 변경은 존재하는 쪽에만 적용한다.
- 장애 중 편집한 Sheets 프로젝트는 로컬 안전본이 최신이므로, 복구 후에도
  로컬 본을 먼저 읽고 다음 저장 때 Sheets 행으로 되돌린다(동기화).
  그 전까지는 Sheets 프로젝트가 조용히 로컬 전용이 되지 않는다.
- Sheets 미설정(primary_factory=None)이면 로컬 저장소와 동일하게 동작한다.

주의(범위 밖):
- Sheets 장애 "중"에는 Sheets 전용 프로젝트를 읽을 수 없다(읽을 방법이
  없음). 장애가 끝나면 다시 보인다. 미저장 인메모리 편집은 플립 시
  덮어씌워질 수 있다 — 저장된 내용은 안전하다.
- 로컬 안전본의 이름 유지를 위해 마지막으로 본 합집합 메타(이름·소유권만,
  상태 데이터 없음)를 로컬에 캐시한다. 캐시 파일은 `*.json`이 아니므로
  ProjectStore의 목록 glob에 걸리지 않는다.
- Sheets 장애 중에 Sheets에만 있는 이름과 같은 이름으로 로컬에 만들면,
  복구 후 이름이 같은 프로젝트가 2개(ID 다름) 보일 수 있다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

# 마지막으로 본 합집합 메타 캐시 (이름·소유권만 저장, 상태 데이터 없음).
# ProjectStore.list_projects()가 *.json을 glob하므로 확장자를 붙이지 않는다.
_CACHE_FILENAME = ".hybrid-meta-cache"


class HybridProjectStore:
    """Sheets(주) + 로컬(폴백) 하이브리드 프로젝트 저장소.

    ProjectStore / SheetsProjectStore와 동일한 인터페이스
    (list/create/save/load/get_meta/rename/delete)를 제공한다.

    primary_factory: SheetsProjectStore 클래스처럼 호출하면 새 주 저장소를
        돌려주는 callable. None이면 로컬 전용으로 동작한다.
    fallback: 로컬 ProjectStore 인스턴스.
    """

    def __init__(self, primary_factory: Callable[[], object] | None = None,
                 fallback=None):
        self._primary_factory = primary_factory
        self._fallback = fallback
        self._primary = None          # 주 저장소 인스턴스 (지연 생성·캐시)
        self._owner: dict[str, str] = {}  # project_id -> "primary" | "local"
        self._last_known: dict[str, dict] = {}  # project_id -> 마지막 합집합 메타
        self._diverged: set[str] = set()  # 양쪽에 있고 로컬이 더 새로운 ID
        # M-1: 마지막으로 주(Sheets) 목록에서 본 적 있는 ID. 분 단위
        # 타임스탬프 동점으로는 diverged를 잡지 못하므로, 저장 시점에
        # "Sheets에 있던 프로젝트를 로컬에 저장했다"는 사실 자체로
        # diverged를 기록하는 데 쓴다. 캐시에 함께 저장해 rerun을 넘긴다.
        self._known_primary: set[str] = set()
        self._owners_live = False     # 이 인스턴스에서 주 목록을 생으로 본 적 있는지
        self.primary_available: bool | None = None  # 마지막 주 시도 결과
        self._cache_path: Path | None = None
        if fallback is not None and hasattr(fallback, "dir"):
            try:
                self._cache_path = Path(fallback.dir) / _CACHE_FILENAME
            except Exception:
                self._cache_path = None
        self._load_cache()

    def _load_cache(self) -> None:
        """이전 rerun이 남긴 메타 캐시를 읽어 소유권·이름을 복원한다."""
        if self._cache_path is None:
            return
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        for pid, meta in raw.items():
            if pid.startswith("__"):
                continue
            if not isinstance(meta, dict):
                continue
            owner = meta.get("owner")
            if owner in ("primary", "local"):
                self._owner[pid] = owner
            self._last_known[pid] = {
                "id": pid,
                "name": meta.get("name", pid),
                "created_at": meta.get("created_at", ""),
                "updated_at": meta.get("updated_at", ""),
            }
        cached_diverged = raw.get("__diverged__")
        if isinstance(cached_diverged, list):
            self._diverged = {pid for pid in cached_diverged
                              if isinstance(pid, str)}
        cached_known = raw.get("__known_primary__")
        if isinstance(cached_known, list):
            self._known_primary = {pid for pid in cached_known
                                  if isinstance(pid, str)}

    def _save_cache(self) -> None:
        """현재 _last_known + 소유권을 로컬에 저장한다(최선 노력)."""
        if self._cache_path is None:
            return
        try:
            payload = {
                pid: {**meta, "owner": self._owner.get(pid, "local")}
                for pid, meta in self._last_known.items()
            }
            payload["__diverged__"] = sorted(self._diverged)
            payload["__known_primary__"] = sorted(self._known_primary)
            self._cache_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------
    def _get_primary(self):
        """주 저장소 인스턴스 반환. 생성 실패 시 None."""
        if self._primary is None and self._primary_factory is not None:
            try:
                self._primary = self._primary_factory()
            except Exception:
                self._primary = None
        return self._primary

    def _drop_primary(self):
        """주 저장소 인스턴스 버리기 — 다음 시도에 새로 만든다."""
        self._primary = None

    def _primary_list(self) -> list[dict]:
        """주 목록 조회. 실패 시 예외를 그대로 던진다(호출부에서 처리)."""
        primary = self._get_primary()
        if primary is None:
            raise RuntimeError("주 저장소가 설정되지 않았습니다.")
        try:
            result = primary.list_projects()
        except Exception:
            self._drop_primary()
            raise
        self.primary_available = True
        return result

    def _refresh_owners(self) -> list[dict]:
        """합집합 목록을 구하고 소유권을 갱신한다. 주 실패 시 로컬 목록만 쓴다.

        주가 살아 있으면 생 목록이 절대 기준이다(사라진 ID의 소유권·캐시를
        함께 지운다). 주가 죽어 있으면 생 로컬 목록만으로 갱신하고,
        캐시에 남은 Sheets 메타는 이름·소유권 힌트로 유지한다.

        양쪽에 모두 있는 ID는 updated_at이 더 새로운 쪽이 소유자다
        (동점은 주 우선). 양쪽 저장소가 같은 앱 서버의 datetime.now()로
        찍으므로 문자열 비교가 유효하다. 로컬이 더 새로우면 diverged로
        기록해, 다음 저장 때 주(Sheets)로 되돌린다 — 그렇지 않으면
        Sheets 프로젝트가 조용히 로컬 전용이 돼 리부트 때 날아간다.
        """
        merged: dict[str, dict] = {}
        primary_metas: dict[str, dict] = {}
        local_metas: dict[str, dict] = {}
        primary_ok = False
        try:
            for p in self._primary_list():
                primary_metas[p["id"]] = p
            primary_ok = True
        except Exception:
            self.primary_available = False
        if self._fallback is not None:
            for p in self._fallback.list_projects():
                local_metas[p["id"]] = p
        # 합집합: 양쪽에 있으면 updated_at이 더 새로운 쪽의 메타를 쓴다.
        for pid, meta in primary_metas.items():
            merged[pid] = meta
        for pid, meta in local_metas.items():
            if pid not in merged:
                merged[pid] = meta
            elif meta["updated_at"] > merged[pid]["updated_at"]:
                merged[pid] = meta
        if primary_ok:
            # 생 데이터가 절대 기준: diverged·소유권을 통째로 재계산한다.
            # - 로컬이 분 단위 타임스탬프보다 확실히 새로우면 diverged.
            # - 동점이면 인과 기록(장애 중 안전본을 쓴 적 있음)이 있을 때만 diverged.
            # - 주가 더 새로우면 diverged 해제(로컬 본은 낡은 것이다).
            # M-1: 생 목록이 절대 기준이므로 주에 있던 ID 집합도 갱신한다.
            self._known_primary = set(primary_metas.keys())
            prev_diverged = set(self._diverged)
            self._diverged = {
                pid for pid, meta in local_metas.items()
                if pid in primary_metas
                and (meta["updated_at"] > primary_metas[pid]["updated_at"]
                     or (meta["updated_at"] == primary_metas[pid]["updated_at"]
                         and pid in prev_diverged))
            }
            for pid in merged:
                if pid in self._diverged or pid not in primary_metas:
                    self._owner[pid] = "local"
                else:
                    self._owner[pid] = "primary"
            self._owner = {pid: self._owner[pid] for pid in merged}
            self._last_known = dict(merged)
            self._owners_live = True
            self._save_cache()
        else:
            # 장애 중: 생긴 항목만 갱신하고, 없는 항목의 캐시는 유지한다.
            # (diverged 판정은 건드리지 않는다 — 로컬 안전본이 여전히 최신이다.)
            for pid, meta in merged.items():
                self._owner[pid] = "local"
                self._last_known[pid] = meta
        projects = list(merged.values())
        projects.sort(key=lambda p: p["updated_at"], reverse=True)
        return projects

    def _mark_diverged_if_known_primary(self, project_id: str) -> None:
        """M-1: Sheets에 있던 프로젝트를 로컬에 저장했으면 diverged 기록.

        분 단위 타임스탬프만으로는 Sheets 최종 저장과 장애 중 첫 로컬
        저장이 같은 분일 때 diverged를 잡지 못한다. 저장 시점에
        "주에 있던 프로젝트의 안전본을 로컬에 썼다"는 인과 자체를
        기록해야, 복구 후 _refresh_owners()의 동점 분기가 안전본을
        고아로 만들지 않는다. 로컬 전용 프로젝트(주에 없던 ID)는
        기록하지 않는다.
        """
        if (project_id in self._known_primary
                and project_id not in self._diverged):
            self._diverged.add(project_id)
            self._save_cache()

    def _owner_of(self, project_id: str) -> str | None:
        """소유 백엔드. 모르면 목록을 한 번 새로 읽어 확정한다.

        저장 전에 호출되므로, 이 인스턴스에서 생 목록으로 검증된 적이
        없으면(캐시에서 온 소유권일 수 있음) 한 번 새로 읽는다.
        캐시가 낡아 시트에서 지워진 행을 주로 저장해 "새 프로젝트" 행을
        append하는 일을 막는다.
        """
        if not self._owners_live:
            self._refresh_owners()
        return self._owner.get(project_id)

    @staticmethod
    def _dup_name(projects: list[dict], name: str,
                  exclude_id: str | None = None) -> bool:
        clean = name.strip().casefold()
        return any(
            p["id"] != exclude_id
            and p["name"].strip().casefold() == clean
            for p in projects
        )

    # ------------------------------------------------------------------
    # 공개 인터페이스 (ProjectStore / SheetsProjectStore와 동일)
    # ------------------------------------------------------------------
    def list_projects(self) -> list[dict]:
        """Sheets ∪ 로컬 합집합 (updated_at 내림차순). 주 장애 시에도 반환한다."""
        return self._refresh_owners()

    def create_project(self, name: str) -> str:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("프로젝트 이름을 입력해 주세요.")
        if self._dup_name(self.list_projects(), clean_name):
            raise ValueError("같은 이름의 프로젝트가 있습니다. 다른 이름으로 입력해 주세요.")

        primary = self._get_primary()
        if primary is not None:
            try:
                pid = primary.create_project(clean_name)
                self._owner[pid] = "primary"
                return pid
            except Exception:
                # 주 생성 실패(429 등) → 로컬에 만든다. 다음 목록 조회에서
                # 합집합에 포함되므로 복구 후에도 사라지지 않는다.
                self._drop_primary()
                self.primary_available = False
        if self._fallback is None:
            raise RuntimeError("프로젝트를 저장할 저장소가 없습니다.")
        pid = self._fallback.create_project(clean_name)
        self._owner[pid] = "local"
        return pid

    def save_project(self, project_id: str, state: dict) -> None:
        # 소유 백엔드를 먼저 확정한다. 모르는 ID를 주에 저장하면 시트에
        # "새 프로젝트" 행이 append되므로, 소유권 없는 저장은 금지한다.
        owner = self._owner_of(project_id)
        if owner == "primary":
            primary = self._get_primary()
            if primary is not None:
                try:
                    primary.save_project(project_id, state)
                    return
                except Exception:
                    # 주 저장 실패 → 로컬에 안전본을 남기고 소유권을 옮긴다.
                    # 다음 load는 로컬 본을 찾아낸다. 안전본을 썼다는 사실
                    # 자체를 diverged로 기록해 캐시에 남긴다(분 단위 동점
                    # 때도 복구 후 최신본을 읽기 위해).
                    self._drop_primary()
                    self.primary_available = False
                    self._diverged.add(project_id)
            if self._fallback is not None:
                self._fallback.save_project(project_id, state)
                self._owner[project_id] = "local"
                self._save_cache()  # 소유권·diverged 둘 다 캐시에 반영
                # 로컬에 없던 파일은 ProjectStore가 이름 "새 프로젝트"로
                # 만드므로, 알고 있던 실제 이름으로 되돌린다(실패해도 무시).
                real_name = self._last_known.get(project_id, {}).get("name")
                if real_name:
                    try:
                        self._fallback.rename_project(project_id, real_name)
                    except Exception:
                        pass
                return
            raise RuntimeError("프로젝트를 저장할 저장소가 없습니다.")
        if owner == "local":
            # 로컬 소유 프로젝트는 로컬에만 쓴다. 저장 실패 시 주로 쓰면
            # 시트에 이름 "새 프로젝트" 행이 append되는 오기입이 생기므로,
            # 실패를 그대로 던지고(세션 상태는 유지됨) 주로 넘기지 않는다.
            if self._fallback is not None:
                self._fallback.save_project(project_id, state)
                # M-1: 주에 있던 프로젝트의 로컬 저장 — 타임스탬프 동점
                # 여부와 무관하게 diverged로 기록한다.
                self._mark_diverged_if_known_primary(project_id)
                if project_id in self._diverged:
                    # diverged(양쪽 존재·로컬이 더 새로움): 주가 살아 있으면
                    # 같은 상태를 주 행에도 되돌려 소유권을 주로 되돌린다.
                    # 이렇게 해야 Sheets 프로젝트가 조용히 로컬 전용이 돼
                    # 리부트 때 날아가는 일이 없다.
                    primary = self._get_primary()
                    if primary is not None:
                        try:
                            primary.save_project(project_id, state)
                            self._owner[project_id] = "primary"
                            self._diverged.discard(project_id)
                            self._save_cache()
                        except Exception:
                            self._drop_primary()
                            self.primary_available = False
                return
            raise RuntimeError("프로젝트를 저장할 저장소가 없습니다.")
        # 소유권 불명(목록에 없던 새 ID): 주가 살아 있으면 주, 아니면 로컬.
        primary = self._get_primary()
        if primary is not None:
            try:
                primary.save_project(project_id, state)
                self._owner[project_id] = "primary"
                return
            except Exception:
                self._drop_primary()
                self.primary_available = False
        if self._fallback is None:
            raise RuntimeError("프로젝트를 저장할 저장소가 없습니다.")
        self._fallback.save_project(project_id, state)
        self._owner[project_id] = "local"
        # M-1: 소유권 불명 분기도 주에 있던 프로젝트면 diverged 기록 —
        # 장애 중 첫 안전본 저장이 이 분기로 오는 경우가 있다(로컬 사본이
        # 아직 없어 목록 합집합에 안 잡힐 때).
        self._mark_diverged_if_known_primary(project_id)

    def load_project(self, project_id: str) -> dict:
        # 소유 백엔드 우선, 비어 있으면 다른 쪽을 본다.
        # (양쪽 모두 {}를 돌려주는 "없음"과 "빈 상태"를 구분할 수 없어
        #  비어 있으면 폴백하는 것이 안전하다 — 결과는 어차피 {}이다.)
        owner = self._owner.get(project_id)
        order = (["primary", "local"] if owner != "local"
                 else ["local", "primary"])
        for side in order:
            try:
                if side == "primary":
                    primary = self._get_primary()
                    state = primary.load_project(project_id) if primary else {}
                else:
                    state = (self._fallback.load_project(project_id)
                             if self._fallback is not None else {})
            except Exception:
                if side == "primary":
                    self._drop_primary()
                    self.primary_available = False
                state = {}
            if state:
                return state
        return {}

    def get_meta(self, project_id: str) -> dict:
        owner = self._owner.get(project_id)
        order = (["primary", "local"] if owner != "local"
                 else ["local", "primary"])
        for side in order:
            try:
                if side == "primary":
                    primary = self._get_primary()
                    meta = primary.get_meta(project_id) if primary else {}
                else:
                    meta = (self._fallback.get_meta(project_id)
                            if self._fallback is not None else {})
            except Exception:
                if side == "primary":
                    self._drop_primary()
                    self.primary_available = False
                meta = {}
            if meta:
                return meta
        return {}

    def rename_project(self, project_id: str, new_name: str) -> None:
        clean_name = new_name.strip()
        if not clean_name:
            raise ValueError("프로젝트 이름을 입력해 주세요.")
        if self._dup_name(self.list_projects(), clean_name,
                          exclude_id=project_id):
            raise ValueError("같은 이름의 프로젝트가 있습니다. 다른 이름으로 입력해 주세요.")
        # 존재하는 쪽에만 적용한다(양쪽 delete/rename은 없는 ID에 no-op이지만,
        #  한쪽이 429로 죽어도 다른 쪽은 반드시 시도한다).
        errors: list[Exception] = []
        attempted = 0
        primary = self._get_primary()
        if primary is not None:
            attempted += 1
            try:
                primary.rename_project(project_id, clean_name)
            except Exception as e:  # noqa: BLE001 — 다른 쪽은 계속 시도
                self._drop_primary()
                self.primary_available = False
                errors.append(e)
        if self._fallback is not None:
            attempted += 1
            try:
                self._fallback.rename_project(project_id, clean_name)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        if attempted and len(errors) == attempted:
            raise errors[0]
        if project_id in self._last_known:
            self._last_known[project_id] = {
                **self._last_known[project_id], "name": clean_name}

    def delete_project(self, project_id: str) -> None:
        # 양쪽 모두 시도한다. 한쪽 실패는 다른 쪽 삭제를 막지 않는다.
        # 둘 다 실패했을 때만 예외를 던진다(시트 행이 남아 복구 후 다시
        # 보이는 것은 데이터 손실이 아니라 안전한 결과다).
        errors: list[Exception] = []
        attempted = 0
        primary = self._get_primary()
        if primary is not None:
            attempted += 1
            try:
                primary.delete_project(project_id)
            except Exception as e:  # noqa: BLE001
                self._drop_primary()
                self.primary_available = False
                errors.append(e)
        if self._fallback is not None:
            attempted += 1
            try:
                self._fallback.delete_project(project_id)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        self._owner.pop(project_id, None)
        self._last_known.pop(project_id, None)
        self._diverged.discard(project_id)
        self._save_cache()
        if attempted and len(errors) == attempted:
            raise errors[0]
