"""프로젝트 관리 (core/projects.py)

여러 설문 프로젝트를 생성·저장·불러오기·수정·삭제한다.
각 프로젝트는 JSON 파일 하나로 저장되며, 설문 문항·페르소나·학습 분포·
생성 결과를 모두 포함한다.

저장 위치: <BASE_DIR>/projects/{project_id}.json
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import BASE_DIR

PROJECTS_DIR = BASE_DIR / "projects"

# 프로젝트에 저장하는 상태 키 목록
PROJECT_STATE_KEYS = [
    "questions",
    "persona_dims",
    "transcripts",
    "use_precision",
    "learned",
    "learned_file",
    "results_df",   # DataFrame → records로 변환
    "gen_meta",
    "provider",
]


def _json_safe(obj):
    """numpy 타입 등을 JSON 직렬화 가능하게 변환."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class ProjectStore:
    """파일 기반 프로젝트 저장소."""

    def __init__(self, projects_dir: Path | None = None):
        self.dir = Path(projects_dir) if projects_dir else PROJECTS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, project_id: str) -> Path:
        return self.dir / f"{project_id}.json"

    def list_projects(self) -> list[dict]:
        """메타데이터 목록 (updated_at 내림차순)."""
        projects = []
        for f in self.dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                meta = data.get("_meta", {})
                projects.append({
                    "id": f.stem,
                    "name": meta.get("name", f.stem),
                    "created_at": meta.get("created_at", ""),
                    "updated_at": meta.get("updated_at", ""),
                })
            except Exception:
                continue
        projects.sort(key=lambda p: p["updated_at"], reverse=True)
        return projects

    def create_project(self, name: str) -> str:
        """새 프로젝트 생성. 프로젝트 ID 반환."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("프로젝트 이름을 입력해 주세요.")
        if any(
            p["name"].strip().casefold() == clean_name.casefold()
            for p in self.list_projects()
        ):
            raise ValueError("같은 이름의 프로젝트가 있습니다. 다른 이름으로 입력해 주세요.")

        project_id = uuid.uuid4().hex[:12]
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        data = {
            "_meta": {
                "name": clean_name,
                "created_at": now,
                "updated_at": now,
            },
            "state": {},
        }
        self._path(project_id).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return project_id

    def save_project(self, project_id: str, state: dict) -> None:
        """현재 상태를 프로젝트에 저장."""
        path = self._path(project_id)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            data = {"_meta": {"name": "새 프로젝트",
                              "created_at": now, "updated_at": now},
                    "state": {}}
        # DataFrame → records 변환
        serializable = {}
        for k, v in state.items():
            if k == "results_df" and isinstance(v, pd.DataFrame):
                serializable[k] = {"__dataframe__": True,
                                   "records": v.to_dict("records")}
            else:
                serializable[k] = _json_safe(v)
        data["state"] = serializable
        data["_meta"]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_project(self, project_id: str) -> dict:
        """프로젝트 상태 불러오기. DataFrame 복원 포함."""
        path = self._path(project_id)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        state = data.get("state", {})
        # DataFrame 복원
        if isinstance(state.get("results_df"), dict):
            df_info = state["results_df"]
            if df_info.get("__dataframe__"):
                records = df_info.get("records", [])
                state["results_df"] = pd.DataFrame(records) if records else None
        return state

    def get_meta(self, project_id: str) -> dict:
        path = self._path(project_id)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("_meta", {})

    def rename_project(self, project_id: str, new_name: str) -> None:
        path = self._path(project_id)
        if not path.exists():
            return
        clean_name = new_name.strip()
        if not clean_name:
            raise ValueError("프로젝트 이름을 입력해 주세요.")
        if any(
            p["id"] != project_id and p["name"].strip().casefold() == clean_name.casefold()
            for p in self.list_projects()
        ):
            raise ValueError("같은 이름의 프로젝트가 있습니다. 다른 이름으로 입력해 주세요.")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_meta"]["name"] = clean_name
        data["_meta"]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete_project(self, project_id: str) -> None:
        path = self._path(project_id)
        if path.exists():
            path.unlink()
