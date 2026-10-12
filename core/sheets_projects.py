"""Google Sheets 기반 프로젝트 저장소 (core/sheets_projects.py)

Streamlit Cloud의 임시 파일시스템 문제를 해결하기 위해
프로젝트 데이터를 Google Sheets에 저장한다.

설정 (Streamlit Secrets):
- GOOGLE_SERVICE_ACCOUNT: 서비스 계정 JSON (문자열)
- PROJECTS_SHEET_ID: 스프레드시트 ID (없으면 자동 생성 시도)

스프레드시트 구조:
- "projects" 시트: id | name | created_at | updated_at | data_1 | data_2 | ...
  (data는 JSON을 40000자씩 분할 저장 — 셀 50000자 제한 대응)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import pandas as pd

from .projects import _json_safe

# 셀 문자 제한 (여유 있게 40000)
CHUNK_SIZE = 40000
HEADER = ["id", "name", "created_at", "updated_at"]


def _get_client():
    """AHPMASTER와 동일한 gcp_service_account secrets로 gspread 클라이언트 생성."""
    import streamlit as st
    from google.oauth2.service_account import Credentials

    if "gcp_service_account" not in st.secrets:
        raise RuntimeError(
            "gcp_service_account이 설정되지 않았습니다. "
            "AHPMASTER와 동일한 서비스 계정을 Streamlit Cloud Secrets에 등록하세요."
        )
    raw_auth = st.secrets.get("gcp_service_account", {})
    # dict(TOML) / JSON 문자열 / Base64 모두 지원 (AHPMASTER survey_manager와 동일)
    if isinstance(raw_auth, dict) or hasattr(raw_auth, "keys"):
        sa_info = dict(raw_auth)
    elif isinstance(raw_auth, str):
        import base64
        import re
        auth_str = raw_auth.strip().strip('"').strip("'")
        try:
            sa_info = json.loads(auth_str)
        except json.JSONDecodeError:
            clean_b64 = re.sub(r"\s+", "", auth_str)
            missing = len(clean_b64) % 4
            if missing:
                clean_b64 += "=" * (4 - missing)
            try:
                decoded = base64.b64decode(clean_b64)
            except Exception:
                decoded = base64.urlsafe_b64decode(clean_b64)
            sa_info = json.loads(decoded.decode("utf-8"))
    else:
        raise RuntimeError("gcp_service_account 형식을 인식할 수 없습니다.")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    import gspread
    return gspread.authorize(creds)


def _get_spreadsheet(client):
    """스프레드시트 열기 (없으면 생성)."""
    import streamlit as st
    sheet_id = st.secrets.get("PROJECTS_SHEET_ID", "")
    if sheet_id:
        return client.open_by_key(sheet_id)
    # 자동 생성
    sh = client.create("PrePanel Projects")
    return sh


def _get_worksheet(sh):
    """projects 시트 가져오기 (없으면 생성)."""
    try:
        ws = sh.worksheet("projects")
    except Exception:
        ws = sh.add_worksheet(title="projects", rows=100, cols=20)
        ws.append_row(HEADER)
    return ws


def _split_data(data_str: str) -> list[str]:
    return [data_str[i:i + CHUNK_SIZE]
            for i in range(0, len(data_str), CHUNK_SIZE)]


def _join_data(row: list[str], data_start_col: int) -> str:
    return "".join(cell for cell in row[data_start_col:] if cell)


class SheetsProjectStore:
    """Google Sheets 기반 프로젝트 저장소.

    FileProjectStore(ProjectStore)와 동일한 인터페이스.
    """

    def __init__(self):
        self._client = None
        self._sh = None
        self._ws = None

    def _ensure(self):
        if self._ws is None:
            self._client = _get_client()
            self._sh = _get_spreadsheet(self._client)
            self._ws = _get_worksheet(self._sh)
        return self._ws

    @staticmethod
    def is_configured() -> bool:
        """Google Sheets 연동 설정 여부 (AHPMASTER와 동일한 gcp_service_account)."""
        try:
            import streamlit as st
            return "gcp_service_account" in st.secrets
        except Exception:
            return False

    @staticmethod
    def setup_guide() -> str:
        return """**Google Sheets 연동 설정 방법:**

AHPMASTER와 동일한 서비스 계정을 사용합니다.

1. AHPMASTER Streamlit Cloud → Settings → Secrets에서
   `gcp_service_account` 값을 복사
2. PrePanel Streamlit Cloud → Settings → Secrets에 붙여넣기
3. 새 Google 스프레드시트를 만들고 서비스 계정 이메일과 공유 (편집자 권한)
4. Secrets에 `PROJECTS_SHEET_ID` 추가 (스프레드시트 URL의 `/d/` 다음 ID 부분)
"""

    def list_projects(self) -> list[dict]:
        ws = self._ensure()
        # get_all_records()는 헤더 중복 시 예외를 던지므로 get_all_values()로 직접 파싱
        values = ws.get_all_values()
        if not values:
            return []
        header = values[0]
        # 헤더가 올바르지 않으면 고침
        if header != HEADER:
            ws.update("A1", [HEADER])
            header = HEADER
        idx = {name: i for i, name in enumerate(header)}
        projects = []
        for row in values[1:]:
            pid = row[idx["id"]] if "id" in idx and idx["id"] < len(row) else ""
            if not pid:
                continue
            projects.append({
                "id": pid,
                "name": row[idx["name"]] if "name" in idx and idx["name"] < len(row) else pid,
                "created_at": row[idx["created_at"]] if "created_at" in idx and idx["created_at"] < len(row) else "",
                "updated_at": row[idx["updated_at"]] if "updated_at" in idx and idx["updated_at"] < len(row) else "",
            })
        projects.sort(key=lambda p: p["updated_at"], reverse=True)
        return projects

    def _find_row(self, project_id: str) -> int | None:
        """프로젝트 ID의 행 번호 (1-based). 헤더 제외."""
        ws = self._ensure()
        ids = ws.col_values(1)
        for i, v in enumerate(ids[1:], start=2):
            if v == project_id:
                return i
        return None

    def create_project(self, name: str) -> str:
        ws = self._ensure()
        project_id = uuid.uuid4().hex[:12]
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        ws.append_row([project_id, name.strip() or "새 프로젝트", now, now])
        return project_id

    def save_project(self, project_id: str, state: dict) -> None:
        ws = self._ensure()
        # DataFrame → records 변환
        serializable = {}
        for k, v in state.items():
            if k == "results_df" and isinstance(v, pd.DataFrame):
                serializable[k] = {"__dataframe__": True,
                                   "records": v.to_dict("records")}
            else:
                serializable[k] = _json_safe(v)
        data_str = json.dumps(serializable, ensure_ascii=False)
        chunks = _split_data(data_str)

        row_num = self._find_row(project_id)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        if row_num is None:
            # 새로 추가
            ws.append_row([project_id, "새 프로젝트", now, now] + chunks)
        else:
            # 기존 행 업데이트: A:D 메타 + E부터 데이터 청크
            # updated_at 갱신
            ws.update_cell(row_num, 4, now)
            # 기존 데이터 영역 지우기 (충분한 범위)
            ws.update(f"E{row_num}:Z{row_num}",
                      [[""] * 22], value_input_option="RAW")
            if chunks:
                end_col = chr(ord("E") + len(chunks) - 1)
                # Z를 넘어가면 확장 (실용적으로 22개면 880KB — 충분)
                ws.update(f"E{row_num}:{end_col}{row_num}",
                          [chunks], value_input_option="RAW")

    def load_project(self, project_id: str) -> dict:
        ws = self._ensure()
        row_num = self._find_row(project_id)
        if row_num is None:
            return {}
        row = ws.row_values(row_num)
        data_str = _join_data(row, 4)
        if not data_str.strip():
            return {}
        state = json.loads(data_str)
        if isinstance(state.get("results_df"), dict):
            df_info = state["results_df"]
            if df_info.get("__dataframe__"):
                records = df_info.get("records", [])
                state["results_df"] = pd.DataFrame(records) if records else None
        return state

    def get_meta(self, project_id: str) -> dict:
        ws = self._ensure()
        row_num = self._find_row(project_id)
        if row_num is None:
            return {}
        row = ws.row_values(row_num)
        return {
            "id": row[0] if len(row) > 0 else "",
            "name": row[1] if len(row) > 1 else "",
            "created_at": row[2] if len(row) > 2 else "",
            "updated_at": row[3] if len(row) > 3 else "",
        }

    def rename_project(self, project_id: str, new_name: str) -> None:
        ws = self._ensure()
        row_num = self._find_row(project_id)
        if row_num is None:
            return
        ws.update_cell(row_num, 2, new_name.strip())
        ws.update_cell(row_num, 4, datetime.now().strftime("%Y-%m-%d %H:%M"))

    def delete_project(self, project_id: str) -> None:
        ws = self._ensure()
        row_num = self._find_row(project_id)
        if row_num is not None:
            ws.delete_rows(row_num)
