"""프리서베이 스튜디오 (AI 합성응답자 기반 사전테스트)

설문 설계 → 페르소나 설계 → 분포 학습 → 가상 응답 생성 → 결과
5단계 마법사형 Streamlit 앱.

아키텍처: 이 파일은 렌더링만 담당한다. 비즈니스 로직은 core/ 패키지의
서비스(QuestionnaireService·LearningService·GenerationService·ExportService)가
처리하고, 상태 접근은 Store 인터페이스(SessionStore)를 통한다.
"""

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (APP_SUBTITLE, APP_TITLE, DEFAULT_PERSONA, DISCLAIMER,
                    METHODOLOGY_NOTE, SAMPLE_XLSX_PATH)
from core import (QUESTION_TYPE_REGISTRY, PROVIDER_REGISTRY, ExportService, GenerationService,
                  LearningService, PdfQuestionnaireService, QuestionnaireService, SessionStore,
                  auth_status, get_provider, get_question_type,
                  ProjectStore, PROJECT_STATE_KEYS, SheetsProjectStore)

STEPS = ["설문 설계", "페르소나 설계", "분포 학습", "가상 응답 생성", "결과"]

# ----------------------------------------------------------------------------
# 서비스 · 저장소 (UI는 이 진입점만 쓴다)
# ----------------------------------------------------------------------------
store = SessionStore()
q_service = QuestionnaireService()
learn_service = LearningService()
gen_service = GenerationService()
export_service = ExportService()

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1080px; padding-top: 2rem; }
    div[data-testid="stExpander"] { background: #ffffff; border-radius: 8px; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# 상태 초기화 (Store 경유 — 향후 DB 저장소로 교체 가능)
# ----------------------------------------------------------------------------
def init_state():
    store.set_default("step", 1)
    store.set_default("questions", [])
    store.set_default("persona_dims",
                      {k: dict(v) for k, v in DEFAULT_PERSONA.items()})
    store.set_default("transcripts", [])
    store.set_default("use_precision", False)
    store.set_default("learned", None)
    store.set_default("learned_file", None)
    store.set_default("results_df", None)
    store.set_default("gen_meta", None)
    store.set_default("provider", "gemini")
    store.set_default("model_override", "")
    store.set_default("gen_cache", {})
    store.set_default("project_id", None)


init_state()


# ----------------------------------------------------------------------------
# 액세스 코드 (사이드바 입력, 미입력 시 화면은 보이지만 생성 기능 제한)
# 코드는 Streamlit Secrets의 ACCESS_CODE에 설정 (레포에 하드코딩 금지)
# ----------------------------------------------------------------------------
def is_code_verified() -> bool:
    """액세스 코드가 확인됐는지 반환."""
    return bool(st.session_state.get("access_granted"))


def verify_access_code(code: str) -> tuple[bool, str]:
    """코드 검증. (성공 여부, 메시지) 반환."""
    expected = ""
    try:
        expected = st.secrets.get("ACCESS_CODE", "")
    except Exception:
        expected = ""
    if not expected:
        return False, "서버에 액세스 코드가 설정되지 않았습니다."
    if code == expected:
        st.session_state["access_granted"] = True
        return True, "코드가 확인되었습니다."
    return False, "코드가 올바르지 않습니다."


def goto(step: int):
    store.set("step", step)
    # 사이드바 radio 위젯 상태도 함께 동기화 (위젯 state가 우선되어 되돌아가는 버그 방지)
    st.session_state["sidebar_step"] = step
    st.rerun()

# ----------------------------------------------------------------------------
# 이론적 배경 (팝업 다이얼로그)
# ----------------------------------------------------------------------------
@st.dialog("📖 이론적 배경", width="large")
def theory_dialog():
    st.caption("합성응답자(Synthetic Respondents) 방법론의 이론적 기초")

    st.subheader("1. 합성응답자란?")
    st.write(
        "합성응답자(Synthetic Respondents)는 대규모 언어모델(LLM)을 이용해 "
        "특정 인구통계학적·심리적 프로필을 가진 가상 인물을 시뮬레이션하고, "
        "이들이 설문 문항에 어떻게 응답할지를 생성하는 방법론입니다. "
        "실제 사람을 대상으로 조사하기 전에 설문지의 품질을 검증하거나, "
        "연구 방향성을 탐색하는 **파일럿 테스트** 용도로 활용됩니다."
    )

    st.subheader("2. 이론적 기반")
    st.markdown(
        """
**① LLM as Human Simulators**

LLM은 방대한 인간 생성 텍스트로 학습되면서 인간의 응답 패턴, 편향, 맥락적 뉘앙스를
내재화합니다. 적절한 페르소나 프롬프팅을 통해 특정 집단의 응답 성향을 모사할 수 있다는 것이
최근 연구들의 공통된 발견입니다.

**② 분포 기반 샘플링 (Prolific 2026)**

본 도구는 Prolific 연구팀(Gordon et al., 2026 pre-print)의 방식을 따릅니다.
페르소나에게 강제 단일 응답을 요구하는 대신, 보기별 선택 **확률분포**를
LLM으로부터 추출한 뒤 샘플링합니다.
이 방식은 단일 응답 방식 대비 분포 오차를 35~48% 감소시키는 것으로 보고되었습니다.

**③ 고정밀 모드 (Stanford Generative Agents)**

인터뷰 트랜스크립트를 활용하는 고정밀 모드는 스탠포드 Generative Agents 연구에 기반합니다.
2시간 분량의 심층 인터뷰를 바탕으로 생성된 에이전트는
정규화 정확도 83%로 가장 높은 fidelity를 보였습니다.
"""
    )

    st.subheader("3. 본 도구의 방법론")
    st.markdown(
        """
본 도구는 **GPU 파인튜닝(모델 가중치 학습)을 하지 않습니다.** 대신 3단계 파이프라인으로 동작합니다.

**Step A. 분포 학습**
업로드된 실제 응답 데이터(Excel)에서 문항별 경험분포를 추출합니다.
이는 모집단의 응답 경향성에 대한 empirical prior 역할을 합니다.

**Step B. 페르소나 설계**
연령, 성별, 직업 등 인구통계 차원별 분포를 설정합니다.
각 가상 응답자는 이 분포에서 샘플링된 프로필을 부여받습니다.

**Step C. 확률분포 샘플링**
각 페르소나에 대해 LLM으로부터 보기별 선택 확률을 추출한 뒤,
이 분포에서 실제 응답을 샘플링합니다.
주관식 문항은 페르소나 프로필을 조건으로 직접 생성합니다.
"""
    )

    st.subheader("4. 사용 중인 합성 모델 (상세)")
    st.markdown(
        """
본 도구는 외부 상용 LLM API를 호출하여 합성 응답을 생성합니다.
모델 가중치를 직접 학습(파인튜닝)하지 않으며, 프롬프트 엔지니어링 + 확률분포 샘플링 방식입니다.
사용자는 4단계 'AI 선택'에서 제공자(Gemini/OpenAI)와 모델을 직접 고를 수 있으며,
선택한 모델이 실제 생성에 사용됩니다.

**① Gemini 모델 (제공: Google AI)**
- `gemini-2.5-flash-lite` (기본값): 2.5 패밀리 중 가장 빠르고 저렴. 대량 생성·파일럿 테스트에 적합. 무료 등급 지원.
- `gemini-2.5-flash`: 속도와 성능의 균형. 추론이 필요한 작업에 적합한 가성비 모델.
- `gemini-2.5-pro`: 2.5 패밀리 최고 성능. 복잡한 추론·코딩에 강함. 무료 등급 할당량이 적어 429가 날 수 있음.
- `gemini-3-flash-preview`: 차세대 Flash 모델. 무료 티어 제공. Preview 버전이라 스펙이 변경될 수 있음.
- 출처: Google AI 공식 문서 (ai.google.dev/gemini-api/docs/models)
- API 키: Streamlit Secrets의 `GEMINI_API_KEY` 사용 (또는 사용자 직접 입력)

**② OpenAI 모델 (제공: OpenAI)**
- `gpt-4o-mini` (기본값): 가장 저렴한 범용 모델. 대부분의 작업에 충분.
- `gpt-4o`: 범용 고성능 모델. 비전·코딩 포함 복잡한 작업에 적합.
- `gpt-4.1-mini`: 1M 컨텍스트. 성능과 비용의 균형.
- `gpt-4.1`: 1M 컨텍스트. 복잡한 추론·코딩·지시 이행에 강함.
- 출처: OpenAI 공식 문서 (platform.openai.com/docs/models)
- API 키: Streamlit Secrets의 `OPENAI_API_KEY` 사용 (또는 사용자 직접 입력, 유료)

**③ 생성 전략 (4종)**

| 전략 | 조건 | 동작 방식 |
|------|------|-----------|
| 학습분포 전략 | 실제 응답 Excel 업로드됨 | 업로드된 데이터의 문항별 경험분포에서 직접 샘플링 (LLM 미사용) |
| 분포추출 전략 | 업로드 없음 (기본) | LLM에 페르소나 프로필을 주고 보기별 선택 확률분포를 추출 → 샘플링 (Prolific Gordon et al., 2026 방식) |
| 정밀생성 전략 | 인터뷰 모드 | 분포추출 전략을 상속, 인터뷰 트랜스크립트 기반 정교한 페르소나 컨텍스트 사용 |
| 주관식 생성 | 주관식 문항 | LLM 배치 호출로 자유 텍스트 직접 생성 |

**④ 페르소나 시뮬레이션 원리**
1. 2단계에서 설정한 인구통계 분포(성별/연령대/지역/직업)에서 각 가상 응답자의 프로필을 샘플링
2. 4단계에서 각 프로필을 LLM 프롬프트에 포함 ("당신은 30대 여성 서울 거주 직장인입니다...")
3. 선택형 문항: LLM이 각 보기의 선택 확률을 반환 → 확률에 따라 샘플링
4. 주관식 문항: 프로필을 조건으로 자유 텍스트 직접 생성

**⑤ 토큰 소모 (참고)**
- 응답자 1명당 약 950 토큰 (입력 750 + 출력 200)
- 1,000명 생성 시 약 95만 토큰
- Gemini 무료 등급 일일 한도: 150만 토큰
"""
    )

    st.subheader("5. 한계점")
    st.warning(
        "합성응답자는 탐색적·방향성 리서치용이며 인간 실측을 대체할 수 없습니다. "
        "(Pew Research: 평균 12%p 오차)"
    )
    st.markdown(
        """
- **생태학적 타당도 한계**: LLM의 학습 데이터에 없는 새로운 현상이나 급변하는 여론은 정확히 모사하기 어렵습니다.
- **편향 증폭 위험**: 학습 데이터의 편향이 합성 응답에 그대로 반영될 수 있습니다.
- **과적합 주의**: 소수 실제 데이터에 과도하게 맞추면 일반화 성능이 떨어집니다.
- **용도 제한**: 본조사 대체가 아닌, 본조사 **전** 파일럿 검증용으로만 사용해야 합니다.
"""
    )

    st.subheader("6. 참고 문헌")
    st.markdown(
        """
- Gordon, A. et al. (Prolific, 2026). 합성응답자(synthetic respondents) 분포 추정 연구 (pre-print). 페르소나별 확률분포 추출 방식이 강제 단일 응답 대비 분포 오차 35~48% 감소. (Research Live, 2026-08-24 보도)
- Park et al. Stanford Generative Agents 연구 (정규화 정확도 83%).
- Pew Research Center. 합성응답자와 인간 실측 비교 (평균 12%p 오차).
- Gemini Pro: 1,000개 설문 공개 벤치마크에서 표준 LLM 중 최고 성능 (67%).
"""
    )


def current_model() -> str:
    override = (store.get("model_override") or "").strip()
    if override:
        return override
    provider = store.get("provider")
    selected = (store.get(f"model_select_{provider}") or "").strip()
    if selected:
        return selected
    return get_provider(provider).default_model


# ----------------------------------------------------------------------------
# 사이드바
# ----------------------------------------------------------------------------
def _metrix_ci_b64() -> str:
    """메트릭스 CI PNG를 base64로 인코딩 (사이드바 표시용)."""
    import base64
    from pathlib import Path
    p = Path(__file__).parent / "assets" / "metrix_ci.png"
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode("ascii")
    return ""


# 사이드바 전체를 CI 배경(검정)과 동일하게 + 텍스트 가시성 확보
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #000000 0%, #1C1C1E 60%, #2C2C2E 100%);
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #CCCCCC !important;
    }
    [data-testid="stSidebar"] input {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #444444 !important;
    }
    [data-testid="stSidebar"] button {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #555555 !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #3A3A3C !important;
        border-color: #6E6E73 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] div {
        color: #FFFFFF !important;
    }
    /* 선택된 라디오 버튼 강조색: 차콜 */
    [data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] > div:first-child {
        border-color: #8E8E93 !important;
        background-color: #48484A !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] > div:first-child > div {
        background-color: #D1D1D6 !important;
    }
    /* 포커스 강조색: 차콜 */
    [data-testid="stSidebar"] input:focus {
        border-color: #8E8E93 !important;
        box-shadow: 0 0 0 1px #8E8E93 !important;
    }
    [data-testid="stSidebar"] .stExpander {
        background-color: rgba(28, 28, 30, 0.7) !important;
        border: 1px solid #3A3A3C !important;
    }
    [data-testid="stSidebar"] details {
        background-color: rgba(28, 28, 30, 0.7) !important;
    }
    [data-testid="stSidebar"] details summary {
        background-color: transparent !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] details summary:hover {
        background-color: #2C2C2E !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #3A3A3C !important;
    }
    [data-testid="stSidebar"] details summary svg {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] input::placeholder {
        color: #888888 !important;
        opacity: 1 !important;
    }
    /* 사이드바 비밀번호 입력칸: 전체 흰색 (React Aria 구조 기준) */
    [data-testid="stSidebar"] [data-testid="stTextInputRootElement"],
    [data-testid="stSidebar"] [data-testid="stTextInputRootElement"] > div,
    [data-testid="stSidebar"] [data-testid="stTextInputRootElement"] div {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInputField"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #000000 !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInputField"]::placeholder {
        color: #999999 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInputField"]:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    /* 눈 아이콘 버튼: 흰 배경 + 검정 테두리 + 검정 아이콘 */
    [data-testid="stSidebar"] [data-testid="stTextInputRootElement"] button {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: 1px solid #000000 !important;
        border-radius: 4px !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInputRootElement"] button:hover {
        background-color: #F0F0F0 !important;
        background: #F0F0F0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInputRootElement"] button svg,
    [data-testid="stSidebar"] [data-testid="stTextInputRootElement"] button span,
    [data-testid="stSidebar"] [data-testid="stTextInputRootElement"] button div {
        fill: #000000 !important;
        color: #000000 !important;
        background-color: transparent !important;
        background: transparent !important;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border-color: #444444 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #FFFFFF !important;
    }
    /* 프로젝트 선택칸: 흰 배경 + 검정 텍스트/아이콘 */
    .st-key-project_selector [data-baseweb="select"] > div,
    .st-key-project_selector [role="combobox"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-color: #D1D1D6 !important;
    }
    .st-key-project_selector [data-baseweb="select"] *,
    .st-key-project_selector [role="combobox"] * {
        color: #000000 !important;
    }
    .st-key-project_selector input {
        background-color: #FFFFFF !important;
        border: none !important;
    }
    .st-key-project_selector button {
        background-color: #FFFFFF !important;
        border-left: 1px solid #D1D1D6 !important;
        color: #000000 !important;
    }
    .st-key-project_selector button:hover {
        background-color: #F0F0F0 !important;
    }
    .st-key-project_selector svg {
        color: #000000 !important;
        fill: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    # 메트릭스 CI (사이드바가 이미 검정이므로 로고만 표시)
    st.markdown(
        """
        <div style="padding: 12px 16px 4px 16px; text-align: center;">
            <img src="data:image/png;base64,{logo_b64}"
                 style="height: 36px; vertical-align: middle;" alt="MetriX CI">
        </div>
        """.format(logo_b64=_metrix_ci_b64()),
        unsafe_allow_html=True,
    )
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    # 액세스 코드 입력 (사이드바)
    if is_code_verified():
        st.success("🔓 액세스 코드 확인됨")
    else:
        with st.expander("🔒 액세스 코드 입력", expanded=True):
            code_input = st.text_input(
                "액세스 코드", type="password", key="sidebar_code_input",
                label_visibility="collapsed", placeholder="액세스 코드",
            )
            if st.button("코드 확인", key="sidebar_code_btn"):
                ok, msg = verify_access_code(code_input)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    st.divider()

    # 프로젝트 관리 (Google Sheets 우선, 실패 시 로컬 파일로 폴백)
    _storage_label = "💻 로컬 (리부트 시 삭제됨)"
    _project_store = ProjectStore()
    if SheetsProjectStore.is_configured():
        try:
            _sheets_store = SheetsProjectStore()
            # 실제 연결 테스트 (권한 문제 등을 조기에 감지)
            _sheets_store.list_projects()
            _project_store = _sheets_store
            _storage_label = "☁️ Google Sheets"
        except Exception as e:
            # Sheets 연결 실패 시 로컬로 폴백하고 경고 표시
            import streamlit as st
            st.warning(
                f"⚠️ Google Sheets 연결 실패로 로컬 저장소를 사용합니다. "
                f"({str(e)[:100]})"
            )

    def _save_current_project():
        pid = store.get("project_id")
        if not pid:
            return
        state = {k: store.get(k) for k in PROJECT_STATE_KEYS}
        # 모델 선택도 저장
        prov_id = store.get("provider")
        state["model_select"] = store.get(f"model_select_{prov_id}")
        _project_store.save_project(pid, state)

    def _load_project_to_store(pid: str):
        state = _project_store.load_project(pid)
        # 기존 상태 초기화 후 불러오기 (없는 키는 기본값)
        defaults = {
            "questions": [],
            "persona_dims": {k: dict(v) for k, v in DEFAULT_PERSONA.items()},
            "transcripts": [],
            "use_precision": False,
            "learned": None,
            "learned_file": None,
            "results_df": None,
            "gen_meta": None,
            "provider": "gemini",
        }
        for k in PROJECT_STATE_KEYS:
            v = state.get(k)
            store.set(k, v if v is not None else defaults.get(k))
        # 모델 선택 복원
        prov_id = state.get("provider", "gemini")
        store.set("provider", prov_id)
        if state.get("model_select"):
            store.set(f"model_select_{prov_id}", state["model_select"])
        store.set("project_id", pid)
        store.set("step", 1)
        # 결과 관련 캐시 초기화
        store.set("gen_cache", {})

    # 기존 프로젝트 재사용: 접속마다 빈 프로젝트를 자동으로 추가하지 않는다.
    projects = _project_store.list_projects()
    current_pid = store.get("project_id")
    project_ids = [p["id"] for p in projects]
    if projects and current_pid not in project_ids:
        current_pid = projects[0]["id"]
        _load_project_to_store(current_pid)
    elif not projects:
        current_pid = None
        store.set("project_id", None)
    st.divider()

    step_labels = [f"{i + 1}. {name}" for i, name in enumerate(STEPS)]
    selected = st.radio(
        "단계",
        options=list(range(1, 6)),
        format_func=lambda i: step_labels[i - 1],
        index=store.get("step") - 1,
        label_visibility="collapsed",
        key="sidebar_step",
    )
    if selected != store.get("step"):
        store.set("step", selected)
        st.rerun()

    st.divider()
    st.subheader("AI 설정")

    # AI 제공자 선택
    from core.providers import PROVIDER_REGISTRY
    provider_ids = list(PROVIDER_REGISTRY.keys())
    provider_labels = {pid: PROVIDER_REGISTRY[pid].label for pid in provider_ids}
    selected_pid = st.selectbox(
        "AI 제공자",
        options=provider_ids,
        format_func=lambda pid: provider_labels[pid],
        index=provider_ids.index(store.get("provider")) if store.get("provider") in provider_ids else 0,
        key="provider_select",
    )
    if selected_pid != store.get("provider"):
        store.set("provider", selected_pid)
        st.rerun()

    provider = get_provider(store.get("provider"))

    # 사용자 API 키 직접 입력
    st.caption("🔑 내 API 키 사용 (선택사항)")
    user_key = st.text_input(
        "API 키 직접 입력",
        type="password",
        placeholder=f"{provider.label} API 키를 입력하세요",
        key=f"custom_key_{store.get('provider')}",
        help="입력한 키는 브라우저 세션 메모리에만 보관되며, 서버에 저장되지 않습니다. 페이지를 새로고침하면 사라집니다.",
    )
    st.caption("⚠️ 입력한 API 키는 저장되지 않습니다. 브라우저를 닫거나 새로고침하면 다시 입력해야 합니다.")

    col_test, col_clear = st.columns(2)
    with col_test:
        if st.button("키 테스트", use_container_width=True):
            if not user_key.strip():
                st.warning("먼저 API 키를 입력하세요.")
            else:
                with st.spinner("키 유효성 확인 중..."):
                    ok_test, msg_test = provider.test_key(user_key)
                if ok_test:
                    provider.set_custom_key(user_key)
                    st.success(msg_test)
                    st.rerun()
                else:
                    st.error(msg_test)
    with col_clear:
        if provider.has_custom_key():
            if st.button("키 지우기", use_container_width=True):
                provider.clear_custom_key()
                st.rerun()

    ok, msg = auth_status(store.get("provider"))
    st.write(f"선택: **{provider.label}**")
    st.caption(f"모델: {current_model()}")
    if ok:
        st.success(msg)
    else:
        st.error(msg)

    n_q = len(store.get("questions"))
    n_t = len(store.get("transcripts"))
    st.divider()
    st.caption(f"문항 {n_q}개 · 페르소나 차원 {len(store.get('persona_dims'))}개")
    if store.get("learned"):
        st.caption(f"분포 학습됨 ({store.get('learned_file')})")
    if n_t:
        st.caption(f"인터뷰 트랜스크립트 {n_t}건")

    st.divider()
    if st.button("📖 이론적 배경", use_container_width=True):
        theory_dialog()


step = store.get("step")


# ----------------------------------------------------------------------------
step = store.get("step")


# ----------------------------------------------------------------------------
# Step 1: 설문 설계
# ----------------------------------------------------------------------------
def step1():
    st.header("1. 설문 설계")
    st.caption("가상 응답을 생성할 설문 문항을 입력하세요.")

    questions = store.get("questions")

    if not questions:
        with st.container(border=True):
            st.write("아직 문항이 없습니다. 샘플 설문으로 시작하거나 직접 추가하세요.")
            if st.button("샘플 설문 불러오기", type="primary"):
                store.set("questions", q_service.load_sample_questions())
                st.rerun()

    # 설문지 파일 자동 인식 (PDF/HWPX/DOC/DOCX)
    with st.expander("📄 설문지 파일에서 자동 인식 (PDF/HWPX/DOC/DOCX)", expanded=False):
        st.caption("설문지 파일을 업로드하면 AI가 문항·보기·분기 로직을 자동 파싱합니다.")
        doc_file = st.file_uploader(
            "파일 선택 (PDF, HWPX, DOC, DOCX)", type=["pdf", "hwpx", "doc", "docx"], key="pdf_upload")
        if doc_file is not None:
            if st.button("문항 자동 인식 시작", type="primary", key="pdf_parse"):
                with st.spinner("파일을 분석하고 있습니다..."):
                    try:
                        doc_text = PdfQuestionnaireService.extract_text_auto(
                            doc_file.read(), doc_file.name)
                        if not doc_text.strip():
                            st.error("파일에서 텍스트를 추출하지 못했습니다. 스캔 이미지 PDF는 지원하지 않습니다.")
                        else:
                            parsed = PdfQuestionnaireService.parse_questions(
                                doc_text, store.get("provider"))
                            if not parsed:
                                st.error("문항 파싱에 실패했습니다. 파일 내용을 확인해주세요.")
                            else:
                                # PDF 인식 결과로 기존 문항을 대체 (기본 문항 제거)
                                new_questions = []
                                for pq in parsed:
                                    pq["id"] = q_service.new_question_id(new_questions)
                                    new_questions.append(pq)
                                store.set("questions", new_questions)
                                n_branch = sum(1 for q in parsed if q.get("branching"))
                                st.success(f"{len(parsed)}개 문항을 인식했습니다. 기존 문항은 제거되었습니다. (분기 로직 {n_branch}건 포함)")
                                st.rerun()
                    except Exception as e:
                        st.error(f"처리 중 오류: {e}")

    for idx, q in enumerate(questions):
        with st.expander(q_service.question_summary(q), expanded=False):
            new_text = st.text_input("문항 내용", value=q["text"], key=f"qtext_{q['id']}")
            if new_text != q["text"]:
                q["text"] = new_text
                store.set("questions", questions)
            qtype = get_question_type(q["type"])
            if qtype.needs_options:
                opts_raw = st.text_area(
                    "보기 (한 줄에 하나씩)",
                    value="\n".join(q.get("options", [])),
                    key=f"qopts_{q['id']}",
                )
                q["options"] = [o.strip() for o in opts_raw.split("\n") if o.strip()]
                store.set("questions", questions)
            # 분기 로직 표시/편집
            branching = st.text_input(
                "분기 로직 (예: 문3에서 ① 응답 시 문5로 이동)",
                value=q.get("branching", ""),
                key=f"qbranch_{q['id']}",
            )
            if branching != q.get("branching", ""):
                q["branching"] = branching
                store.set("questions", questions)
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("위로", key=f"qup_{q['id']}", disabled=idx == 0):
                    questions[idx - 1], questions[idx] = questions[idx], questions[idx - 1]
                    store.set("questions", questions)
                    st.rerun()
            with c2:
                if st.button("아래로", key=f"qdown_{q['id']}", disabled=idx == len(questions) - 1):
                    questions[idx + 1], questions[idx] = questions[idx], questions[idx + 1]
                    store.set("questions", questions)
                    st.rerun()
            with c3:
                if st.button("삭제", key=f"qdel_{q['id']}"):
                    questions.pop(idx)
                    store.set("questions", questions)
                    st.rerun()

    st.subheader("문항 추가")
    # 문항 유형 선택을 폼 바깥에 두어 변경 시 즉시 리런되도록 함
    # (폼 안에서는 selectbox 변경이 리런을 유발하지 않아 보기 입력란이 갱신되지 않는 버그)
    qtype_name = st.selectbox(
        "문항 유형",
        options=list(QUESTION_TYPE_REGISTRY.keys()),
        format_func=lambda k: QUESTION_TYPE_REGISTRY[k].label,
        key="add_qtype",
    )
    with st.container(border=True):
        with st.form("add_question"):
            qtext = st.text_input("문항 내용")
            qopts = ""
            if get_question_type(qtype_name).needs_options:
                qopts = st.text_area("보기 (한 줄에 하나씩 입력)")
            qbranch = st.text_input("분기 로직 (선택사항)")
            submitted = st.form_submit_button("추가", type="primary")
        if submitted:
            opts = [o.strip() for o in qopts.split("\n") if o.strip()]
            qdict = {
                "id": q_service.new_question_id(questions),
                "text": qtext.strip(),
                "type": qtype_name,
                "options": opts,
                "branching": qbranch.strip(),
            }
            errs = q_service.validate_question(qdict)
            if errs:
                for e in errs:
                    st.error(e)
            else:
                questions.append(qdict)
                store.set("questions", questions)
                st.rerun()

    st.divider()
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button(
            "다음: 페르소나 설계 →",
            type="primary",
            disabled=not questions,
            on_click=lambda: goto(2),
        )
    if not questions:
        with col2:
            st.caption("문항을 1개 이상 추가해야 다음 단계로 이동할 수 있습니다.")


# ----------------------------------------------------------------------------
# Step 2: 페르소나 설계
# ----------------------------------------------------------------------------
def step2():
    st.header("2. 페르소나 설계")
    st.caption("가상 응답자의 인구통계 분포를 설정합니다. 항목별 비율의 합은 100%여야 합니다.")

    dims = store.get("persona_dims")
    valid = True

    for dim in list(dims.keys()):
        with st.container(border=True):
            h1, h2 = st.columns([5, 1])
            with h1:
                st.subheader(dim)
            with h2:
                if st.button("차원 삭제", key=f"dimdel_{dim}"):
                    del dims[dim]
                    store.set("persona_dims", dims)
                    st.rerun()
            cats = dims[dim]
            cols = st.columns(max(1, len(cats)))
            for i, cat in enumerate(list(cats.keys())):
                with cols[i % len(cols)]:
                    v = st.number_input(
                        cat, min_value=0.0, max_value=100.0, value=float(cats[cat]),
                        step=1.0, key=f"pct_{dim}_{cat}",
                    )
                    cats[cat] = v
            store.set("persona_dims", dims)
            total = sum(cats.values())
            if abs(total - 100.0) > 0.01:
                st.error(f"합계가 100%가 아닙니다 (현재 {total:.1f}%).")
                valid = False
            else:
                st.success("합계 100% 확인")
            c1, c2 = st.columns(2)
            with c1:
                new_cat = st.text_input("구분 추가", key=f"newcat_{dim}", placeholder="예: 70대")
            with c2:
                st.write("")
                if st.button("구분 추가 실행", key=f"addcat_{dim}") and new_cat.strip():
                    cats[new_cat.strip()] = 0.0
                    store.set("persona_dims", dims)
                    st.rerun()
            for cat in list(cats.keys()):
                if st.button(f"'{cat}' 삭제", key=f"catdel_{dim}_{cat}"):
                    del cats[cat]
                    store.set("persona_dims", dims)
                    st.rerun()

    with st.expander("새 인구통계 차원 추가"):
        new_dim = st.text_input("차원 이름", placeholder="예: 소득대")
        if st.button("차원 추가") and new_dim.strip():
            dims[new_dim.strip()] = {"구분1": 50.0, "구분2": 50.0}
            store.set("persona_dims", dims)
            st.rerun()

    st.subheader("고정밀 모드 (인터뷰 기반)")
    with st.container(border=True):
        st.caption(
            "스탠포드 Generative Agents 연구에서 정규화 정확도 83%로 가장 높았던 방식입니다. "
            "1인당 약 2시간의 인터뷰 트랜스크립트가 필요합니다. "
            "트랜스크립트를 업로드하면 생성 단계에서 고정밀 모드를 켤 수 있습니다."
        )
        files = st.file_uploader(
            "인터뷰 트랜스크립트 업로드 (.txt, .md)",
            type=["txt", "md"],
            accept_multiple_files=True,
            key="transcript_uploader",
        )
        if files:
            loaded = []
            for f in files:
                try:
                    text = f.read().decode("utf-8", errors="replace")
                except Exception:
                    text = ""
                loaded.append({"name": f.name, "text": text})
            store.set("transcripts", loaded)
            st.success(f"{len(loaded)}건 로드됨")
        elif store.get("transcripts"):
            st.info(f"{len(store.get('transcripts'))}건 로드됨")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.button("← 이전: 설문 설계", on_click=lambda: goto(1))
    with c2:
        st.button("다음: 분포 학습 →", type="primary", disabled=not valid, on_click=lambda: goto(3))


# ----------------------------------------------------------------------------
# Step 3: 분포 학습
# ----------------------------------------------------------------------------
def step3():
    st.header("3. 분포 학습 (선택)")
    st.caption("실제 응답 엑셀을 올리면 문항별 응답 분포를 학습합니다. 건너뛰면 페르소나 기반 생성만 합니다.")

    with st.expander("방법론 안내", expanded=True):
        st.markdown(METHODOLOGY_NOTE.replace("\n", "  \n"))

    uploaded = st.file_uploader("실제 응답 엑셀 업로드 (.xlsx)", type=["xlsx"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("샘플 데이터로 체험"):
            df = pd.read_excel(SAMPLE_XLSX_PATH)
            store.set("learned",
                      learn_service.learn_distributions(df, store.get("questions")))
            store.set("learned_file", "sample_real_data.xlsx (샘플)")
            st.rerun()
    with c2:
        if st.button("분포 학습 초기화"):
            store.set("learned", None)
            store.set("learned_file", None)
            st.rerun()

    if uploaded is not None:
        try:
            df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"엑셀 읽기 실패: {e}")
            df = None
        if df is not None:
            with st.spinner("분포 학습 중..."):
                learned = learn_service.learn_distributions(df, store.get("questions"))
            store.set("learned", learned)
            store.set("learned_file", uploaded.name)
            st.success(f"{len(learned)}개 문항 매칭됨")

    learned = store.get("learned")
    if learned:
        st.subheader(f"학습된 분포 ({store.get('learned_file')})")
        for q in store.get("questions"):
            info = learned.get(q["id"])
            with st.container(border=True):
                if not info:
                    st.write(f"{q_service.question_summary(q)} — 매칭된 컬럼 없음")
                    continue
                st.write(f"**{q_service.question_summary(q)}**  ·  n={info['n']}")
                if info["dist"]:
                    ddf = pd.DataFrame(
                        {"보기": list(info["dist"].keys()), "비율": list(info["dist"].values())}
                    )
                    fig = px.bar(ddf, x="보기", y="비율", text_auto=".1%")
                    fig.update_layout(height=280, margin=dict(t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("주관식 문항 — 분포 학습 대상 아님")
    else:
        st.info("학습된 분포가 없습니다. 생성 단계에서는 페르소나 + LLM 분포 추정 방식으로 생성합니다.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.button("← 이전: 페르소나 설계", on_click=lambda: goto(2))
    with c2:
        st.button("다음: 가상 응답 생성 →", type="primary", on_click=lambda: goto(4))


# ----------------------------------------------------------------------------
# Step 4: 가상 응답 생성
# ----------------------------------------------------------------------------
def step4():
    st.header("4. 가상 응답 생성")
    st.caption("AI로 가상 응답자를 생성합니다.")

    with st.container(border=True):
        st.subheader("AI 선택")
        provider = st.radio(
            "사용할 AI",
            options=list(PROVIDER_REGISTRY.keys()),
            format_func=lambda k: get_provider(k).label,
            horizontal=True,
            key="provider",
        )
        ok, msg = auth_status(provider)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

        # 모델 선택 (옵션)
        prov = get_provider(provider)
        model_options = prov.models or [
            {"id": prov.default_model, "label": prov.default_model,
             "description": "", "source": ""}
        ]
        model_ids = [m["id"] for m in model_options]
        # 기본값 인덱스
        try:
            default_idx = model_ids.index(prov.default_model)
        except ValueError:
            default_idx = 0
        selected_model_id = st.selectbox(
            "모델 선택",
            options=model_ids,
            format_func=lambda mid: next(
                (m["label"] for m in model_options if m["id"] == mid), mid),
            index=default_idx,
            key=f"model_select_{provider}",
        )
        # 선택된 모델의 설명·출처 표시
        selected_info = next(
            (m for m in model_options if m["id"] == selected_model_id), {})
        if selected_info.get("description"):
            st.info(f"📝 {selected_info['description']}")
        if selected_info.get("source"):
            st.caption(f"출처: {selected_info['source']}")
        # 직접 입력 (목록에 없는 모델용)
        with st.expander("모델 직접 입력", expanded=False):
            st.text_input(
                "모델 ID 직접 입력 (비워두면 위 선택 사용)",
                placeholder=prov.default_model,
                key="model_override",
            )

    with st.container(border=True):
        st.subheader("생성 방식")
        has_learned = bool(store.get("learned"))
        st.write(
            "- 학습된 분포가 있는 선택형 문항: 경험분포에서 직접 샘플링 (API 호출 없음)"
            if has_learned
            else "- 선택형 문항: Prolific 연구팀(Gordon et al., 2026) 방식 — LLM이 페르소나별 보기별 선택 확률분포를 추정하면 샘플링 (분포 오차 35~48% 감소 보고)"
        )
        st.write("- 주관식 문항: LLM 배치 생성")
        n_trans = len(store.get("transcripts"))
        use_precision = st.checkbox(
            "고정밀 모드 (인터뷰 기반)",
            value=store.get("use_precision") and n_trans > 0,
            disabled=n_trans == 0,
            help="스탠포드 Generative Agents 연구: 정규화 정확도 83%. 트랜스크립트 업로드 시 활성화.",
        )
        store.set("use_precision", use_precision and n_trans > 0)
        if n_trans == 0:
            st.caption("인터뷰 트랜스크립트를 업로드하면 활성화됩니다 (2단계에서 업로드).")

    n = st.number_input("생성할 응답자 수", min_value=1, max_value=200, value=30, step=1)

    can_run = ok and len(store.get("questions")) > 0
    code_ok = is_code_verified()
    if st.button("생성 시작", type="primary", disabled=not can_run):
        if not code_ok:
            st.error("🔒 액세스 코드를 입력해야 생성을 시작할 수 있습니다. (좌측 사이드바에서 입력)")
        elif not ok:
            st.error("AI 인증이 필요합니다.")
        else:
            progress = st.progress(0.0, text="생성 중...")
            try:
                df = gen_service.generate_responses(
                    questions=store.get("questions"),
                    persona_dims=store.get("persona_dims"),
                    n=int(n),
                    provider_id=provider,
                    model=current_model(),
                    learned=store.get("learned"),
                    progress_cb=lambda f: progress.progress(f, text=f"생성 중... {int(f * 100)}%"),
                    cache=store.get("gen_cache"),
                    transcripts=store.get("transcripts") if store.get("use_precision") else None,
                    use_precision=store.get("use_precision"),
                )
                store.set("results_df", df)
                prov = get_provider(provider)
                used_model = prov.last_used_model or current_model()
                auto_switched = used_model != current_model()
                store.set("gen_meta", {
                    "생성일시": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "AI": prov.label,
                    "모델": used_model,
                    "응답자 수": int(n),
                    "분포 학습": "사용" if has_learned else "미사용",
                    "고정밀 모드": "사용" if store.get("use_precision") else "미사용",
                })
                progress.progress(1.0, text="완료")
                st.success(f"{len(df)}건 생성 완료")
                if auto_switched:
                    st.warning(
                        f"⚠️ {current_model()}의 할당량이 소진되어 "
                        f"{used_model}(으)로 자동 전환되었습니다."
                    )
                # 생성 완료 후 프로젝트 자동 저장
                _save_current_project()
            except Exception as e:
                st.error(f"생성 실패: {e}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.button("← 이전: 분포 학습", on_click=lambda: goto(3))
    with c2:
        st.button(
            "다음: 결과 확인 →",
            type="primary",
            disabled=store.get("results_df") is None,
            on_click=lambda: goto(5),
        )


# ----------------------------------------------------------------------------
# Step 5: 결과
# ----------------------------------------------------------------------------
def step5():
    st.header("5. 결과")

    # 정직한 한계 고지 — 결과 탭 상단 고정 표시
    st.warning(DISCLAIMER)

    df = store.get("results_df")
    if df is None:
        st.info("아직 생성된 데이터가 없습니다. 4단계에서 가상 응답을 생성하세요.")
        if st.button("← 생성 단계로 이동", on_click=lambda: goto(4)):
            pass
        return

    meta = store.get("gen_meta") or {}
    with st.container(border=True):
        cols = st.columns(len(meta) or 1)
        for i, (k, v) in enumerate(meta.items()):
            cols[i].metric(k, v)

    st.subheader("데이터 미리보기")
    st.dataframe(df, use_container_width=True, height=400)

    # 파일럿 진단 리포트 (규칙 기반 — 실제 데이터 통계만 사용, 환각 없음)
    from core.diagnostics import diagnose
    st.subheader("파일럿 진단 리포트")
    st.caption("실제 응답 데이터에서 계산된 통계에만 근거합니다. 임계값을 넘지 않은 문제는 보고하지 않습니다.")
    findings = diagnose(df, store.get("questions") or [])
    if not findings:
        st.success("진단 결과: 문제가 발견되지 않았습니다. 설문지가 정상적으로 작동합니다.")
    else:
        for f in findings:
            severity_icon = "🔴" if f["심각도"] == "높음" else "🟡"
            with st.container(border=True):
                st.write(f"{severity_icon} **{f['문제']}**")
                st.write(f"문항: {f['문항']}")
                st.write(f"근거: {f['근거']}")
                st.info(f"💡 해결책: {f['해결책']}")
    # 진단 결과를 스토어에 저장 (엑셀 다운로드에서 사용)
    store.set("diagnostics", findings)

    # 생성 분포 vs 학습 분포 비교
    learned = store.get("learned") or {}
    comp_qs = [
        q for q in store.get("questions")
        if q["type"] in ("single", "likert") and q["id"] in learned and learned[q["id"]].get("dist")
    ]
    if comp_qs:
        st.subheader("분포 비교 (생성 vs 학습)")
        for q in comp_qs:
            with st.container(border=True):
                st.write(f"**{q_service.question_summary(q)}**")
                gen_counts = df[q["id"]].astype(str).value_counts(normalize=True)
                cats = list(learned[q["id"]]["dist"].keys())
                cdf = pd.DataFrame({
                    "보기": cats,
                    "학습 분포": [learned[q["id"]]["dist"][c] for c in cats],
                    "생성 분포": [float(gen_counts.get(c, 0)) for c in cats],
                })
                mdf = cdf.melt(id_vars="보기", var_name="구분", value_name="비율")
                fig = px.bar(mdf, x="보기", y="비율", color="구분", barmode="group", text_auto=".1%")
                fig.update_layout(height=300, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("다운로드")
    c1, c2, c3 = st.columns(3)
    with c1:
        xlsx = export_service.to_excel_bytes(
            df, meta=meta, diagnostics=store.get("diagnostics") or []
        )
        st.download_button(
            "엑셀 다운로드 (.xlsx)",
            data=xlsx,
            file_name="synthetic_responses.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        st.caption("방법론_안내 시트(한계 고지 포함)가 함께 저장됩니다.")
    with c2:
        st.download_button(
            "CSV 다운로드 (.csv)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="synthetic_responses.csv",
            mime="text/csv",
        )
    with c3:
        word_bytes = export_service.to_word_bytes(
            df,
            meta=meta,
            diagnostics=store.get("diagnostics") or [],
            questions=store.get("questions") or [],
        )
        st.download_button(
            "리포트 다운로드 (.docx)",
            data=word_bytes,
            file_name="pilot_diagnostic_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="secondary",
        )
        st.caption("진단 리포트 + 문항별 요약이 깔끔하게 정리됩니다.")

    st.divider()
    if st.button("← 이전: 가상 응답 생성", on_click=lambda: goto(4)):
        pass


# ----------------------------------------------------------------------------
def _render_project_panel():
    with st.container(border=True):
        st.subheader("📁 프로젝트")
        st.caption(f"저장소: {_storage_label}")
        if not SheetsProjectStore.is_configured():
            with st.expander("Google Sheets 연동 설정", expanded=False):
                st.markdown(SheetsProjectStore.setup_guide())

        if projects:
            proj_options = [p["id"] for p in projects]
            proj_names = {p["id"]: p["name"] for p in projects}
            selected_pid = st.selectbox(
                "프로젝트 선택",
                options=proj_options,
                format_func=lambda pid: proj_names.get(pid, pid),
                index=proj_options.index(current_pid),
                key="project_selector",
                label_visibility="collapsed",
            )
            if selected_pid != current_pid:
                _save_current_project()
                _load_project_to_store(selected_pid)
                st.rerun()

            with st.expander("➕ 새 프로젝트 추가", expanded=False):
                with st.form("new_project_form"):
                    new_project_name = st.text_input("프로젝트 이름")
                    create_submitted = st.form_submit_button(
                        "프로젝트 만들기", use_container_width=True
                    )
                if create_submitted:
                    try:
                        new_pid = _project_store.create_project(new_project_name)
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        _save_current_project()
                        _load_project_to_store(new_pid)
                        st.rerun()

            if st.button("💾 저장", use_container_width=True, key="btn_save_project"):
                _save_current_project()
                st.success("저장되었습니다.")

            # 이름 변경·삭제
            with st.expander("프로젝트 관리", expanded=False):
                cur_meta = _project_store.get_meta(selected_pid)
                new_name = st.text_input(
                    "프로젝트 이름",
                    value=cur_meta.get("name", ""),
                    key="project_rename_input",
                )
                if st.button("이름 변경", key="btn_rename_project"):
                    try:
                        _project_store.rename_project(selected_pid, new_name)
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        st.rerun()
                st.divider()
                if st.button("🗑️ 프로젝트 삭제", key="btn_delete_project", type="secondary"):
                    if len(projects) <= 1:
                        st.error("마지막 프로젝트는 삭제할 수 없습니다.")
                    else:
                        _project_store.delete_project(selected_pid)
                        remaining = [p for p in projects if p["id"] != selected_pid]
                        _load_project_to_store(remaining[0]["id"])
                        st.rerun()
        else:
            st.info("프로젝트가 없습니다. 이름을 입력해 첫 프로젝트를 만들어 주세요.")
            with st.form("first_project_form"):
                first_project_name = st.text_input("프로젝트 이름")
                first_create_submitted = st.form_submit_button(
                    "첫 프로젝트 만들기", use_container_width=True
                )
            if first_create_submitted:
                try:
                    first_pid = _project_store.create_project(first_project_name)
                except ValueError as e:
                    st.error(str(e))
                else:
                    _load_project_to_store(first_pid)
                    st.rerun()
            st.stop()



# 라우팅
# ----------------------------------------------------------------------------
main_col, project_col = st.columns([3, 1], gap="large")
with project_col:
    _render_project_panel()
with main_col:
    {"step1": step1, "step2": step2, "step3": step3, "step4": step4, "step5": step5}[
        f"step{step}"
    ]()
