"""프리서베이 스튜디오 (AI 합성응답자 기반 사전테스트)

설문 설계 → 페르소나 설계 → 분포 학습 → 가상 응답 생성 → 결과
5단계 마법사형 Streamlit 앱.

아키텍처: 이 파일은 렌더링만 담당한다. 비즈니스 로직은 core/ 패키지의
서비스(QuestionnaireService·LearningService·GenerationService·ExportService)가
처리하고, 상태 접근은 Store 인터페이스(SessionStore)를 통한다.
"""

from datetime import datetime
from pathlib import Path

import hashlib
import json

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from config import (AGE_BANDS, APP_TITLE, DEFAULT_PERSONA, DISCLAIMER,
                    METHODOLOGY_NOTE, NEMOTRON_ALL_FACETS, NEMOTRON_ATTRIBUTION,
                    NEMOTRON_EXTRACT_RELEASE_DATE, NEMOTRON_EXTRACT_SIZE,
                    NEMOTRON_EXTRACT_VERSION, NEMOTRON_FACETS,
                    NEMOTRON_FACET_LABELS, NEMOTRON_PROVINCES, PERSONA_SOURCES,
                    REGION_MAP, SAMPLE_XLSX_PATH)
from core import (QUESTION_TYPE_REGISTRY, PROVIDER_REGISTRY, ExportService, GenerationService,
                  LearningService, PdfQuestionnaireService, QuestionnaireService, SessionStore,
                  auth_status, get_provider, get_question_type,
                  ProjectStore, PROJECT_STATE_KEYS, SheetsProjectStore,
                  load_nemotron_extract, nemotron_default_quotas)

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
    [data-testid="stMain"] .block-container {
        max-width: none;
        padding: 2rem clamp(1rem, 2.5vw, 3rem);
    }
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
    store.set_default("page", "step1")
    store.set_default("questions", [])
    store.set_default("persona_dims",
                      {k: dict(v) for k, v in DEFAULT_PERSONA.items()})
    store.set_default("persona_source", "nemotron")
    store.set_default("nemotron_quotas", None)
    store.set_default("nemotron_quota_mode", "ratio")
    store.set_default("nemotron_count_quotas", None)
    store.set_default("nemotron_region_level", "region")
    store.set_default("nemotron_facets", None)
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
        # 코드 입력 직후에는 기존 작업이 아닌 빈 상태로 시작
        st.session_state["_fresh_after_verify"] = True
        return True, "코드가 확인되었습니다."
    return False, "코드가 올바르지 않습니다."


def goto(step: int):
    _autosave_current_project()
    store.set("step", step)
    store.set("page", f"step{step}")
    st.rerun()


def goto_page(page: str):
    """상단 메뉴별 페이지 이동 (페이지 상태 방식, 네이티브 멀티페이지 미사용).
    단계 페이지면 step도 함께 맞춘다.
    """
    _autosave_current_project()
    store.set("page", page)
    if page.startswith("step"):
        try:
            store.set("step", int(page[4:]))
        except ValueError:
            pass
    st.rerun()


# ----------------------------------------------------------------------------
# 저장 상태 추적 (UX 개선 ②·④: 저장 상태 지속 표시 + 미저장 소실 방지)
# ----------------------------------------------------------------------------
def _current_state_snapshot() -> dict:
    """현재 프로젝트 상태를 직렬화 가능한 스냅샷으로 반환."""
    state = {k: store.get(k) for k in PROJECT_STATE_KEYS}
    prov_id = store.get("provider")
    state["model_select"] = store.get(f"model_select_{prov_id}")
    return state


def _state_fingerprint(state: dict) -> str:
    """상태 스냅샷의 지문 (dirty 판정용). DataFrame은 내용 해시로 변환."""
    def _norm(v):
        if v is None:
            return None
        if isinstance(v, pd.DataFrame):
            try:
                digest = hashlib.md5(
                    v.to_csv(index=False).encode("utf-8")
                ).hexdigest()[:12]
            except Exception:
                digest = str(len(v))
            return {"__df__": [len(v), list(v.columns), digest]}
        try:
            return json.loads(json.dumps(v, default=str, ensure_ascii=False))
        except Exception:
            return str(v)
    normed = {k: _norm(v) for k, v in state.items()}
    raw = json.dumps(normed, default=str, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _autosave_current_project():
    """단계 이동 시 자동 저장. 저장 실패는 무시하고 이동을 계속한다."""
    try:
        save_fn = globals().get("_save_current_project")
        if save_fn is not None and is_code_verified() and store.get("project_id"):
            save_fn()
    except Exception:
        pass


def _is_current_state_saved() -> bool:
    """현재 상태가 마지막 저장(또는 불러오기)과 동일한지 반환."""
    pid = store.get("project_id")
    if not pid:
        return False
    if st.session_state.get("_last_saved_pid") != pid:
        return False
    saved_fp = st.session_state.get("_last_saved_fp")
    if not saved_fp:
        return False
    try:
        return _state_fingerprint(_current_state_snapshot()) == saved_fp
    except Exception:
        return False

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

    st.subheader("5. Nemotron 페르소나 출처 (Phase 1)")
    st.markdown(
        """
본 도구의 페르소나 출처 중 하나는 **Nemotron-Personas-Korea v1.0 (NVIDIA, CC BY 4.0)**입니다.
KOSIS·대법원·국민건강보험공단·농촌경제연구원·네이버 클라우드의 실제 통계를 씨앗으로 합성한
한국인 페르소나 100만 건 중, 계층추출한 **10,000건 발췌본**을 사용합니다.

- **분포 오차**: 셀 기준 최대 0.44%p (통과 기준 ±1%p 이내)
- **대표성의 근거**: 손으로 정한 가중치가 아니라 실측 분포 기반 샘플링(계층추출)이므로,
  패널 구성이 모집단을 닮았다는 근거를 제시할 수 있습니다.
- **주의**: 대표성이 좋아졌다고 응답 정확도가 좋아졌다는 뜻은 아닙니다.
  패널 구성이 모집단을 닮았다는 것과, 그 패널이 실제 사람처럼 응답한다는 것은 별개의 문제입니다.
"""
    )

    st.subheader("6. 응답 동질화·분산 붕괴 문제와 대응 방향")
    st.markdown(
        """
LLM 합성응답자의 가장 큰 약점은 **응답 동질화**입니다. 실제 사람들은 같은 질문에도
다양하게 응답하지만, LLM은 다수 의견(Modal Response) 하나로 과도하게 쏠리고
응답 분산이 붕괴되는 경향이 있습니다.

**근거 — Li 등 (2025, arXiv:2507.02919)**
ANES 실측과 비교한 극단적 동질화 비율(VR < 0.05)은 낙태 이슈에서 실제 40% 대
GPT-4·Llama 405B 80%, 이민 이슈에서 실제 30% 대 LLM 100%에 육박했습니다.
즉, LLM은 실제보다 훨씬 더 한쪽 의견으로 몰립니다.

**대응 방향 (분산 복원 고도화 진행 중)**
- **확률분포 샘플링**: 강제 단일 응답을 금지하고 보기별 선택 확률분포에서 샘플링
- **temperature 조절**: 생성 온도를 조정해 응답 다양성을 확보
- **반최빈값 보정**: 최빈값으로의 쏠림을 완화하는 보정
- **혼합 샘플링**: 여러 샘플링 방식을 혼합해 분산을 복원
"""
    )

    st.subheader("7. 검증 방식")
    st.markdown(
        """
**통계청 자료를 벤치마크**로 사용해 합성응답자의 품질을 재는 검증 체계를 구축 중입니다.
측정 지표는 다음과 같습니다.

- **평균 오차**: 합성 분포와 실측 분포의 문항별 평균 차이
- **SD 비율**: 합성 응답 표준편차 / 실측 표준편차 (1에 가까울수록 분산 복원 양호)
- **변이율(VR, Variation Ratio)**: 최빈값 외 응답 비율 — 동질화 정도를 재는 핵심 지표
- **JSD (Jensen-Shannon Divergence)**: 합성 분포와 실측 분포 사이의 거리
- **소수 의견 재현율**: 실측에 존재하는 소수 의견이 합성 응답에서도 나타나는지 여부
"""
    )

    st.subheader("8. 한계점")
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
- **성인만 포함**: 페르소나 발췌본은 19세 이상 성인만 포함하므로, 미성년 대상 조사의 파일럿에는 쓸 수 없습니다.
- **교차 특성 주의**: 직업 배정 시 성별·소득·학력을 독립으로 가정하므로, 교차 특성(예: 특정 성별·학력 조합의 직업 분포)이 실제와 다를 수 있습니다.
"""
    )

    st.subheader("9. 참고 문헌")
    st.markdown(
        """
- Gordon, A. et al. (Prolific, 2026). 합성응답자(synthetic respondents) 분포 추정 연구 (pre-print). 페르소나별 확률분포 추출 방식이 강제 단일 응답 대비 분포 오차 35~48% 감소. (Research Live, 2026-08-24 보도)
- Park et al. Stanford Generative Agents 연구 (정규화 정확도 83%).
- Pew Research Center. 합성응답자와 인간 실측 비교 (평균 12%p 오차).
- Gemini Pro: 1,000개 설문 공개 벤치마크에서 표준 LLM 중 최고 성능 (67%).
- Li, X. et al. (2025). LLM 합성응답자의 응답 동질화 연구 (arXiv:2507.02919). ANES 실측 대비 극단적 동질화 비율(VR < 0.05) — 낙태 이슈 실제 40% vs GPT-4·Llama 405B 80%, 이민 이슈 실제 30% vs LLM 100%에 육박.
"""
    )


def _render_update_log_page():
    """업데이트 기록 페이지 — UPDATE_LOG.md 내용을 별도 페이지로 보여준다.

    Streamlit Cloud 배포 환경에서도 레포 파일이 함께 배포되므로
    app.py와 같은 디렉터리의 UPDATE_LOG.md를 직접 읽는다.
    """
    st.header("업데이트 기록")
    st.caption("PrePanel 변경 이력")
    log_path = Path(__file__).parent / "UPDATE_LOG.md"
    try:
        text = log_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.info("업데이트 기록이 없습니다.")
        return
    except OSError:
        st.info("업데이트 기록을 불러오지 못했습니다.")
        return
    st.markdown(text)


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


# 사이드바 밝은 회색 배경(#E4F2ED, 테마 secondary 배경색) + 어두운 텍스트
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: #E4F2ED;
    }
    [data-testid="stSidebar"] * {
        color: #17242C !important;
    }
    /* 사이드바 알림 박스(success/warning/info/error)는 밝은 배경이라 글씨를 검정으로 */
    [data-testid="stSidebar"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stSidebar"] [data-testid="stAlert"] p {
        color: #1e293b !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #6B7280 !important;
    }
    [data-testid="stSidebar"] input {
        background-color: #FFFFFF !important;
        color: #17242C !important;
        border: 1px solid #D1D1D6 !important;
    }
    [data-testid="stSidebar"] button {
        background-color: #FFFFFF !important;
        color: #17242C !important;
        border: 1px solid #D1D1D6 !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #D2E8E0 !important;
        border-color: #9CA3AF !important;
    }
    /* 현재 프로젝트 버튼(primary): 차콜 배경 + 흰 텍스트 */
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
        background-color: #176F68 !important;
        color: #FFFFFF !important;
        border-color: #176F68 !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
        background-color: #115750 !important;
        border-color: #115750 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] div {
        color: #17242C !important;
    }
    /* 선택된 라디오 버튼 강조색: 밝은 파랑 */
    [data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] > div:first-child {
        border-color: #176F68 !important;
        background-color: #176F68 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] > div:first-child > div {
        background-color: #FFFFFF !important;
    }
    /* 포커스 강조색: 차콜 */
    [data-testid="stSidebar"] input:focus {
        border-color: #176F68 !important;
        box-shadow: 0 0 0 1px #176F68 !important;
    }
    [data-testid="stSidebar"] .stExpander {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D1D6 !important;
    }
    [data-testid="stSidebar"] details {
        background-color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] details summary {
        background-color: transparent !important;
        color: #17242C !important;
    }
    [data-testid="stSidebar"] details summary:hover {
        background-color: #D2E8E0 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #D1D1D6 !important;
    }
    [data-testid="stSidebar"] details summary svg {
        fill: #17242C !important;
        color: #17242C !important;
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
        color: #17242C !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #17242C !important;
        border-color: #D1D1D6 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #17242C !important;
    }
    /* AI 제공자 선택칸과 화살표: 흰 배경 + 검정 표시 */
    [data-testid="stSidebar"] .st-key-provider_select [data-baseweb="select"] > div,
    [data-testid="stSidebar"] .st-key-provider_select [role="combobox"],
    [data-testid="stSidebar"] .st-key-provider_select [role="combobox"] > div,
    [data-testid="stSidebar"] .st-key-provider_select button,
    [data-testid="stSidebar"] .st-key-provider_select button:hover,
    [data-testid="stSidebar"] .st-key-provider_select input {
        background: #FFFFFF !important;
        color: #000000 !important;
        border-color: #D1D1D6 !important;
    }
    [data-testid="stSidebar"] .st-key-provider_select [data-baseweb="select"] *,
    [data-testid="stSidebar"] .st-key-provider_select [role="combobox"] *,
    [data-testid="stSidebar"] .st-key-provider_select button * {
        color: #000000 !important;
    }
    [data-testid="stSidebar"] .st-key-provider_select svg {
        color: #000000 !important;
        fill: #000000 !important;
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
    /* 카드 내부 접이식 메뉴는 별도 카드 테두리 없이 표시 */
    .st-key-ai_settings_card [data-testid="stExpander"] details,
    .st-key-project_management_card [data-testid="stExpander"] details {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    .st-key-ai_settings_card [data-testid="stExpander"] details summary,
    .st-key-project_management_card [data-testid="stExpander"] details summary {
        padding-left: 0.25rem;
        padding-right: 0.25rem;
    }
    /* 단계 진행 메뉴: 완료(초록), 현재(파랑), 예정(회색) 삼각형 */
    [data-testid="stSidebar"] [class*="st-key-step_nav_"] button {
        background: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        justify-content: flex-start !important;
        min-height: 32px;
        padding: 4px 8px !important;
        gap: 10px;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [class*="st-key-step_nav_"] button::before {
        content: "▼";
        font-size: 22px;
        line-height: 1;
        flex-shrink: 0;
    }
    [data-testid="stSidebar"] [class*="st-key-step_nav_"] button > div {
        flex: 0 1 auto !important;
        width: auto !important;
        text-align: left !important;
        justify-content: flex-start !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] [class*="st-key-step_nav_"] button p {
        text-align: left !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #17242C !important;
    }
    [data-testid="stSidebar"] [class*="st-key-step_nav_complete_"] button::before {
        color: #39BCA7 !important;
    }
    [data-testid="stSidebar"] [class*="st-key-step_nav_current_"] button::before {
        color: #176F68 !important;
    }
    [data-testid="stSidebar"] [class*="st-key-step_nav_upcoming_"] button::before {
        color: #6B7280 !important;
    }
    [data-testid="stSidebar"] [class*="st-key-step_nav_current_"] button {
        background: rgba(57, 188, 167, 0.15) !important;
    }
    [data-testid="stSidebar"] [class*="st-key-step_nav_"] button:hover {
        background: rgba(0, 0, 0, 0.06) !important;
    }
    /* 결과 요약: 긴 날짜와 모델 이름도 생략 없이 표시 */
    .st-key-results_summary [data-testid="stMarkdownContainer"] p {
        font-size: 18px !important;
        line-height: 1.5 !important;
        white-space: normal !important;
        overflow: hidden !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# 레이아웃 개편 (2026-11-23, 이사님 결정)
# - 사이드바 제거 → 상단 헤더 + 1~5단계 진행 표시 + 5개 섹션을 한 페이지에 쌓기 (A안)
# - 첫 화면은 인덱스 랜딩페이지, 로그인하면 앱이 나오는 Qualtrics 방식
# - 색상: 딥틸·민트 톤 (primary #176F68, accent #39BCA7, 배경 #F8F9F6, 본문 #17242C, 테마는 .streamlit/config.toml 참조)
# 이 파일은 app.py에 합치는 조각(splice)용이다. 직접 실행하지 않는다.
# ----------------------------------------------------------------------------
# 프로젝트 저장소 설정 (기존 사이드바에서 이동)
# ----------------------------------------------------------------------------
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
        # 로그인 전 랜딩에서는 경고를 숨기고, 로그인 후에만 표시한다.
        if is_code_verified():
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
    # 저장 상태 기록 (지속 표시·dirty 판정용)
    st.session_state["_last_saved_pid"] = pid
    st.session_state["_last_saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state["_last_saved_fp"] = _state_fingerprint(state)


def _load_project_to_store(pid: str):
    state = _project_store.load_project(pid)
    # 기존 상태 초기화 후 불러오기 (없는 키는 기본값)
    defaults = {
        "questions": [],
        "persona_dims": {k: dict(v) for k, v in DEFAULT_PERSONA.items()},
        "persona_source": "nemotron",
        "nemotron_quotas": None,
        "nemotron_quota_mode": "ratio",
        "nemotron_count_quotas": None,
        "nemotron_region_level": "region",
        "nemotron_facets": None,
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
    store.set("page", "step1")
    # 결과 관련 캐시 초기화
    store.set("gen_cache", {})
    # 불러온 상태는 저장된 상태와 동일하므로 저장 지점을 기록
    st.session_state["_last_saved_pid"] = pid
    try:
        st.session_state["_last_saved_fp"] = _state_fingerprint(
            _current_state_snapshot()
        )
    except Exception:
        st.session_state["_last_saved_fp"] = ""
    try:
        st.session_state["_last_saved_at"] = (
            _project_store.get_meta(pid).get("updated_at", "")
        )
    except Exception:
        st.session_state["_last_saved_at"] = ""


# 기존 프로젝트 재사용: 접속마다 빈 프로젝트를 자동으로 추가하지 않는다.
# 단, 액세스 코드 입력 전/직후에는 기존 작업을 불러오지 않고 기본 페이지로 시작
projects = _project_store.list_projects()
current_pid = store.get("project_id")
project_ids = [p["id"] for p in projects]
if st.session_state.pop("_fresh_after_verify", False):
    # 코드 입력 직후: 빈 새 프로젝트를 생성해 기본 페이지로 시작
    # 같은 이름이 이미 있으면 기존 것을 재사용 (중복 생성 방지)
    try:
        new_pid = _project_store.create_project("새 프로젝트")
    except ValueError:
        existing = [p for p in projects if p.get("name") == "새 프로젝트"]
        new_pid = existing[0]["id"] if existing else project_ids[0] if project_ids else None
        if new_pid is None:
            # 프로젝트가 하나도 없으면 타임스탬프로 고유 이름 생성
            from datetime import datetime
            new_pid = _project_store.create_project(
                f"새 프로젝트 {datetime.now().strftime('%m%d-%H%M')}"
            )
    store.set("project_id", new_pid)
    store.set("questions", [])
    store.set("persona_dims", {k: dict(v) for k, v in DEFAULT_PERSONA.items()})
    store.set("persona_source", "nemotron")
    store.set("nemotron_quotas", None)
    store.set("nemotron_quota_mode", "ratio")
    store.set("nemotron_count_quotas", None)
    store.set("nemotron_region_level", "region")
    store.set("nemotron_facets", None)
    store.set("transcripts", [])
    store.set("use_precision", False)
    store.set("learned", None)
    store.set("learned_file", None)
    store.set("results_df", None)
    store.set("gen_meta", None)
    store.set("gen_cache", {})
    store.set("step", 1)
    store.set("page", "step1")
    projects = _project_store.list_projects()
    current_pid = new_pid
elif not is_code_verified():
    # 코드 입력 전: 기존 작업을 불러오지 않음
    current_pid = None
    store.set("project_id", None)
elif projects and current_pid not in project_ids:
    current_pid = projects[0]["id"]
    _load_project_to_store(current_pid)
elif not projects:
    current_pid = None
    store.set("project_id", None)


# 액세스 코드 입력란이 흰 배경에 묻히지 않도록 검정색을 명시한다.
st.markdown(
    """
    <style>
    div[data-testid="stTextInput"] input[type="password"] {
        background-color: white !important;
    }
    input[type="password"][placeholder="액세스 코드"] {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        caret-color: #000000 !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.dialog("PrePanel에 로그인")
def login_dialog():
    st.caption("액세스 코드를 입력하면 앱으로 이동합니다.")
    code_input = st.text_input(
        "액세스 코드", type="password", key="login_code_input",
        label_visibility="collapsed", placeholder="액세스 코드",
    )
    if st.button("로그인", type="primary", use_container_width=True, key="login_btn"):
        ok, msg = verify_access_code(code_input)
        if ok:
            st.rerun()
        else:
            st.error(msg)
    st.caption("액세스 코드는 관리자에게 문의하세요.")


def _report_mockup_html():
    """히어로 우측 파일럿 진단 리포트 화면 목업 — 순수 HTML (이미지 파일 아님).

    다크 틸 사이드바(실제 앱 5단계) + 리포트 카드 형태. Streamlit의
    st.markdown(unsafe_allow_html=True)에 그대로 넣는 것을 전제로 한다.
    검증되지 않은 수치·진단 문구는 일절 쓰지 않고, 예시로 보이는 부분에는
    '예시' 표시를 단다.
    """
    steps = [
        ("1", "설문 설계", "문항·보기 인식", True),
        ("2", "페르소나 설계", "분포 설정", True),
        ("3", "분포 학습", "분포 반영", True),
        ("4", "가상 응답 생성", "10,000건 생성", True),
        ("5", "결과", "진단 리포트", False),  # 현재 화면
    ]
    step_rows = []
    for num, name, sub, done in steps:
        if done:
            circle = (
                '<span style="flex-shrink:0; width:20px; height:20px; border-radius:50%; '
                'background:#39BCA7; color:#FFFFFF; font-size:11px; font-weight:800; '
                'display:inline-flex; align-items:center; justify-content:center;">✓</span>'
            )
            row_style = "display:flex; gap:8px; align-items:flex-start; padding:7px 8px; border-radius:8px;"
        else:
            circle = (
                '<span style="flex-shrink:0; width:20px; height:20px; border-radius:50%; '
                'background:#FFFFFF; color:#176F68; font-size:11px; font-weight:800; '
                'display:inline-flex; align-items:center; justify-content:center;">5</span>'
            )
            row_style = (
                "display:flex; gap:8px; align-items:flex-start; padding:7px 8px; "
                "border-radius:8px; background:rgba(255,255,255,0.14);"
            )
        step_rows.append(
            f'<div style="{row_style}">{circle}'
            f'<span><span style="display:block; font-size:11.5px; font-weight:700; color:#FFFFFF; '
            f'word-break:keep-all;">'
            f"{num}&nbsp; {name}</span>"
            f'<span class="pp-mockup-step-sub" style="display:block; font-size:10px; '
            f'color:#BFE3DA; margin-top:1px;">{sub}</span></span></div>'
        )
    sidebar = (
        '<div class="pp-mockup-side" style="width:158px; flex-shrink:0; background:#176F68; '
        'color:#FFFFFF; padding:14px 12px;">'
        '<div style="font-size:14px; font-weight:800;">PrePanel</div>'
        '<div style="font-size:10px; color:#BFE3DA; margin-top:1px;">by MetriX</div>'
        '<div style="font-size:10px; letter-spacing:0.08em; color:#BFE3DA; margin:14px 0 5px;">프로젝트</div>'
        '<div style="font-size:11.5px; font-weight:700; line-height:1.4;">신제품 만족도 조사 (v1) '
        '<span style="display:inline-block; font-size:9px; font-weight:700; '
        'border:1px solid rgba(255,255,255,0.6); border-radius:999px; padding:1px 6px; '
        'margin-left:2px; vertical-align:1px;">예시</span></div>'
        '<div style="margin-top:12px;">' + "".join(step_rows) + "</div></div>"
    )
    # 문항별 응답 분포 막대 — 숫자 없는 모양만 (예시 분포임을 표시)
    bar_heights = [16, 24, 40, 58, 74]
    bar_opacities = ["0.35", "0.5", "0.65", "0.85", "1"]
    bars = "".join(
        f'<div style="flex:1; height:{h}px; background:#39BCA7; opacity:{o}; '
        f'border-radius:4px 4px 0 0;"></div>'
        for h, o in zip(bar_heights, bar_opacities)
    )
    bar_labels = "".join(
        f'<div style="flex:1; text-align:center;">{i}점</div>' for i in range(1, 6)
    )
    main = (
        '<div class="pp-mockup-main" style="flex:1; min-width:0; background:#F8F9F6; padding:14px 16px;">'
        '<div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px;">'
        '<span style="font-size:13px; font-weight:800; color:#17242C;">← 파일럿 진단 리포트</span>'
        '<span style="display:flex; gap:6px; align-items:center; flex-shrink:0;">'
        '<span style="font-size:10px; font-weight:700; color:#176F68; background:#E4F2ED; '
        'border-radius:999px; padding:3px 8px;">예시 화면</span>'
        '<span style="font-size:11px; font-weight:700; color:#17242C; background:#FFFFFF; '
        'border:1px solid #D8E6DF; border-radius:8px; padding:5px 9px;">↓ 리포트 다운로드</span>'
        "</span></div>"
        '<div style="font-size:16px; font-weight:800; color:#17242C; margin-top:12px;">파일럿 진단 리포트</div>'
        '<div style="font-size:11.5px; color:#5A6E6B; margin-top:2px;">'
        "가상 응답 10,000건을 기반으로 설문 품질을 진단했습니다.</div>"
        '<div class="pp-mockup-cards" style="display:flex; gap:10px; margin-top:12px;">'
        '<div style="flex:1; background:#FFFFFF; border:1px solid #D8E6DF; border-radius:10px; padding:10px 12px;">'
        '<div style="font-size:11px; color:#5A6E6B;">가상 응답</div>'
        '<div style="font-size:19px; font-weight:800; color:#176F68; margin-top:2px; white-space:nowrap;">10,000건</div></div>'
        '<div style="flex:1; background:#FFFFFF; border:1px solid #D8E6DF; border-radius:10px; padding:10px 12px;">'
        '<div style="font-size:11px; color:#5A6E6B;">진단 단계</div>'
        '<div style="font-size:19px; font-weight:800; color:#176F68; margin-top:2px; white-space:nowrap;">5단계</div></div>'
        "</div>"
        '<div class="pp-mockup-tabs" style="display:flex; gap:14px; margin-top:14px; border-bottom:1px solid #D8E6DF; '
        'padding-bottom:7px; font-size:11.5px; white-space:nowrap; overflow:hidden;">'
        '<span style="font-weight:800; color:#176F68; border-bottom:2px solid #176F68; '
        'padding-bottom:6px; margin-bottom:-8px;">주요 진단 결과</span>'
        '<span style="color:#5A6E6B;">문항별 상세 분석</span>'
        '<span style="color:#5A6E6B;">응답자 프로필</span>'
        '<span style="color:#5A6E6B;">개선 제안</span></div>'
        '<div style="background:#FFFFFF; border:1px solid #D8E6DF; border-radius:10px; '
        'padding:10px 12px; margin-top:12px;">'
        '<div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">'
        '<span style="font-size:10px; font-weight:800; color:#C62828; background:#FDECEC; '
        'border-radius:6px; padding:2px 6px;">Q07</span>'
        '<span style="font-size:11.5px; font-weight:700; color:#17242C;">응답 편향 가능성</span>'
        '<span style="font-size:9px; color:#176F68; background:#E4F2ED; border-radius:999px; '
        'padding:2px 6px;">예시</span></div>'
        '<div style="font-size:11px; color:#5A6E6B; line-height:1.6; margin-top:6px;">'
        "Q07. '해당 제품을 다른 사람에게 추천할 의향이 있다.' — "
        "긍정 응답(4~5점)으로 치우친 분포입니다. 유사 문항과 중복될 가능성이 있습니다.</div>"
        "</div>"
        '<div class="pp-mockup-panels" style="display:flex; flex-direction:column; gap:10px; margin-top:10px;">'
        '<div style="flex:1.15; min-width:0; background:#FFFFFF; border:1px solid #D8E6DF; '
        'border-radius:10px; padding:10px 12px;">'
        '<div style="font-size:11.5px; font-weight:700; color:#17242C;">Q07 응답 분포 '
        '<span style="font-size:9px; color:#176F68; background:#E4F2ED; border-radius:999px; '
        'padding:2px 6px;">예시</span></div>'
        f'<div style="display:flex; align-items:flex-end; gap:8px; height:74px; margin-top:10px;">{bars}</div>'
        f'<div style="display:flex; gap:8px; font-size:10px; color:#5A6E6B; margin-top:4px; white-space:nowrap;">{bar_labels}</div>'
        "</div>"
        '<div style="flex:1; min-width:0; background:#FFFFFF; border:1px solid #D8E6DF; '
        'border-radius:10px; padding:10px 12px;">'
        '<div style="font-size:11.5px; font-weight:700; color:#17242C;">응답자 프로필</div>'
        '<div style="font-size:11px; color:#17242C; line-height:1.9; margin-top:6px;">'
        "✓ 가상 페르소나 10,000명<br>✓ 17개 시도 지역 분포 반영<br>"
        "✓ Nemotron-Personas-Korea 기반</div></div>"
        "</div></div>"
    )
    return (
        '<div style="display:flex; border:1px solid #D8E6DF; border-radius:16px; overflow:hidden; '
        'background:#FFFFFF; box-shadow:0 14px 34px rgba(23,36,44,0.12); text-align:left;">'
        + sidebar + main + "</div>"
    )


def _render_landing():
    """첫 화면 — 인덱스 랜딩페이지 (Qualtrics 방식). 로그인 전에는 앱 대신 이 화면을 보여준다.

    2026-11-24 지서방 1차 시안 방향: 밝은 배경(#F8F9F6)·딥틸(#176F68)·민트(#39BCA7).
    상단 '무료로 시작하기' 버튼 없음(로그인만). 히어로는 좌 텍스트 + 우 파일럿 진단
    리포트 HTML 목업. 수치(92%·3분·54%)는 검증되지 않아 쓰지 않는다.
    """
    # 모바일(≤768px)에서는 가로 컬럼을 세로로 쌓는다 — 랜딩에만 적용되는 스타일
    # (_render_landing이 실행될 때만 페이지에 있으므로 로그인 후 앱에는 영향 없음)
    st.markdown(
        """<style>
        @media (max-width: 768px) {
          [data-testid="stHorizontalBlock"] { flex-direction: column !important; align-items: stretch !important; }
          [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important; }
          .pp-stats { flex-wrap: wrap !important; }
          .pp-stats > div { flex: 1 1 45% !important; padding: 0.5rem 0 !important; }
        }
        @media (max-width: 560px) {
          .pp-mockup-side { width: 118px !important; padding: 10px 8px !important; }
          .pp-mockup-step-sub { display: none !important; }
          .pp-mockup-main { padding: 10px 12px !important; }
          .pp-mockup-cards { flex-direction: column !important; }
          .pp-mockup-panels { flex-direction: column !important; }
          .pp-mockup-tabs { flex-wrap: wrap !important; }
        }
        </style>""",
        unsafe_allow_html=True,
    )
    # 상단 네비: 로고 + 메뉴 + 로그인만 ('무료로 시작하기' 버튼 삭제)
    nav_logo, nav_login = st.columns([6, 1], vertical_alignment="center")
    with nav_logo:
        st.markdown(
            '<div style="display:flex; align-items:baseline; gap:14px;">'
            '<span style="font-size:22px; font-weight:800; color:#176F68; letter-spacing:-0.02em;">MetriX</span>'
            '<span style="font-size:18px; font-weight:800; color:#176F68;">PrePanel</span>'
            '<span style="font-size:14px;">'
            '<a href="#pp-product" style="color:#5A6E6B; text-decoration:none; margin-right:18px;">제품</a>'
            '<a href="#pp-how" style="color:#5A6E6B; text-decoration:none; margin-right:18px;">기능</a>'
            '<a href="#pp-pricing" style="color:#5A6E6B; text-decoration:none; margin-right:18px;">요금제</a>'
            '<a href="#pp-contact" style="color:#5A6E6B; text-decoration:none;">문의</a>'
            "</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    with nav_login:
        if st.button("로그인", key="landing_login_btn", use_container_width=True):
            login_dialog()
    st.divider()
    # 히어로: 좌 텍스트 + 우 리포트 목업
    st.markdown('<div id="pp-product"></div>', unsafe_allow_html=True)
    _hero_l, _hero_r = st.columns([1, 1.08], vertical_alignment="center")
    with _hero_l:
        st.markdown(
            '<div style="padding: 0.6rem 0;">'
            '<span style="display:inline-block; background:#E4F2ED; color:#176F68; font-size:12px; '
            'font-weight:700; letter-spacing:0.08em; padding:6px 12px; border-radius:999px;">'
            "PREPANEL · AI 설문 사전테스트</span>"
            '<div style="font-size:1.85rem; font-weight:800; line-height:1.3; letter-spacing:-0.02em; '
            'color:#17242C; margin-top:1rem;">'
            "설문을 배포하기 전,<br>"
            '<span style="background:linear-gradient(transparent 62%, #9ADCD0 62%);">응답을 먼저</span> '
            "만나보세요.</div>"
            '<div style="font-size:1.02rem; line-height:1.65; color:#5A6E6B; margin-top:0.9rem;">'
            "10,000명의 가상 페르소나가 설문에 응답합니다.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("첫 프로젝트 시작하기 ›", type="primary", key="landing_hero_btn"):
            login_dialog()
    with _hero_r:
        st.markdown(_report_mockup_html(), unsafe_allow_html=True)
    # 수치 스트립 (검증된 내용만: 10,000건·5단계·17개 시도·CC BY 4.0)
    st.markdown(
        '<div class="pp-stats" style="display:flex; background:#E4F2ED; border-radius:14px; '
        'margin-top:1.4rem; padding:0.9rem 0.4rem;">'
        '<div style="flex:1; text-align:center;">'
        '<div style="font-size:1.35rem; font-weight:800; color:#176F68; white-space:nowrap;">10,000명</div>'
        '<div style="font-size:12px; color:#5A6E6B;">가상 페르소나 응답</div></div>'
        '<div style="flex:1; text-align:center;">'
        '<div style="font-size:1.35rem; font-weight:800; color:#176F68; white-space:nowrap;">5단계</div>'
        '<div style="font-size:12px; color:#5A6E6B;">설계부터 결과까지</div></div>'
        '<div style="flex:1; text-align:center;">'
        '<div style="font-size:1.35rem; font-weight:800; color:#176F68; white-space:nowrap;">17개 시도</div>'
        '<div style="font-size:12px; color:#5A6E6B;">지역 분포 쿼터</div></div>'
        '<div style="flex:1; text-align:center;">'
        '<div style="font-size:1.35rem; font-weight:800; color:#176F68; white-space:nowrap;">CC BY 4.0</div>'
        '<div style="font-size:12px; color:#5A6E6B;">Nemotron-Personas-Korea</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    # 기능 3단 (시안: 01 설문 가져오기 · 02 가상 응답 생성 · 03 문제 문항 찾기)
    st.markdown(
        '<div id="pp-how" style="font-size:13px; font-weight:700; letter-spacing:0.08em; color:#176F68; '
        'margin:1.8rem 0 0.9rem;">어떻게 동작하나요?</div>',
        unsafe_allow_html=True,
    )
    _f1, _f2, _f3 = st.columns(3)
    with _f1:
        with st.container(border=True):
            st.markdown(
                '<span style="display:inline-block; background:#176F68; color:#FFFFFF; font-size:12px; '
                'font-weight:800; border-radius:999px; padding:3px 10px;">01</span>'
                ' <span style="font-size:1.05rem; font-weight:800; color:#17242C;">설문 가져오기</span>',
                unsafe_allow_html=True,
            )
            st.caption("이미 만든 설문을 그대로 업로드하거나, 링크를 붙여넣어 간편하게 불러올 수 있습니다.")
            st.markdown(
                '<div style="display:flex; gap:8px; align-items:center; margin-top:10px;">'
                '<div style="flex:1; background:#FFFFFF; border:1.5px dashed #BFE3DA; border-radius:10px; '
                'padding:10px 10px;">'
                '<div style="height:6px; background:#E4F2ED; border-radius:3px;"></div>'
                '<div style="height:6px; background:#E4F2ED; border-radius:3px; margin-top:6px; width:80%;"></div>'
                '<div style="display:flex; gap:6px; margin-top:8px; align-items:center;">'
                '<span style="width:14px; height:14px; border-radius:50%; background:#39BCA7; color:#FFFFFF; '
                'font-size:9px; font-weight:800; display:inline-flex; align-items:center; justify-content:center;">✓</span>'
                '<span style="width:14px; height:14px; border-radius:50%; background:#39BCA7; color:#FFFFFF; '
                'font-size:9px; font-weight:800; display:inline-flex; align-items:center; justify-content:center;">✓</span>'
                '<span style="font-size:10px; color:#5A6E6B;">문항 인식 완료</span></div></div>'
                '<div style="display:flex; flex-direction:column; gap:6px;">'
                '<span style="background:#2E7D32; color:#FFFFFF; font-size:10px; font-weight:800; '
                'border-radius:6px; padding:4px 8px; text-align:center;">XLSX</span>'
                '<span style="background:#1565C0; color:#FFFFFF; font-size:10px; font-weight:800; '
                'border-radius:6px; padding:4px 8px; text-align:center;">DOCX</span>'
                '<span style="background:#C62828; color:#FFFFFF; font-size:10px; font-weight:800; '
                'border-radius:6px; padding:4px 8px; text-align:center;">PDF</span>'
                "</div></div>",
                unsafe_allow_html=True,
            )
    with _f2:
        with st.container(border=True):
            st.markdown(
                '<span style="display:inline-block; background:#176F68; color:#FFFFFF; font-size:12px; '
                'font-weight:800; border-radius:999px; padding:3px 10px;">02</span>'
                ' <span style="font-size:1.05rem; font-weight:800; color:#17242C;">가상 응답 생성</span>',
                unsafe_allow_html=True,
            )
            st.caption("다양한 인구통계와 행동 특성을 가진 10,000명의 AI 가상 페르소나가 설문에 응답합니다.")
            st.markdown(
                '<div style="display:flex; justify-content:center; gap:10px; margin-top:12px;">'
                '<div style="width:52px; height:52px; border-radius:50%; background:#E4F2ED; '
                'position:relative; overflow:hidden; flex-shrink:0;">'
                '<div style="position:absolute; left:50%; top:11px; width:20px; height:20px; border-radius:50%; '
                'background:#F6CBB8; transform:translateX(-50%);"></div>'
                '<div style="position:absolute; left:50%; top:33px; width:32px; height:20px; '
                'border-radius:12px 12px 0 0; background:#7B8FB8; transform:translateX(-50%);"></div></div>'
                '<div style="width:62px; height:62px; border-radius:50%; background:#E4F2ED; '
                'position:relative; overflow:hidden; flex-shrink:0; margin-top:-6px; '
                'border:2px solid #39BCA7;">'
                '<div style="position:absolute; left:50%; top:8px; width:24px; height:24px; border-radius:50%; '
                'background:#F6CBB8; transform:translateX(-50%);"></div>'
                '<div style="position:absolute; left:50%; top:2px; width:26px; height:14px; '
                'border-radius:13px 13px 0 0; background:#2A3663; transform:translateX(-50%);"></div>'
                '<div style="position:absolute; left:50%; top:38px; width:38px; height:24px; '
                'border-radius:14px 14px 0 0; background:#39BCA7; transform:translateX(-50%);"></div></div>'
                '<div style="width:52px; height:52px; border-radius:50%; background:#E4F2ED; '
                'position:relative; overflow:hidden; flex-shrink:0;">'
                '<div style="position:absolute; left:50%; top:11px; width:20px; height:20px; border-radius:50%; '
                'background:#F6CBB8; transform:translateX(-50%);"></div>'
                '<div style="position:absolute; left:50%; top:33px; width:32px; height:20px; '
                'border-radius:12px 12px 0 0; background:#5A6E6B; transform:translateX(-50%);"></div></div>'
                "</div>"
                '<div style="text-align:center; margin-top:10px;">'
                '<span style="font-size:11px; font-weight:700; color:#176F68; background:#E4F2ED; '
                'border-radius:999px; padding:3px 10px;">가상 페르소나 10,000명 ✓</span></div>',
                unsafe_allow_html=True,
            )
    with _f3:
        with st.container(border=True):
            st.markdown(
                '<span style="display:inline-block; background:#176F68; color:#FFFFFF; font-size:12px; '
                'font-weight:800; border-radius:999px; padding:3px 10px;">03</span>'
                ' <span style="font-size:1.05rem; font-weight:800; color:#17242C;">문제 문항 찾기</span>',
                unsafe_allow_html=True,
            )
            st.caption("응답 편향, 낮은 변별력, 문항 중복 등 개선이 필요한 문항을 AI가 정확히 짚어줍니다.")
            st.markdown(
                '<div style="position:relative; background:#FFFFFF; border:1px solid #D8E6DF; '
                'border-radius:10px; padding:10px 12px; margin-top:10px;">'
                '<div style="display:flex; align-items:center; gap:8px; padding:5px 0; '
                'border-bottom:1px solid #E4F2ED;">'
                '<span style="width:16px; height:16px; border-radius:50%; border:1.5px solid #BFE3DA; '
                'flex-shrink:0;"></span>'
                '<span style="font-size:11px; font-weight:700; color:#17242C;">Q06</span>'
                '<span style="font-size:11px; color:#5A6E6B;">문항 중복 가능성</span>'
                '<span style="margin-left:auto; width:15px; height:15px; border-radius:50%; background:#E4F2ED; '
                'color:#176F68; font-size:10px; font-weight:800; display:inline-flex; align-items:center; '
                'justify-content:center; flex-shrink:0;">i</span></div>'
                '<div style="display:flex; align-items:center; gap:8px; padding:5px 0; '
                'background:#FDECEC; border-radius:6px; margin:4px -4px; padding-left:8px; padding-right:8px;">'
                '<span style="width:16px; height:16px; border-radius:50%; border:1.5px solid #E57373; '
                'flex-shrink:0;"></span>'
                '<span style="font-size:11px; font-weight:700; color:#17242C;">Q07</span>'
                '<span style="font-size:11px; color:#5A6E6B; white-space:nowrap;">응답 편향 가능성</span>'
                '<span style="margin-left:auto; width:15px; height:15px; border-radius:50%; background:#E57373; '
                'color:#FFFFFF; font-size:10px; font-weight:800; display:inline-flex; align-items:center; '
                'justify-content:center; flex-shrink:0;">!</span></div>'
                '<div style="display:flex; align-items:center; gap:8px; padding:5px 0;">'
                '<span style="width:16px; height:16px; border-radius:50%; border:1.5px solid #BFE3DA; '
                'flex-shrink:0;"></span>'
                '<span style="font-size:11px; font-weight:700; color:#17242C;">Q11</span>'
                '<span style="font-size:11px; color:#5A6E6B;">변별력 낮음</span>'
                '<span style="margin-left:auto; width:15px; height:15px; border-radius:50%; background:#E4F2ED; '
                'color:#176F68; font-size:10px; font-weight:800; display:inline-flex; align-items:center; '
                'justify-content:center; flex-shrink:0;">i</span></div>'
                '<div style="position:absolute; right:8px; bottom:-14px; width:34px; height:34px; '
                'border:3px solid #176F68; border-radius:50%; background:rgba(255,255,255,0.65);"></div>'
                '<div style="position:absolute; right:2px; bottom:-22px; width:14px; height:4px; '
                'background:#176F68; border-radius:2px; transform:rotate(45deg);"></div>'
                "</div>",
                unsafe_allow_html=True,
            )
    # 문제 제기 섹션
    st.markdown(
        '<div style="text-align:center; background:#E4F2ED; border-radius:12px; padding:1.6rem 1rem; margin-top:1.2rem;">'
        '<div style="font-size:1.25rem; font-weight:800; letter-spacing:-0.01em; color:#17242C;">'
        "가장 비싼 실수는 현장에서 터진다.</div>"
        '<div style="font-size:0.95rem; line-height:1.65; color:#5A6E6B; max-width:520px; margin:0.6rem auto 0;">'
        "잘못된 문항 하나가 현장 나가서야 드러나면, 다시 돌릴 수 없는 비용이 됩니다. "
        "사전테스트는 그 비용을 배포 전에 치릅니다.</div></div>",
        unsafe_allow_html=True,
    )
    # 요금제 섹션 (네비 '요금제' 앵커 대상)
    # 정식 요금제는 아직 정해지지 않았다 — 파일럿 기간 첫 프로젝트 무료만 안내한다.
    st.markdown(
        '<div id="pp-pricing" style="margin-top:1.8rem;">'
        '<div style="font-size:13px; font-weight:700; letter-spacing:0.08em; color:#176F68; '
        'margin-bottom:0.9rem;">요금제</div>'
        '<div style="background:#FFFFFF; border:1px solid #D8E6DF; border-radius:14px; '
        'padding:1.4rem 1.2rem; text-align:center;">'
        '<div style="font-size:1.15rem; font-weight:800; color:#17242C;">파일럿 기간</div>'
        '<div style="font-size:1.6rem; font-weight:800; color:#176F68; margin-top:0.4rem;">'
        "첫 프로젝트 무료</div>"
        '<div style="font-size:0.95rem; line-height:1.65; color:#5A6E6B; margin-top:0.5rem;">'
        "신용카드가 필요 없습니다. 첫 프로젝트를 무료로 시작해 보세요.<br>"
        "정식 요금제는 준비 중입니다. 출시 시 안내해 드립니다.</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    _p_l, _p_c, _p_r = st.columns([1, 1, 1])
    with _p_c:
        if st.button(
            "첫 프로젝트 무료로 시작하기", type="primary", key="landing_pricing_btn",
            use_container_width=True,
        ):
            login_dialog()
    # 문의 섹션 (네비 '문의' 앵커 대상)
    st.markdown(
        '<div id="pp-contact" style="background:#E4F2ED; border-radius:12px; '
        'padding:1.4rem 1.2rem; margin-top:1.8rem; text-align:center;">'
        '<div style="font-size:1.15rem; font-weight:800; color:#17242C;">문의</div>'
        '<div style="font-size:0.95rem; line-height:1.65; color:#5A6E6B; margin-top:0.5rem;">'
        "PrePanel에 대해 궁금한 점이 있으시면 메트릭스로 문의해 주세요.<br>"
        "액세스 코드는 관리자에게 문의하세요.</div>"
        '<div style="margin-top:0.9rem;">'
        '<a href="https://www.metrix.co.kr" target="_blank" '
        'style="display:inline-block; background:#176F68; color:#FFFFFF; font-size:14px; '
        'font-weight:700; text-decoration:none; border-radius:8px; padding:10px 18px;">'
        "메트릭스 홈페이지 바로가기</a></div></div>",
        unsafe_allow_html=True,
    )
    # 마지막 CTA
    st.markdown(
        '<div style="text-align:center; font-size:1.25rem; font-weight:800; color:#17242C; margin:1.6rem 0 0.9rem;">'
        "설문 배포 전에, 먼저 테스트하세요.</div>",
        unsafe_allow_html=True,
    )
    _b_l, _b_c, _b_r = st.columns([1, 1, 1])
    with _b_c:
        if st.button(
            "첫 프로젝트 시작하기", type="primary", key="landing_final_btn",
            use_container_width=True,
        ):
            login_dialog()
    st.divider()
    st.caption("© 2026 MetriX · 페르소나: Nemotron-Personas-Korea (NVIDIA, CC BY 4.0)")
    st.caption(
        "합성응답자는 실측을 대체할 수 없으며, 파일럿·사전 검증 용도로만 사용합니다."
    )
def _render_projects_page():
    """프로젝트 관리 페이지 — 프로젝트명 텍스트 목록 + 프로젝트별 삭제, 생성·이름 변경.

    상단 메뉴 '프로젝트 관리'에서 들어오는 별도 페이지. 프로젝트명은
    드롭다운이 아니라 텍스트 목록으로 제시하고, 각 프로젝트마다 삭제 버튼을 둔다.
    삭제는 core/projects.py의 delete_project()를 재사용하고, 마지막 프로젝트는
    삭제할 수 없다. 삭제 후에는 남은 첫 프로젝트로 이동한다.
    """
    st.header("프로젝트 관리")
    st.caption("프로젝트 목록 · 새 프로젝트 만들기 · 이름 변경")
    if not SheetsProjectStore.is_configured():
        with st.expander("Google Sheets 연동 설정", expanded=False):
            st.markdown(SheetsProjectStore.setup_guide())
    # 프로젝트 목록 (텍스트 + 프로젝트별 열기/삭제)
    st.subheader("프로젝트 목록")
    _all_projects = _project_store.list_projects()
    if not _all_projects:
        st.caption("프로젝트가 없습니다.")
    for _p in _all_projects:
        _is_current = _p["id"] == current_pid
        _label = ("● " if _is_current else "") + _p["name"]
        if _is_current:
            _label += " (현재)"
        _c_name, _c_open, _c_del = st.columns([5.0, 1.0, 1.0])
        with _c_name:
            st.markdown(_label)
        with _c_open:
            if not _is_current:
                if st.button("열기", key=f"btn_proj_open_{_p['id']}", use_container_width=True):
                    _save_current_project()
                    _load_project_to_store(_p["id"])
                    st.rerun()
        with _c_del:
            if st.button("🗑️ 삭제", key=f"btn_proj_del_{_p['id']}", use_container_width=True):
                if len(_all_projects) <= 1:
                    st.error("마지막 프로젝트는 삭제할 수 없습니다.")
                else:
                    _autosave_current_project()
                    _project_store.delete_project(_p["id"])
                    _remaining = _project_store.list_projects()
                    _load_project_to_store(_remaining[0]["id"])
                    store.set("page", "projects")
                    st.rerun()
    st.divider()
    # 새 프로젝트 추가
    with st.expander("➕ 새 프로젝트 추가", expanded=False):
        with st.form("new_project_form", border=False):
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
    # 이름 변경 (현재 프로젝트)
    with st.expander("이름 변경", expanded=False):
        cur_meta = _project_store.get_meta(current_pid)
        new_name = st.text_input(
            "프로젝트 이름",
            value=cur_meta.get("name", ""),
            key=f"project_rename_input_{current_pid}",
        )
        if st.button("이름 변경", key="btn_rename_project"):
            try:
                _project_store.rename_project(current_pid, new_name)
            except ValueError as e:
                st.error(str(e))
            else:
                st.rerun()


def _render_ai_settings():
    """AI 설정 팝오버 — 제공자 선택, 내 API 키, 인증 상태, 작업 요약."""
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
    # 사용자 API 키 직접 입력 (기본: 미사용)
    use_custom = st.checkbox(
        "🔑 내 API 키 사용", value=False, key=f"use_custom_{store.get('provider')}"
    )
    user_key = ""
    if use_custom:
        user_key = st.text_input(
            "API 키 직접 입력",
            type="password",
            placeholder=f"{provider.label} API 키를 입력하세요",
            key=f"custom_key_{store.get('provider')}",
            help="입력한 키는 브라우저 세션 메모리에만 보관되며, 서버에 저장되지 않습니다. 페이지를 새로고침하면 사라집니다.",
        )
        st.caption("⚠️ 입력한 API 키는 저장되지 않습니다. 브라우저를 닫거나 새로고침하면 다시 입력해야 합니다.")
    # 체크 해제 시 커스텀 키 제거 (secrets 키로 복귀)
    if not use_custom and provider.has_custom_key():
        provider.clear_custom_key()
    if use_custom:
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


@st.dialog("⚙️ AI 설정", width="large")
def ai_settings_dialog():
    """AI 설정 다이얼로그 — _render_ai_settings() 본문 재사용."""
    _render_ai_settings()


def _render_sidebar():
    """좌측 패널 (Qualtrics '프로젝트 폴더' 카피) — 귀하의 폴더 + 새 프로젝트.

    없는 기능(별표 표시·공유)은 UI에 흔적도 남기지 않는다.
    접기는 네이티브 사이드바 접기 컨트롤(<<)을 쓴다.
    메뉴(프로젝트 관리·AI 설정·이론·업데이트·로그아웃)는 상단 헤더로 이동했다.
    """
    with st.sidebar:
        st.subheader("프로젝트 폴더")
        with st.expander("📁 귀하의 폴더", expanded=True):
            if projects:
                for p in projects:
                    is_current = p["id"] == current_pid
                    label = ("● " if is_current else "") + p["name"]
                    if st.button(
                        label,
                        key=f"sidebar_proj_{p['id']}",
                        use_container_width=True,
                        type="primary" if is_current else "secondary",
                    ):
                        if not is_current:
                            _save_current_project()
                            _load_project_to_store(p["id"])
                            st.rerun()
            else:
                st.caption("프로젝트가 없습니다.")
        if st.button("＋ 새 프로젝트", key="btn_sidebar_new_project", use_container_width=True):
            # 이름 입력 없이 한 번에 생성한다. "새 프로젝트"가 이미 있으면
            # 날짜를 붙여 고유한 이름을 만든다 (기존 폼은 이름 중복 시 오류).
            existing_names = {
                p["name"].strip().casefold()
                for p in _project_store.list_projects()
            }
            new_name = "새 프로젝트"
            if new_name.casefold() in existing_names:
                new_name = f"새 프로젝트 {datetime.now().strftime('%m%d-%H%M')}"
                suffix = 2
                while new_name.casefold() in existing_names:
                    new_name = (
                        f"새 프로젝트 {datetime.now().strftime('%m%d-%H%M')}-{suffix}"
                    )
                    suffix += 1
            new_pid = _project_store.create_project(new_name)
            _save_current_project()
            _load_project_to_store(new_pid)
            st.rerun()


def _render_header():
    """상단 헤더 (Qualtrics 카피) — 로고 + 주 메뉴 팝오버 + 저장 + 도움말 + 내 계정.

    주 메뉴: 5개 단계(스테퍼와 같은 goto 이동) + 프로젝트 관리·업데이트 기록
    (별도 페이지로 이동). 내 계정: AI 설정 다이얼로그 + 로그아웃 + 액세스 코드 확인 표시.
    알림 종·전환기는 해당 기능이 없어 넣지 않는다.
    """
    c_logo, c_menu, c_save, c_help, c_account = st.columns(
        [5.0, 1.0, 0.7, 0.4, 1.3], vertical_alignment="center"
    )
    with c_logo:
        st.markdown(
            '<div style="display:flex; align-items:baseline; gap:10px;">'
            '<span style="font-size:20px; font-weight:800; color:#176F68;">MetriX</span>'
            '<span style="font-size:15px; font-weight:700;">AI 합성응답자 기반 Pre-test</span>'
            "</div>"
            '<div style="font-size:12px; color:#5A6E6B; margin-top:-2px;">'
            "Synthetic Respondent Pre-Test</div>",
            unsafe_allow_html=True,
        )
    with c_menu:
        with st.popover("☰ 주 메뉴", use_container_width=True):
            current_step = store.get("step")
            for step_number, step_name in enumerate(STEPS, start=1):
                mark = "✓ " if step_number < current_step else ""
                if st.button(
                    f"{mark}{step_number}. {step_name}",
                    key=f"mainmenu_step_{step_number}",
                    use_container_width=True,
                    type="primary" if step_number == current_step else "secondary",
                ):
                    goto(step_number)
            st.divider()
            if st.button("📁 프로젝트 관리", key="btn_mainmenu_project_mgmt", use_container_width=True):
                goto_page("projects")
            if st.button("업데이트 기록", key="btn_mainmenu_update_log", use_container_width=True):
                goto_page("updates")
    with c_save:
        if st.button("💾 저장", key="btn_save_project", use_container_width=True):
            _save_current_project()
            st.success("저장되었습니다.")
    with c_help:
        if st.button("?", key="btn_help_theory", use_container_width=True, help="이론적 배경"):
            theory_dialog()
    with c_account:
        with st.popover("👤 내 계정", use_container_width=True):
            st.caption("🔓 액세스 코드 확인됨")
            if st.button("⚙️ AI 설정", key="btn_account_ai_settings", use_container_width=True):
                ai_settings_dialog()
            if st.button("로그아웃", key="btn_account_logout", use_container_width=True):
                st.session_state["access_granted"] = False
                store.set("project_id", None)
                st.rerun()
    # 2행: 작업 순서 가이드 + 저장 상태
    _render_workflow_guide()
    _render_save_status()


def _render_stepper():
    """1~5단계 진행 표시 (완료 ✓ · 현재 · 예정). 버튼을 누르면 해당 단계로 이동한다."""
    current_step = store.get("step")
    cols = st.columns(5)
    for step_number, step_name in enumerate(STEPS, start=1):
        status = (
            "complete" if step_number < current_step
            else "current" if step_number == current_step
            else "upcoming"
        )
        mark = "✓ " if status == "complete" else ""
        with cols[step_number - 1]:
            if st.button(
                f"{mark}{step_number}. {step_name}",
                key=f"stepper_{status}_{step_number}",
                use_container_width=True,
                type="primary" if status == "current" else "secondary",
                help="현재 단계" if status == "current" else f"{step_name} 단계로 이동",
            ):
                goto(step_number)
step = store.get("step")


# ----------------------------------------------------------------------------
# Step 1: 설문 설계
# ----------------------------------------------------------------------------
def _replace_questionnaire(questions):
    store.set("questions", questions)
    st.session_state["questionnaire_revision"] = st.session_state.get("questionnaire_revision", 0) + 1
    for state_key in ("learned", "learned_file", "results_df", "gen_meta"):
        store.set(state_key, None)

def step1():



    parse_notice = st.session_state.pop("questionnaire_parse_notice", None)
    if parse_notice:
        st.success(parse_notice["success"])
        if parse_notice.get("info"):
            st.info(parse_notice["info"])
        if parse_notice.get("warning"):
            st.warning(parse_notice["warning"])

    questions = store.get("questions")
    sample_texts = {q["text"] for q in q_service.load_sample_questions()}
    custom_questions = [q for q in questions if q["text"] not in sample_texts]
    if custom_questions and len(custom_questions) < len(questions):
        _replace_questionnaire(custom_questions)
        _save_current_project()
        st.rerun()
    question_namespace = f"{store.get('project_id')}_{st.session_state.get('questionnaire_revision', 0)}"

    if not questions:
        with st.container(border=True):
            st.write("아직 문항이 없습니다. 샘플 설문으로 시작하거나 직접 추가하세요.")
            if st.button("샘플 설문 불러오기", type="primary"):
                _replace_questionnaire(q_service.load_sample_questions())
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
                                _replace_questionnaire(new_questions)
                                n_branch = sum(1 for q in parsed if q.get("branching"))
                                stats = PdfQuestionnaireService.last_parse_stats
                                chunk_note = ""
                                if stats.get("total_chunks", 1) > 1:
                                    chunk_note = (
                                        f" {stats['total_chunks']}개 구간으로 나누어 분석했습니다."
                                    )
                                st.session_state["questionnaire_parse_notice"] = {
                                    "success": (
                                        f"{len(parsed)}개 문항을 인식했습니다.{chunk_note} "
                                        f"기존 문항은 제거되었습니다. "
                                        f"(분기 로직 {n_branch}건 포함)"
                                    ),
                                    "info": "",
                                    "warning": "",
                                }
                                if stats.get("duplicates_removed", 0):
                                    st.session_state["questionnaire_parse_notice"]["info"] = (
                                        f"겹치는 구간에서 중복 인식된 "
                                        f"{stats['duplicates_removed']}개 문항을 정리했습니다."
                                    )
                                if stats.get("failed_chunks", 0):
                                    st.session_state["questionnaire_parse_notice"]["warning"] = (
                                        f"{stats['failed_chunks']}개 구간은 AI 응답을 받지 못해 "
                                        "해당 구간의 문항이 일부 누락될 수 있습니다."
                                    )
                                st.rerun()
                    except Exception as e:
                        st.error(f"처리 중 오류: {e}")

    for idx, q in enumerate(questions):
        with st.expander(q_service.question_summary(q), expanded=False):
            new_text = st.text_input("문항 내용", value=q["text"], key=f"qtext_{question_namespace}_{q['id']}")
            if new_text != q["text"]:
                q["text"] = new_text
                store.set("questions", questions)
            qtype = get_question_type(q["type"])
            if qtype.needs_options:
                opts_raw = st.text_area(
                    "보기 (한 줄에 하나씩)",
                    value="\n".join(q.get("options", [])),
                    key=f"qopts_{question_namespace}_{q['id']}",
                )
                q["options"] = [o.strip() for o in opts_raw.split("\n") if o.strip()]
                store.set("questions", questions)
            # 분기 로직 표시/편집
            branching = st.text_input(
                "분기 로직 (예: 문3에서 ① 응답 시 문5로 이동)",
                value=q.get("branching", ""),
                key=f"qbranch_{question_namespace}_{q['id']}",
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
        with st.form("add_question", border=False):
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
                if questions and all(q["text"] in sample_texts for q in questions):
                    questions = []
                    qdict["id"] = q_service.new_question_id(questions)
                questions.append(qdict)
                _replace_questionnaire(questions)
                st.rerun()

    st.divider()
    st.button(
        "다음: 페르소나 설계 →",
        type="primary",
        disabled=not questions,
        on_click=lambda: goto(2),
    )
    if not questions:
        st.caption("문항을 1개 이상 추가해야 다음 단계로 이동할 수 있습니다.")


# ----------------------------------------------------------------------------
# Step 2: 페르소나 설계 — Nemotron 쿼터 UI
# ----------------------------------------------------------------------------
_NEMOTRON_QUOTA_DIMS = {
    "연령대": [label for label, _, _ in AGE_BANDS],
    "성별": ["여성", "남성"],
    "지역": list(REGION_MAP.keys()),
}


def _render_nemotron_quota_ui() -> bool:
    """Nemotron 쿼터 입력 UI. 발췌본 로드 상태와 합계 검증을 함께 보여준다.

    Phase 2: 비율/건수 쿼터 방식 선택, 지역 단위(권역/시도) 선택,
    조사 주제별 페르소나 측면 선택을 지원한다.

    Returns:
        쿼터가 유효한지 (비율 모드: 모든 차원 합계 100% + 발췌본 로드 성공,
        건수 모드: 셀 1개 이상 등록 + 발췌본 로드 성공).
    """
    try:
        records = load_nemotron_extract()
    except ValueError as exc:
        st.error(str(exc))
        st.info("Nemotron 출처를 쓸 수 없습니다. "
                "발췌본 파일(data/nemotron_personas_10k.parquet)을 확인해 주세요.")
        return False
    st.success(
        f"Nemotron 발췌본 {len(records):,}건 로드됨 "
        f"({NEMOTRON_EXTRACT_VERSION}, {NEMOTRON_EXTRACT_RELEASE_DATE})"
    )

    # (c) 지역 단위 선택 (비율·건수 공통)
    level = st.radio(
        "지역 단위", options=["region", "province"],
        format_func=lambda k: {"region": "권역 (5개)",
                               "province": "시도 (17개)"}[k],
        index=0 if store.get("nemotron_region_level") != "province" else 1,
        horizontal=True, key="nemotron_region_level_radio",
    )
    if level != store.get("nemotron_region_level"):
        store.set("nemotron_region_level", level)
        store.set("nemotron_quotas",
                  nemotron_default_quotas(records, region_level=level))
        # 지역 키 체계가 바뀌므로 건수 쿼터는 초기화한다.
        store.set("nemotron_count_quotas", None)
        st.rerun()
    region_cats = (list(NEMOTRON_PROVINCES) if level == "province"
                   else list(REGION_MAP.keys()))

    # (a) 쿼터 방식 선택
    mode = st.radio(
        "쿼터 방식", options=["ratio", "count"],
        format_func=lambda k: {"ratio": "비율 쿼터 (%)",
                               "count": "건수 쿼터 (셀별 목표 건수)"}[k],
        index=0 if store.get("nemotron_quota_mode") != "count" else 1,
        horizontal=True, key="nemotron_quota_mode_radio",
    )
    store.set("nemotron_quota_mode", mode)

    quotas = store.get("nemotron_quotas")
    if (not quotas
            or set(quotas.get("지역", {}).keys()) != set(region_cats)):
        quotas = nemotron_default_quotas(records, region_level=level)
        store.set("nemotron_quotas", quotas)

    valid = True
    if mode == "count":
        # (b) 건수 쿼터 입력 UI
        st.caption("건수 쿼터가 비율 쿼터보다 우선합니다. 비율 쿼터는 무시됩니다.")
        counts = store.get("nemotron_count_quotas") or {}
        c1, c2, c3, c4 = st.columns(4)
        age = c1.selectbox("연령대", [label for label, _, _ in AGE_BANDS],
                           key="count_age")
        sex = c2.selectbox("성별", ["여성", "남성"], key="count_sex")
        region_key = c3.selectbox("지역", region_cats, key="count_region")
        n_cell = c4.number_input("건수", min_value=1, max_value=200, value=10,
                                 step=1, key="count_n")
        if st.button("셀 추가", key="count_add"):
            counts[f"{age}|{sex}|{region_key}"] = int(n_cell)
            store.set("nemotron_count_quotas", counts)
            st.rerun()
        for key in sorted(counts):
            col_l, col_r = st.columns([5, 1])
            with col_l:
                st.write(f"{key.replace('|', ' · ')} — {counts[key]}건")
            with col_r:
                if st.button("삭제", key=f"count_del_{key}"):
                    del counts[key]
                    store.set("nemotron_count_quotas", counts)
                    st.rerun()
        total = sum(counts.values())
        st.write(f"건수 쿼터 합계: {total}건")
        if total <= 0:
            st.error("건수 쿼터 셀을 1개 이상 추가해 주세요.")
            valid = False
        st.caption("합계가 생성 인원과 다르면 생성 시 경고가 기록되고, 부족분은 "
                   "발췌본 분포대로 채우거나 가장 큰 셀부터 줄여 맞춥니다. "
                   "생성 인원은 생성 단계에서 정합니다.")
        if level == "province":
            st.caption("시도 단위 셀 부족 시 권역으로 완화합니다. 예: 제주 70대 "
                       "여성은 발췌본에 11건뿐이라 강원·제주 권역에서 채우고 "
                       "완화 기록을 남깁니다.")
    else:
        dims = {"연령대": [label for label, _, _ in AGE_BANDS],
                "성별": ["여성", "남성"],
                "지역": region_cats}
        for dim, cats in dims.items():
            with st.container(border=True):
                st.subheader(dim)
                dim_quota = quotas.setdefault(dim, {})
                cols = st.columns(max(1, len(cats)))
                for i, cat in enumerate(cats):
                    with cols[i % len(cols)]:
                        v = st.number_input(
                            cat, min_value=0.0, max_value=100.0,
                            value=float(dim_quota.get(cat, 0.0)),
                            step=1.0, key=f"nemotron_pct_{dim}_{cat}",
                        )
                        dim_quota[cat] = v
                store.set("nemotron_quotas", quotas)
                total = sum(dim_quota.get(c, 0.0) for c in cats)
                if abs(total - 100.0) > 0.01:
                    st.error(f"합계가 100%가 아닙니다 (현재 {total:.1f}%).")
                    valid = False
                else:
                    st.success("합계 100% 확인")
    if st.button("쿼터 초기화 (발췌본 분포대로)"):
        store.set("nemotron_quotas",
                  nemotron_default_quotas(records, region_level=level))
        store.set("nemotron_count_quotas", None)
        st.rerun()

    # (d) 측면 선택 (조사 주제별)
    facets = st.multiselect(
        "프롬프트에 쓸 페르소나 측면 (조사 주제별 선택)",
        options=list(NEMOTRON_ALL_FACETS),
        format_func=lambda f: NEMOTRON_FACET_LABELS[f],
        default=list(store.get("nemotron_facets") or NEMOTRON_FACETS),
        key="nemotron_facets_select",
    )
    store.set("nemotron_facets", facets or None)
    st.caption("기본값은 요약+직업입니다. 예: 식생활 조사는 식생활·가족을 "
               "추가하세요. 여행·스포츠·예술은 10,000건 발췌본에 없어 "
               "지금은 반영되지 않습니다.")
    st.caption(
        "초기값은 발췌본 분포 그대로입니다. 비율 쿼터는 합계가 100%가 아니면 "
        "다음 단계로 이동할 수 없습니다. 좁은 셀(예: 제주 70대 이상 여성)은 "
        "발췌본에 부족할 수 있으며, 이 경우 완화 규칙에 따라 채우고 완화 "
        "기록을 결과에 남깁니다."
    )
    return valid


# ----------------------------------------------------------------------------
# Step 2: 페르소나 설계
# ----------------------------------------------------------------------------
def step2():



    source = store.get("persona_source") or "nemotron"
    source = st.radio(
        "페르소나 출처",
        options=list(PERSONA_SOURCES.keys()),
        format_func=lambda k: PERSONA_SOURCES[k],
        index=list(PERSONA_SOURCES.keys()).index(source)
        if source in PERSONA_SOURCES else 0,
        horizontal=True,
        key="persona_source_radio",
    )
    if source != store.get("persona_source"):
        store.set("persona_source", source)
        st.rerun()
    if source == "nemotron":
        st.caption(
            "실측 분포 기반: NVIDIA Nemotron-Personas-Korea(v1.0)의 한국인 "
            "페르소나 100만 건에서 계층추출한 10,000건 발췌본에서 쿼터대로 추출합니다. "
            "손으로 정한 가중치가 아니라 실측 분포에서 샘플링하므로 패널 구성이 "
            "모집단을 닮았다는 근거를 제시할 수 있습니다."
        )

    dims = store.get("persona_dims")
    valid = True

    manual_dims = list(dims.keys()) if source == "manual" else []
    for dim in manual_dims:
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

    if source == "manual":
        with st.expander("새 인구통계 차원 추가"):
            new_dim = st.text_input("차원 이름", placeholder="예: 소득대")
            if st.button("차원 추가") and new_dim.strip():
                dims[new_dim.strip()] = {"구분1": 50.0, "구분2": 50.0}
                store.set("persona_dims", dims)
                st.rerun()

    if source == "nemotron":
        valid = _render_nemotron_quota_ui()

    st.subheader("고정밀 모드 (인터뷰 기반)")
    with st.container(border=True):
        st.caption(
            "스탠포드 Generative Agents 연구에서 정규화 정확도 83%로 가장 높았던 방식입니다. "
            "1인당 약 2시간의 인터뷰 트랜스크립트가 필요합니다. "
            "트랜스크립트를 업로드하면 생성 단계에서 고정밀 모드를 켤 수 있습니다."
        )
        if store.get("persona_source") == "nemotron":
            st.info(
                "Nemotron 출처와는 함께 쓸 수 없습니다 — 고정밀 모드는 인터뷰 기반 "
                "조건을 쓰기 때문에, Nemotron 선택 시 생성 단계에서 고정밀 모드가 "
                "자동으로 꺼집니다."
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



    with st.container(border=True, key="ai_settings_card"):
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
        source_label = PERSONA_SOURCES.get(
            store.get("persona_source") or "nemotron", "")
        st.write(f"- 페르소나 출처: {source_label}")
        n_trans = len(store.get("transcripts"))
        nemotron_on = store.get("persona_source") == "nemotron"
        use_precision = st.checkbox(
            "고정밀 모드 (인터뷰 기반)",
            value=store.get("use_precision") and n_trans > 0 and not nemotron_on,
            disabled=n_trans == 0 or nemotron_on,
            help="스탠포드 Generative Agents 연구: 정규화 정확도 83%. 트랜스크립트 업로드 시 활성화.",
        )
        store.set("use_precision", use_precision and n_trans > 0 and not nemotron_on)
        if nemotron_on:
            st.caption("Nemotron 출처와는 함께 쓸 수 없습니다 (인터뷰 기반 고정밀 모드와 중복됩니다).")
        elif n_trans == 0:
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
                    persona_source=store.get("persona_source") or "nemotron",
                    nemotron_quotas=store.get("nemotron_quotas"),
                    nemotron_count_quotas=(store.get("nemotron_count_quotas")
                                           if store.get("nemotron_quota_mode") == "count"
                                           else None),
                    nemotron_region_level=store.get("nemotron_region_level") or "region",
                    nemotron_facets=store.get("nemotron_facets"),
                )
                store.set("results_df", df)
                prov = get_provider(provider)
                used_model = prov.last_used_model or current_model()
                auto_switched = used_model != current_model()
                gen_meta = {
                    "생성일시": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "AI": prov.label,
                    "모델": used_model,
                    "응답자 수": int(n),
                    "분포 학습": "사용" if has_learned else "미사용",
                    "고정밀 모드": "사용" if store.get("use_precision") else "미사용",
                }
                if gen_service.last_gen_meta:
                    gen_meta.update(gen_service.last_gen_meta)
                store.set("gen_meta", gen_meta)
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


    # 정직한 한계 고지 — 결과 탭 상단 고정 표시
    st.warning(DISCLAIMER)

    df = store.get("results_df")
    if df is None:
        st.info("아직 생성된 데이터가 없습니다. 4단계에서 가상 응답을 생성하세요.")
        if st.button("← 생성 단계로 이동", on_click=lambda: goto(4)):
            pass
        return

    meta = store.get("gen_meta") or {}
    with st.container(border=True, key="results_summary"):
        cols = st.columns([3 if k == "모델" else 2 if k == "생성일시" else 1 for k in meta] or [1])
        for i, (k, v) in enumerate(meta.items()):
            with cols[i]:
                st.caption(k)
                st.write(str(v))

    st.subheader("데이터 미리보기")
    preview_df = df.copy()
    preview_df.index = pd.RangeIndex(start=1, stop=len(preview_df) + 1)
    st.dataframe(preview_df, use_container_width=True, height=400)

    # 파일럿 진단 리포트 (규칙 기반 — 실제 데이터 통계만 사용, 환각 없음)
    from core.diagnostics import diagnose, reference_distribution_from_records
    st.subheader("파일럿 진단 리포트")
    st.caption("실제 응답 데이터에서 계산된 통계에만 근거합니다. 임계값을 넘지 않은 문제는 보고하지 않습니다.")
    panel_ref = None
    if store.get("persona_source") == "nemotron":
        try:
            panel_ref = reference_distribution_from_records(load_nemotron_extract())
        except ValueError:
            panel_ref = None  # 발췌본 로드 실패 시 패널 진단은 생략
    findings = diagnose(df, store.get("questions") or [], panel_reference=panel_ref)
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
def _render_workflow_guide():
    """작업 순서 가이드 (UX 개선 ③). 현재 단계를 체크 표시로 보여준다."""
    has_project = bool(store.get("project_id"))
    has_questions = bool(store.get("questions"))
    has_results = store.get("results_df") is not None
    saved = _is_current_state_saved()
    items = [
        ("1. 프로젝트 만들기", has_project),
        ("2. 설문·페르소나 설계", has_questions),
        ("3. 가상 응답 생성", has_results),
        ("4. 저장", saved),
    ]
    line = " → ".join(f"{'✓' if done else '○'} {label}" for label, done in items)
    st.caption(f"작업 순서: {line}")
    if has_project and not saved:
        st.caption("프로젝트를 먼저 만들어야 작업 내용이 저장됩니다. 탭을 닫기 전에 저장 버튼을 눌러주세요.")


def _render_save_status():
    """저장 상태·마지막 저장 시각을 지속 표시 (UX 개선 ②)."""
    pid = store.get("project_id")
    if not pid:
        return
    if _is_current_state_saved():
        at = st.session_state.get("_last_saved_at") or ""
        storage = globals().get("_storage_label", "")
        suffix = f" · {at}" if at else ""
        suffix += f" · {storage}" if storage else ""
        st.caption(f"✅ 저장됨{suffix}")
    else:
        st.caption("● 저장되지 않은 변경 있음 — 저장 버튼을 눌러주세요.")


def _render_leave_guard():
    """미저장 변경이 있을 때 탭 닫기·새로고침 전 경고 (UX 개선 ④, 보조 수단).
    1차 방어선은 단계 이동·생성 완료 시 자동 저장이다."""
    if not is_code_verified() or not store.get("project_id"):
        return
    if _is_current_state_saved():
        return
    components.html(
        """<script>
        window.parent.addEventListener('beforeunload', function (e) {
            e.preventDefault();
            e.returnValue = '';
        });
        </script>""",
        height=0,
    )



# 라우팅 — Qualtrics 방식: 로그인 전에는 랜딩, 로그인 후에는 상단 메뉴별 페이지 앱
# ----------------------------------------------------------------------------
page_titles = {
    1: ("1. 설문 설계", "가상 응답을 생성할 설문 문항을 입력하세요."),
    2: ("2. 페르소나 설계", "가상 응답자의 인구통계 분포를 설정합니다. 항목별 비율의 합은 100%여야 합니다."),
    3: ("3. 분포 학습 (선택)", "실제 응답 엑셀을 올리면 문항별 응답 분포를 학습합니다. 건너뛰면 페르소나 기반 생성만 합니다."),
    4: ("4. 가상 응답 생성", "AI로 가상 응답자를 생성합니다."),
    5: ("5. 결과", ""),
}

if not is_code_verified():
    _render_landing()
    st.stop()

_render_sidebar()
_render_header()
_render_stepper()
_render_leave_guard()

if not projects:
    st.info("프로젝트가 없습니다. 이름을 입력해 첫 프로젝트를 만들어 주세요.")
    with st.form("first_project_form", border=False):
        first_project_name = st.text_input("프로젝트 이름")
        first_create_submitted = st.form_submit_button("첫 프로젝트 만들기")
    if first_create_submitted:
        try:
            first_pid = _project_store.create_project(first_project_name)
        except ValueError as e:
            st.error(str(e))
        else:
            _load_project_to_store(first_pid)
            st.rerun()
    st.stop()

# 상단 메뉴별 페이지 분할 (페이지 상태 방식) — 현재 페이지만 본문에 렌더링한다.
# 5개 단계 + 프로젝트 관리 + 업데이트 기록 = 7개 페이지. 스테퍼는 상단에 유지된다.
_step_fns = {"step1": step1, "step2": step2, "step3": step3, "step4": step4, "step5": step5}
_current_page = store.get("page") or "step1"
if _current_page == "projects":
    _render_projects_page()
elif _current_page == "updates":
    _render_update_log_page()
else:
    _step_no = store.get("step") or 1
    if _current_page.startswith("step"):
        try:
            _step_no = int(_current_page[4:])
        except ValueError:
            pass
    if _step_no not in page_titles:
        _step_no = 1
    st.divider()
    _page_title, _page_caption = page_titles[_step_no]
    st.header(_page_title)
    if _page_caption:
        st.caption(_page_caption)
    _step_fns[f"step{_step_no}"]()

# ----------------------------------------------------------------------------
# 푸터 — 데이터 출처 표기 (CC BY 4.0) + 정직한 한계 고지
# ----------------------------------------------------------------------------
st.divider()
st.caption(NEMOTRON_ATTRIBUTION)
st.caption(DISCLAIMER)
