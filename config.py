"""중앙 설정 (config.py)

상수·모델명·기본 분포·경로 등 앱 전역에서 쓰는 값을 한 곳에 모은다.
기능을 확장하거나 기본값을 바꿀 때는 이 파일만 수정한다.
API 키 같은 비밀값은 절대 여기에 두지 않는다 (`.streamlit/secrets.toml` 사용).
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
SAMPLE_Q_PATH = BASE_DIR / "samples" / "sample_questionnaire.json"
SAMPLE_XLSX_PATH = BASE_DIR / "samples" / "sample_real_data.xlsx"

APP_TITLE = "AI 합성응답자 기반 Pre-test"
APP_SUBTITLE = "AI 합성응답자 기반 사전테스트 · Synthetic Respondent Pre-Test"

# ----------------------------------------------------------------------------
# 생성 기본값
# ----------------------------------------------------------------------------
MULTI_SEP = ";"            # 다중선택 응답 구분자
DEFAULT_BATCH_SIZE = 5     # LLM 배치 호출 단위 (응답자 수)
MAX_RESPONDENTS = 200      # 1회 생성 상한
TRANSCRIPT_EXCERPT_CHARS = 2500  # 고정밀 모드에서 쓰는 인터뷰 발췌 길이
# 긴 설문 자동 인식은 입력 문서를 여러 구간으로 나눠 처리한다.
# 겹치는 구간을 두어 문항·보기·분기 설명이 경계에서 잘리지 않도록 한다.
QUESTIONNAIRE_PARSE_CHUNK_CHARS = 10000
QUESTIONNAIRE_PARSE_OVERLAP_CHARS = 1200

LIKERT_OPTIONS = ["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"]

DEFAULT_PERSONA = {
    "성별": {"남성": 50.0, "여성": 50.0},
    "연령대": {"20대": 20.0, "30대": 25.0, "40대": 25.0, "50대": 20.0, "60대 이상": 10.0},
    "지역": {"수도권": 50.0, "영남권": 25.0, "호남권": 15.0, "충청권": 5.0, "강원·제주": 5.0},
    "직업": {"직장인": 45.0, "자영업": 15.0, "학생": 10.0, "주부": 15.0, "기타": 15.0},
}

# ----------------------------------------------------------------------------
# Nemotron 페르소나 출처 (Phase 1 — 실측분포 기반 페르소나)
# 원본: NVIDIA Nemotron-Personas-Korea v1.0 (CC BY 4.0), 한국인 페르소나 100만 건.
# 앱에는 계층추출한 10,000건 발췌본만 내장한다 (원본 1.98GB는 리포지토리 불가).
# ----------------------------------------------------------------------------
NEMOTRON_EXTRACT_PATH = BASE_DIR / "data" / "nemotron_personas_10k.parquet"
NEMOTRON_EXTRACT_SIZE = 10_000
NEMOTRON_EXTRACT_VERSION = "v1.0"
NEMOTRON_EXTRACT_RELEASE_DATE = "2026-04-20"
NEMOTRON_SOURCE_URL = (
    "https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea"
)
NEMOTRON_LICENSE = "CC BY 4.0"
NEMOTRON_FACETS = ("persona", "professional_persona")
# 1인당 프롬프트 프로필 텍스트 상한 (실측: 1인 합성 평균 287자, 최대 430자)
NEMOTRON_TEXT_MAX_CHARS = 550

# 연령 밴딩표 (Nemotron age → 연령대). 60대와 70대 이상을 분리한다 (설계안 §6.4).
# 직접 분포 모드의 DEFAULT_PERSONA는 "60대 이상" 하나를 유지한다 — 기존 출력
# 형식(프로필_연령대 구분)을 바꾸지 않기 위해서다. 밴딩표는 Nemotron 전용으로 쓴다.
AGE_BANDS = [
    ("20대", 19, 29),    # 19세 포함 — 원본은 19세 이상 성인만 포함
    ("30대", 30, 39),
    ("40대", 40, 49),
    ("50대", 50, 59),
    ("60대", 60, 69),
    ("70대 이상", 70, 99),
]

# 권역 매핑표 (시도 → 권역, 설계안 §6.5). Phase 1 쿼터는 권역 단위다.
REGION_MAP = {
    "수도권": ["서울", "경기", "인천"],
    "영남권": ["부산", "대구", "울산", "경상북", "경상남"],
    "호남권": ["광주", "전북", "전라남"],
    "충청권": ["대전", "세종", "충청북", "충청남"],
    "강원·제주": ["강원", "제주"],
}

# 페르소나 출처 선택지 (기본값: 직접 분포 설정 — 기존 동작 유지)
PERSONA_SOURCES = {
    "manual": "직접 분포 설정",
    "nemotron": "Nemotron 실측분포 (한국 인구 분포 기반)",
}

# CC BY 4.0 출처 표기 (앱 푸터·엑셀 방법론 시트·Word 리포트·README 공통 문구)
NEMOTRON_ATTRIBUTION = (
    "페르소나: Nemotron-Personas-Korea (NVIDIA, CC BY 4.0), "
    "https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea"
)

# ----------------------------------------------------------------------------
# AI Provider 스펙 (인스턴스는 core/providers.py 의 레지스트리에서 생성)
# 새 AI 추가 = AIProvider 서브클래스 1개 + 아래 PROVIDER_SPECS에 한 줄.
# ----------------------------------------------------------------------------
PROVIDER_SPECS = {
    "gemini": {
        "kind": "gemini",
        "label": "Gemini",
        "default_model": "gemini-2.5-flash-lite",
        "fallback_model": "gemini-2.5-flash",
        "cli": "~/workspace/skills/gemini/bin/gemini",
        "secret_name": "GEMINI_API_KEY",
        # 모델 목록 (2026-10 검증, 출처: Google AI 공식 문서
        # https://ai.google.dev/gemini-api/docs/models)
        "models": [
            {
                "id": "gemini-2.5-flash-lite",
                "label": "Gemini 2.5 Flash-Lite (기본값)",
                "description": "2.5 패밀리 중 가장 빠르고 저렴. 대량 생성·파일럿 테스트에 적합. 무료 등급 지원.",
                "source": "Google AI 공식 문서 (ai.google.dev/gemini-api/docs/models)",
            },
            {
                "id": "gemini-2.5-flash",
                "label": "Gemini 2.5 Flash",
                "description": "속도와 성능의 균형. 추론이 필요한 작업에 적합한 가성비 모델.",
                "source": "Google AI 공식 문서 (ai.google.dev/gemini-api/docs/models)",
            },
            {
                "id": "gemini-3.1-pro-preview",
                "label": "Gemini 3.1 Pro (Preview)",
                "description": "차세대 Pro 모델. 복잡한 추론·코딩에 강함. Preview 버전이라 스펙이 변경될 수 있음.",
                "source": "Google AI 공식 문서 (ai.google.dev/gemini-api/docs/models)",
            },
            {
                "id": "gemini-3-flash-preview",
                "label": "Gemini 3 Flash (Preview)",
                "description": "차세대 Flash 모델. 무료 티어 제공. Preview 버전이라 스펙이 변경될 수 있음.",
                "source": "Google AI 공식 문서 (ai.google.dev/gemini-api/docs/models)",
            },
        ],
    },
    "groq": {
        "kind": "groq",
        "label": "Groq",
        "default_model": "llama-3.3-70b-versatile",
        "fallback_model": "llama-3.1-8b-instant",
        "secret_name": "GROQ_API_KEY",
        "models": [
            {
                "id": "llama-3.3-70b-versatile",
                "label": "Llama 3.3 70B (기본값)",
                "description": "고품질 오픈 모델. 무료 티어 제공. 한국어 지원.",
                "source": "Groq 공식 문서 (console.groq.com/docs/models)",
            },
            {
                "id": "llama-3.1-8b-instant",
                "label": "Llama 3.1 8B Instant",
                "description": "초고속 경량 모델. 대량 생성에 적합. 무료 티어 제공.",
                "source": "Groq 공식 문서 (console.groq.com/docs/models)",
            },
        ],
    },
}

# ----------------------------------------------------------------------------
# 생성 전략 선택 우선순위 (앞에 있을수록 먼저 매칭)
# 새 전략 추가 = GenerationStrategy 서브클래스 1개 + 아래 리스트에 이름 추가.
# ----------------------------------------------------------------------------
STRATEGY_ORDER = ["learned", "precision", "elicitation", "open_ended"]

# ----------------------------------------------------------------------------
# 방법론 고지 (결과 화면·엑셀에 고정 표시 — 정직한 한계 고지)
# ----------------------------------------------------------------------------
DISCLAIMER = (
    "합성응답자는 탐색적·방향성 리서치용이며 인간 실측을 대체할 수 없음 "
    "(Pew: 평균 12%p 오차)"
)

METHODOLOGY_NOTE = """[방법론 안내]
1. 본 도구는 GPU 파인튜닝(모델 가중치 학습)을 하지 않습니다.
   '분포 학습'은 업로드된 실제 응답 데이터의 문항별 경험분포를 추출하는 것이며,
   '페르소나 시뮬레이션'은 LLM에 페르소나 프로필을 주고 가상 응답을 생성하게 하는 방식입니다.
2. 선택형 문항은 Prolific 연구팀(Gordon et al., 2026 pre-print) 방식에 따라,
   페르소나별 강제 단일 응답이 아닌
   보기별 선택 확률분포를 LLM으로부터 추출한 뒤 샘플링합니다.
   (연구 보고: 분포 오차 35~48% 감소)
3. 고정밀 모드(인터뷰 기반)는 스탠포드 Generative Agents 연구에서 정규화 정확도 83%로
   가장 높았으나, 1인당 약 2시간의 인터뷰가 필요합니다.
4. 한계: """ + DISCLAIMER + """
5. 참고: Gemini Pro 기본값은 1,000개 설문 공개 벤치마크에서 표준 LLM 중 최고(67%, GPT-5 62% 대비)를 기록한 리서치를 반영한 것입니다.
6. 페르소나 출처가 'Nemotron 실측분포'인 경우, 가상 응답자의 인구통계 구성은
   NVIDIA의 Nemotron-Personas-Korea(v1.0, CC BY 4.0)를 따릅니다. 이 데이터셋은
   KOSIS·대법원·국민건강보험공단·농촌경제연구원·네이버 클라우드의 실제 통계를
   씨앗으로 합성한 한국인 페르소나 100만 건이며, 본 도구는 이 가운데
   계층추출한 10,000건 발췌본을 사용합니다. 장점: (1) 손으로 정한 가중치가
   아니라 실측 분포에서 샘플링하므로 패널 구성이 모집단을 닮았다는 근거를
   제시할 수 있습니다. (2) 요약·직업·가족·식생활 등 풍부한 서술을 프롬프트
   조건으로 써서 가상 응답자의 맥락이 입체적입니다. (3) 페르소나 단계가 LLM
   호출에서 로컬 조회로 바뀌어 이 단계의 타임아웃이 원천적으로 차단됩니다.
   (4) 252개 시군구 기반 데이터라 정밀한 지역 구성이 가능합니다
   (Phase 1은 권역 단위 쿼터). 단, 패널 구성이 모집단을 닮았다는 것이
   응답이 더 정확하다는 뜻은 아닙니다.
7. Nemotron 출처의 한계: 데이터셋은 성인(만 19세 이상)만 포함하므로 청소년
   대상 조사에는 쓸 수 없습니다. 직업 배정 시 성별·소득·학력 등을 독립으로
   가정해 만들어졌기 때문에, 세부 교차 특성(예: 특정 성별×특정 전공의 직업
   분포)은 실제와 다를 수 있습니다. 페르소나 서술 속 이름은 합성 데이터의
   일부이며 실존 인물과 무관합니다. 응답자 식별은 pid(P001…)로만 표시합니다."""
