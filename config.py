"""중앙 설정 (config.py)

상수·모델명·기본 분포·경로 등 앱 전역에서 쓰는 값을 한 곳에 모은다.
기능을 확장하거나 기본값을 바꿀 때는 이 파일만 수정한다.
API 키 같은 비밀값은 절대 여기에 두지 않는다 (`.streamlit/secrets.toml` 사용).
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
SAMPLE_Q_PATH = BASE_DIR / "samples" / "sample_questionnaire.json"
SAMPLE_XLSX_PATH = BASE_DIR / "samples" / "sample_real_data.xlsx"

APP_TITLE = "가상응답자 스튜디오"
APP_SUBTITLE = "Synthetic Respondent Studio"

# ----------------------------------------------------------------------------
# 생성 기본값
# ----------------------------------------------------------------------------
MULTI_SEP = ";"            # 다중선택 응답 구분자
DEFAULT_BATCH_SIZE = 5     # LLM 배치 호출 단위 (응답자 수)
MAX_RESPONDENTS = 200      # 1회 생성 상한
TRANSCRIPT_EXCERPT_CHARS = 2500  # 고정밀 모드에서 쓰는 인터뷰 발췌 길이

LIKERT_OPTIONS = ["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"]

DEFAULT_PERSONA = {
    "성별": {"남성": 50.0, "여성": 50.0},
    "연령대": {"20대": 20.0, "30대": 25.0, "40대": 25.0, "50대": 20.0, "60대 이상": 10.0},
    "지역": {"수도권": 50.0, "영남권": 25.0, "호남권": 15.0, "충청권": 5.0, "강원·제주": 5.0},
    "직업": {"직장인": 45.0, "자영업": 15.0, "학생": 10.0, "주부": 15.0, "기타": 15.0},
}

# ----------------------------------------------------------------------------
# AI Provider 스펙 (인스턴스는 core/providers.py 의 레지스트리에서 생성)
# 새 AI 추가 = AIProvider 서브클래스 1개 + 아래 PROVIDER_SPECS에 한 줄.
# ----------------------------------------------------------------------------
PROVIDER_SPECS = {
    "gemini": {
        "kind": "gemini",
        "label": "Gemini",
        # Prolific/벤치마크 리서치 기준 기본값: Pro 계열 (표준 LLM 중 최고 67%)
        # 무료 등급에서 할당량 오류(429)가 나면 gemini-3.6-flash로 변경
        "default_model": "gemini-2.5-flash-lite",
        "fallback_model": "gemini-2.0-flash",
        "cli": "~/workspace/skills/gemini/bin/gemini",
        "secret_name": "GEMINI_API_KEY",
    },
    "openai": {
        "kind": "openai",
        "label": "OpenAI (ChatGPT)",
        "default_model": "gpt-4o-mini",
        "fallback_model": "",
        "cli": "~/workspace/skills/openai/bin/chatgpt",
        "secret_name": "OPENAI_API_KEY",
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
2. 선택형 문항은 Prolific 2026 연구 방식에 따라, 페르소나별 강제 단일 응답이 아닌
   보기별 선택 확률분포를 LLM으로부터 추출한 뒤 샘플링합니다.
   (연구 보고: 분포 오차 35~48% 감소)
3. 고정밀 모드(인터뷰 기반)는 스탠포드 Generative Agents 연구에서 정규화 정확도 83%로
   가장 높았으나, 1인당 약 2시간의 인터뷰가 필요합니다.
4. 한계: """ + DISCLAIMER + """
5. 참고: Gemini Pro 기본값은 1,000개 설문 공개 벤치마크에서 표준 LLM 중 최고(67%, GPT-5 62% 대비)를 기록한 리서치를 반영한 것입니다."""
