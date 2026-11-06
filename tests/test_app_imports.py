"""app.py가 가져오는 이름들이 config/core에 실제로 존재하는지 검증한다.

2026-11-06 배포 장애 회귀 테스트: Phase 1에서 app.py가
`nemotron_default_quotas`를 core에서 가져오도록 바뀌었으나
core/__init__.py에 export가 빠져 있어, pytest 40개 통과에도 불구하고
Streamlit 배포본이 ImportError로 뜨지 않았다. app.py 자체는 streamlit이
없으면 import할 수 없으므로, 가져오는 이름 목록을 그대로 가져와 검증한다.
app.py의 import 목록이 바뀌면 이 테스트도 함께 바꿀 것.
"""


def test_app_config_import_surface():
    from config import (AGE_BANDS, APP_TITLE, DEFAULT_PERSONA, DISCLAIMER,
                        METHODOLOGY_NOTE, NEMOTRON_ATTRIBUTION,
                        NEMOTRON_EXTRACT_RELEASE_DATE, NEMOTRON_EXTRACT_SIZE,
                        NEMOTRON_EXTRACT_VERSION, PERSONA_SOURCES, REGION_MAP,
                        SAMPLE_XLSX_PATH)

    assert AGE_BANDS and APP_TITLE and DEFAULT_PERSONA
    assert DISCLAIMER and METHODOLOGY_NOTE and NEMOTRON_ATTRIBUTION
    assert NEMOTRON_EXTRACT_RELEASE_DATE and NEMOTRON_EXTRACT_SIZE
    assert NEMOTRON_EXTRACT_VERSION and PERSONA_SOURCES and REGION_MAP
    assert SAMPLE_XLSX_PATH


def test_app_core_import_surface():
    from core import (QUESTION_TYPE_REGISTRY, PROVIDER_REGISTRY, ExportService,
                      GenerationService, LearningService,
                      PdfQuestionnaireService, QuestionnaireService,
                      SessionStore, auth_status, get_provider,
                      get_question_type, ProjectStore, PROJECT_STATE_KEYS,
                      SheetsProjectStore, load_nemotron_extract,
                      nemotron_default_quotas)

    assert QUESTION_TYPE_REGISTRY and PROVIDER_REGISTRY
    assert ExportService and GenerationService and LearningService
    assert PdfQuestionnaireService and QuestionnaireService
    assert SessionStore and auth_status and get_provider and get_question_type
    assert ProjectStore and PROJECT_STATE_KEYS and SheetsProjectStore
    assert callable(load_nemotron_extract)
    assert callable(nemotron_default_quotas)
