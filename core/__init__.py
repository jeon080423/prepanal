"""core 패키지 공개 API.

UI(app.py)나 향후 REST API는 이 모듈에서 import한다.
core/* 하위 모듈을 직접 import해도 되지만, 이 __init__이 권장 진입점이다.
"""

from .models import LearnedDist, Persona, Question, Transcript
from .persona import (build_persona_texts, persona_to_text, sample_persona,
                      weighted_choice)
from .providers import (AIProvider, PROVIDER_REGISTRY, UnknownProviderError,
                        auth_status, generate, get_provider, register_provider)
from .question_types import (QUESTION_TYPE_REGISTRY, QuestionType,
                             UnknownQuestionTypeError, get_question_type,
                             register_question_type)
from .services import (ExportService, GenerationService, LearningService,
                       PdfQuestionnaireService, QuestionnaireService)
from .store import MemoryStore, SessionStore, Store
from .strategies import (STRATEGY_REGISTRY, GenerationContext,
                         GenerationStrategy, UnknownStrategyError,
                         get_strategy, register_strategy, select_strategy)

__all__ = [
    # models
    "Question", "LearnedDist", "Persona", "Transcript",
    # question types (plugin point 1)
    "QuestionType", "QUESTION_TYPE_REGISTRY", "register_question_type",
    "get_question_type", "UnknownQuestionTypeError",
    # providers (plugin point 2)
    "AIProvider", "PROVIDER_REGISTRY", "register_provider",
    "get_provider", "UnknownProviderError", "generate", "auth_status",
    # strategies (plugin point 3)
    "GenerationStrategy", "GenerationContext", "STRATEGY_REGISTRY",
    "register_strategy", "get_strategy", "select_strategy",
    "UnknownStrategyError",
    # persona / services / store
    "sample_persona", "persona_to_text", "build_persona_texts",
    "weighted_choice",
    "QuestionnaireService", "LearningService", "GenerationService",
    "ExportService", "PdfQuestionnaireService",
    "Store", "MemoryStore", "SessionStore",
]
