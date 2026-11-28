"""서비스 레이어 (core/services.py)


app.py(UI)는 이 서비스들만 호출한다. 비즈니스 로직은 전부 여기(와 core/*)에.

향후 REST API를 만들 때도 같은 서비스를 재사용하면 된다.

"""
from __future__ import annotations

import hashlib
import io
import json
import random
import re
import unicodedata
import pandas as pd
from config import (DEFAULT_BATCH_SIZE, METHODOLOGY_NOTE,
                    NEMOTRON_EXTRACT_SIZE, NEMOTRON_EXTRACT_VERSION,
                    PERSONA_SOURCES, QUESTIONNAIRE_PARSE_CHUNK_CHARS,
                    QUESTIONNAIRE_PARSE_OVERLAP_CHARS, SAMPLE_Q_PATH,
                    STRATEGY_ORDER)
from .persona import (load_nemotron_extract, sample_nemotron_personas,
                      sample_persona)
from .question_types import get_question_type
from .strategies import (GenerationContext, get_strategy, select_strategy)

# ----------------------------------------------------------------------------
# 설문 문항 관리
# ----------------------------------------------------------------------------
class QuestionnaireService:
    @staticmethod
    def new_question_id(questions: list[dict]) -> str:
        nums = [int(q["id"][1:]) for q in questions
                if q["id"].startswith("Q") and q["id"][1:].isdigit()]
        return f"Q{(max(nums) + 1) if nums else 1}"

    @staticmethod
    def question_summary(q: dict) -> str:
        qtype = get_question_type(q["type"])
        base = f"{q['id']}. {q['text']} ({qtype.label})"
        if qtype.needs_options and q.get("options"):
            base += " — " + ", ".join(qtype.options_for(q))
        if q.get("branching"):
            base += " ⤷분기"
        return base

    @staticmethod
    def validate_question(q: dict) -> list[str]:
        """문항 유효성 검사. 오류 메시지 목록 (빈 리스트 = 유효)."""
        return get_question_type(q["type"]).validate(q)

    @staticmethod
    def load_sample_questions() -> list[dict]:
        return json.loads(SAMPLE_Q_PATH.read_text(encoding="utf-8"))

# ----------------------------------------------------------------------------
# 분포 학습
# ----------------------------------------------------------------------------
class LearningService:
    @staticmethod
    def _match_column(df: pd.DataFrame, q: dict) -> str | None:
        """엑셀 컬럼을 문항에 매칭: 문항 텍스트 일치 → Q번호 일치 순."""
        cols = list(df.columns)
        if q["text"] in cols:
            return q["text"]
        if q["id"] in cols:
            return q["id"]
        qid_lower = q["id"].lower()
        for c in cols:
            if str(c).strip().lower() == qid_lower:
                return c
        return None

    def learn_distributions(self, df: pd.DataFrame,
                            questions: list[dict]) -> dict:
        """실제 응답 엑셀에서 문항별 응답 분포를 추출한다.


        Returns: {qid: {"matched_col": str, "type": str,
                         "dist": {option: prob}, "n": int}}

        주관식은 분포 없이 n만 기록한다.

        """
        learned: dict = {}
        for q in questions:
            qtype = get_question_type(q["type"])
            col = self._match_column(df, q)
            if col is None:
                continue
            series = df[col].dropna()
            n = len(series)
            if n == 0:
                continue
            dist = qtype.learn_distribution(q, series)
            if dist is None:
                continue
            learned[q["id"]] = {
                "matched_col": str(col),
                "type": q["type"],
                "dist": {str(k): float(v) for k, v in dist.items()},
                "n": n,
            }

        return learned


# ----------------------------------------------------------------------------

# 분기(branching) 적용 — 생성 단계 스킵 로직

# ----------------------------------------------------------------------------

# branching 필드는 자유 서술 문자열이다(예: "문3에서 ① 응답 시 문5로 이동",
# "문2의 ② 응답자만 응답"). 파싱·편집 단계에서는 표시만 되고 생성 단계에서는
# 전혀 쓰이지 않았다 — 그래서 Q12(동반인원)가 공란인데도 Q13(동반 유형)에
# 전원이 응답하는 오류가 실제 데이터로 확인됐다.
#
# 생성은 전략별 그룹으로 나뉘어 문항 단위로 독립 수행되므로(각 전략은 branching을
# 모른다), 분기는 모든 전략 결과가 합쳐지는 단일 지점인
# GenerationService.generate_responses()에서 생성 직후 적용한다. 전략 종류
# (학습/분포추출/고정밀/주관식)와 무관하게 동작하고, LLM 프롬프트는 건드리지
# 않는다. 해당 없음 문항은 공란("")으로 둔다 — 문항별 유효 n은 공란 제외 건수다.
#
# 지원하는 서술 패턴 (파싱이 안 되면 해당 문항은 건드리지 않는다 — 오적용
# 방지를 위해 모호한 조건은 무시하는 것이 원칙):
#   (A) 종속 조건 — "문12의 응답자만 응답", "문2의 ② 응답자만 응답",
#       "문3에서 ① 응답 시 응답": 해당 문항은 조건이 맞을 때만 답한다.
#       트리거가 없으면 "원문항에 답함(공란 아님)"이 조건이다.
#       부정 표현은 트리거 조건절 안에 있을 때만 극성을 뒤집는다
#       ("Q12가 없음이면 스킵" → Q12가 없음이 아닐 때만 답한다).
#       "…만 응답" 같은 긍정 한정이 있으면 뒤집지 않고
#       ("문12의 ② 응답자만 응답, 나머지는 스킵"의 "스킵"은 조건절이 아니다),
#       자기 문항 번호를 먼저 쓴 서술("Q13은 문12의 응답자만 응답")은
#       자기 언급을 건너뛰고 다음 언급을 원문항으로 본다.
#       파싱에 실패한 branching은 last_branch_meta["파싱 실패"]에 남긴다.
#   (B) 건너뛰기 — "문3에서 ① 응답 시 문5로 이동": 원문항의 답이 트리거와
#       맞으면 원문항과 목표 문항 사이의 문항들을 공란으로 둔다.
#       트리거가 없는 "문5로 이동"은 적용하지 않는다 (무조건 건너뛰기로
#       오독할 수 있어 무시한다).
# ----------------------------------------------------------------------------

_CIRCLED_DIGITS = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5,
                   "⑥": 6, "⑦": 7, "⑧": 8, "⑨": 9, "⑩": 10}

_BLANK_ANSWER_STRS = {"", "nan", "none", "null", "nat"}

# "없음" 계열 표현 — "없음/없다/없는/없어"는 같은 뜻으로 본다.
_NONE_WORDS = {"없음", "없다", "없는", "없어", "없으", "없습니다",
               "없어요", "해당없음", "해당 없음", "없"}

# 부정(스킵) 표현 — 종속 조건의 극성을 뒤집는다.
_NEGATIVE_MARKERS = ("스킵", "건너뛰", "제외", "생략", "응답하지 않",
                     "답하지 않", "넘어갑", "건너뜀")

# 긍정 한정 표현 — "…만 응답"은 조건절에 부정어가 붙어도 극성을 뒤집지 않는다.
# ("문12의 ② 응답자만 응답, 나머지는 스킵"의 "스킵"은 '나머지'를 설명할 뿐,
#  해당 문항이 답해야 할 조건 자체가 아니다.)
_POSITIVE_LIMIT_RE = re.compile(r"만\s*(?:응답|답|해당|적용)")

# 조건 연결어 — 부정어가 조건절 안에 있을 때만 극성을 뒤집는다.
# ("Q12가 없음이면 스킵"의 "이면"은 조건절, "문12의 응답자만 응답, 나머지는
#  스킵"의 "스킵" 앞에는 조건 연결어가 없어 뒤집지 않는다.)
_CONDITION_CONNECTORS = ("이면", "으면", "라면", "인 경우", "일 경우",
                         "응답 시", "응답하면", "인 때", "일 때")


def _is_negative_condition(text: str) -> bool:
    """종속 조건의 극성이 부정인지 (트리거 조건절 안의 부정 표현만 본다).

    "Q12가 없음이면 스킵" → "스킵" 앞 조건절에 "이면"이 있어 True.
    "문12의 ② 응답자만 응답, 나머지는 스킵" → "만 응답" 긍정 한정이
    있어 False (서술 전체를 검사하면 정반대로 뒤집히는 오적용).
    조건절과 무관한 위치의 부정어(연결어 없음)도 False.
    """
    if _POSITIVE_LIMIT_RE.search(text):
        return False
    for marker in _NEGATIVE_MARKERS:
        idx = text.find(marker)
        if idx < 0:
            continue
        clause = text[:idx]
        if any(conn in clause for conn in _CONDITION_CONNECTORS):
            return True
    return False


def _is_blank_answer(value) -> bool:
    """공란 판정: None·빈 문자열·공백만·'nan'/'None' 등 결측 표기."""
    if value is None:
        return True
    return str(value).strip().lower() in _BLANK_ANSWER_STRS


def _normalize_answer(value) -> str:
    """답변 비교용 정규화: NFKC + 공백 정리."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def _canonical_none(value: str) -> str:
    """'없음' 계열 표현을 '없음'으로 통일한다. 아니면 정규화한 값 그대로."""
    s = _normalize_answer(value)
    return "없음" if s in _NONE_WORDS else s


def _is_zero_number(value: str) -> bool:
    """'0'·'0명'처럼 0인 숫자·금액 응답인지 (동반인원 0명 = 없음)."""
    s = str(value).strip().replace(",", "").replace(" ", "")
    for unit in ("명", "원", "개", "회", "번", "일", "시간", "분", "초", "점", "건"):
        if s.endswith(unit) and len(s) > len(unit):
            s = s[: -len(unit)]
            break
    try:
        return float(s) == 0.0
    except ValueError:
        return False


def _answer_matches(answer, trigger: str, qtype: str | None = None) -> bool:
    """원문항 답변이 트리거와 맞는지.

    단일선택·리커트 문항은 완전일치만 쓴다 — "그렇다"가 "매우 그렇다"에,
    "그렇지 않다"가 "전혀 그렇지 않다"에 부분일치로 매칭되는 오판을 막는다.
    다중선택 문항(type == "multiple")만 구분자(, / ;) 단위 토큰 포함 검사를
    쓴다 ("A, B" 형태의 답에서 트리거 "A"가 토큰 하나와 완전히 일치하면 맞음).
    "없음" 계열은 "없음/없다/없는"과 "0명"을 같은 뜻으로 본다.
    """
    ans = _normalize_answer(answer)
    trg = _normalize_answer(trigger)
    if not ans or not trg:
        return False
    ans_none = _canonical_none(ans) == "없음" or _is_zero_number(ans)
    trg_none = _canonical_none(trg) == "없음" or _is_zero_number(trg)
    if ans_none and trg_none:
        return True
    if ans == trg:
        return True
    if qtype == "multiple":
        tokens = [t.strip() for t in re.split(r"[,;]", ans)]
        return trg in [_normalize_answer(t) for t in tokens if t]
    return False


def _extract_trigger(branching: str, source_q: dict | None) -> str | None:
    """branching 서술에서 트리거(원문항의 특정 보기/답)를 뽑는다.

    우선순위: 동그라미 숫자(①→원문항 보기 1번, 서술 등장 순서) → 따옴표 안 문구 →
    원문항 보기 문구가 서술에 그대로 등장 → "없음 응답 시"처럼
    '응답 시/인 경우/이면' 바로 앞 단어. 없으면 None
    (트리거 없음 = '원문항에 답함'이 조건).
    동그라미 숫자가 있는데 보기와 연결이 안 되면 None을 반환한다
    (호출부가 _has_unresolvable_trigger()로 규칙 전체를 무시한다).
    """
    text = _normalize_answer(branching)
    if source_q:
        options = [str(o) for o in source_q.get("options") or []]
        circled = [(text.index(ch), num)
                   for ch, num in _CIRCLED_DIGITS.items() if ch in text]
        if circled:
            _, num = min(circled)
            if 1 <= num <= len(options):
                return options[num - 1]
            return None
        quoted = re.search(r"[\"'「]([^\"'」]+)[\"'」]", text)
        if quoted:
            return quoted.group(1).strip()
        for opt in sorted(options, key=len, reverse=True):
            if opt and _normalize_answer(opt) in text:
                return opt
    word_m = re.search(
        r"([가-힣A-Za-z0-9]+)\s*(?:응답\s*시|응답\s*하면|인\s*경우|이면|일\s*경우)",
        text)
    if word_m:
        return word_m.group(1)
    return None


def _has_unresolvable_trigger(branching: str, source_q: dict | None) -> bool:
    """트리거를 언급했으나 보기와 연결할 수 없는지 (예: ①인데 보기 없음).

    이 경우 규칙 전체를 무시한다 — 트리거 없는 조건으로 격하하면
    '① 응답자만'이 '아무나 답함'이 되어 오적용된다.
    """
    text = _normalize_answer(branching)
    options = [str(o) for o in (source_q or {}).get("options") or []]
    for ch, num in _CIRCLED_DIGITS.items():
        if ch in text and not (1 <= num <= len(options)):
            return True
    return False


def _parse_branch_rule(q: dict,
                       questions_by_id: dict[str, dict]) -> dict | None:
    """한 문항의 branching 서술을 규칙 dict로 바꾼다. 파싱이 안 되면 None.

    {"kind": "requires" | "requires_not" | "skip_to",
     "source": 원문항 id, "trigger": 트리거 문자열 또는 None,
     "target": 목표 문항 id (skip_to만)}
    """
    branching = str(q.get("branching") or "").strip()
    if not branching:
        return None
    text = _normalize_answer(branching)
    qid = q.get("id", "")

    # (B) 건너뛰기: "문5로 이동" / "문5로 넘어감" / "문5로 건너뛰기"
    move_m = re.search(
        r"(?:문|Q)\s*(\d+)\s*(?:번\s*)?(?:으로|로)\s*(?:이동|넘어|건너뛰)", text)
    if move_m:
        target = f"Q{int(move_m.group(1))}"
        if target in questions_by_id and target != qid:
            if _has_unresolvable_trigger(text, q):
                return None
            trigger = _extract_trigger(text, q)
            if trigger is None:
                return None  # 트리거 없는 건너뛰기는 적용하지 않음
            return {"kind": "skip_to", "source": qid, "trigger": trigger,
                    "target": target, "source_type": q.get("type", "")}
        return None

    # (A) 종속 조건: 언급된 문항 중 자기 자신이 아닌 첫 번째가 원문항.
    # "Q13은 문12의 응답자만 응답"처럼 자기 번호를 먼저 쓰면 첫 매치가
    # 자기 자신이므로 건너뛰고 다음 언급을 찾는다 (첫 매치가 자기 자신이면
    # 규칙이 소실되던 오적용). 자기 자신이 아닌 첫 언급이 목록에 없으면
    # 규칙을 만들지 않는다.
    for ref_m in re.finditer(r"(?:문|Q)\s*(\d+)", text):
        source = f"Q{int(ref_m.group(1))}"
        if source == qid:
            continue
        if source not in questions_by_id:
            return None
        source_q = questions_by_id[source]
        if _has_unresolvable_trigger(text, source_q):
            return None
        trigger = _extract_trigger(text, source_q)
        kind = ("requires_not"
                if _is_negative_condition(text)
                else "requires")
        return {"kind": kind, "source": source, "trigger": trigger,
                "source_type": source_q.get("type", "")}
    return None


def _describe_branch_rule(qid: str, rule: dict) -> str:
    """규칙을 사람이 읽을 수 있는 한 줄 설명으로 바꾼다 (메타 표시용)."""
    if rule["kind"] == "skip_to":
        return (f"{rule['source']}='{rule.get('trigger')}'이면 "
                f"{rule['source']}→{rule['target']} 사이 문항 공란")
    cond = (f"{rule['source']}='{rule.get('trigger')}'"
            if rule.get("trigger") else f"{rule['source']}에 답함")
    if rule["kind"] == "requires_not":
        return f"{qid}는 {cond}가 아닐 때만 응답"
    return f"{qid}는 {cond}일 때만 응답"


def _compile_branch_rules(questions: list[dict]) -> dict:
    """문항 목록의 branching 서술을 생성 시 적용할 규칙으로 컴파일한다.

    Returns: {"requires": {qid: rule}, "skip_to": {source_qid: rule},
              "summaries": {qid: 설명},
              "unparsed": {qid: 파싱에 실패한 branching 원문}}
    — 파싱이 안 된 문항은 제외된다. "unparsed"는 적용 안 된 분기를
    last_branch_meta에 경고로 남기기 위한 것으로, 생성 후 확인용이다.
    """
    by_id = {q.get("id"): q for q in questions if q.get("id")}
    requires: dict[str, dict] = {}
    skip_to: dict[str, dict] = {}
    summaries: dict[str, str] = {}
    unparsed: dict[str, str] = {}
    for q in questions:
        rule = _parse_branch_rule(q, by_id)
        if rule is None:
            branching = str(q.get("branching") or "").strip()
            if branching:
                unparsed[q["id"]] = branching
            continue
        if rule["kind"] == "skip_to":
            skip_to[q["id"]] = rule
        else:
            requires[q["id"]] = rule
        summaries[q["id"]] = _describe_branch_rule(q["id"], rule)
    return {"requires": requires, "skip_to": skip_to, "summaries": summaries,
            "unparsed": unparsed}


def _branch_condition_holds(rule: dict, answers: dict) -> bool:
    """종속 조건이 맞는지: 원문항에 답했고, 트리거가 있으면 답이 트리거와 맞음."""
    src_ans = answers.get(rule["source"], "")
    if _is_blank_answer(src_ans):
        return False
    trigger = rule.get("trigger")
    if trigger is None:
        return True
    return _answer_matches(src_ans, trigger, rule.get("source_type"))


def _branch_rule_applies(rule: dict, answers: dict) -> bool:
    """해당 문항에 답해야 하는지 (requires_not은 조건의 반대)."""
    if rule["kind"] == "requires_not":
        return not _branch_condition_holds(rule, answers)
    return _branch_condition_holds(rule, answers)


def _skip_trigger_matches(rule: dict, answers: dict) -> bool:
    """skip_to 규칙: 원문항 답이 트리거와 맞아 건너뛰기가 발동하는지."""
    src_ans = answers.get(rule["source"], "")
    if _is_blank_answer(src_ans):
        return False
    trigger = rule.get("trigger")
    if trigger is None:
        return False  # 트리거 없는 건너뛰기는 발동하지 않음
    return _answer_matches(src_ans, trigger, rule.get("source_type"))


def _questions_between(order: list[str], source: str, target: str) -> list[str]:
    """문항 순서에서 source와 target 사이에 있는 문항 id 목록 (역방향이면 빈 목록)."""
    try:
        i, j = order.index(source), order.index(target)
    except ValueError:
        return []
    if i >= j:
        return []
    return order[i + 1:j]


# ----------------------------------------------------------------------------
# 배치 중간 저장 / 재개 (체크포인트) — 워커 D
#
# generate_responses()는 배치 단위(DEFAULT_BATCH_SIZE=5)로 돌다가 LLM 실패·
# 세션 중단 등으로 중간에 끊기면 처음부터 다시 돌아야 했다 (300명=60배치).
# 체크포인트는 배치 완료 시점마다 {완료 배치 수, 지금까지의 행, 지문}을
# 스냅샷으로 남기고, 같은 시드·같은 조건으로 재진입하면 완료 배치를
# 건너뛰어 마지막 완료 배치부터 이어간다.
#
# 재개 방식 — 재생성 스킵(replay-skip):
#   완료 배치는 같은 코드 경로로 다시 실행하되 행은 버리고 체크포인트 행을
#   그대로 쓴다. 재생성은 페르소나 샘플링·전략 generate()의 난수 소비를
#   원래 순서대로 재현하기 위해서만 필요하고, 그 소비량은 LLM 출력값과
#   무관하다 (가중 샘플링은 호출당 uniform 1회, 다중선택은 보기 수만큼
#   random() 1회씩, 고정밀은 페르소나당 choice 1회 — 값에 따라 횟수가
#   달라지는 경로는 없다). 그래서 실제 LLM이 같은 프롬프트에 다른 분포를
#   돌려줘도 (a) 완료 배치 행은 체크포인트 그대로 보존되고, (b) 그 뒤
#   배치의 난수 순서가 원래 실행과 일치해 최종 DataFrame이 중단 없이
#   한 번에 돌린 결과와 행 단위로 동일하다.
#   재생성 중 캐시에 프롬프트가 있으면 LLM 호출이 없고, 없으면 재호출이
#   발생한다. 재호출 건수는 cache_misses에 기록해 last_resume_info로
#   보고한다 (캐시 커버 여부 확인 — 최소 조건 (1)).
#   분기(branching) 적용은 난수를 쓰지 않으므로 재생성 시 생략하고,
#   체크포인트에 저장된 공란 건수를 그대로 이어받는다.
#
# 저장소: 새 외부 저장소는 쓰지 않는다. 체크포인트는 세션 상태
# (app.py의 store "gen_checkpoint")에 두고, LLM 캐시는 기존 gen_cache
# dict를 그대로 쓴다. seed가 None이면 중단 전 난수 순서를 재현할 수
# 없어 재개가 불가하다 — ValueError를 낸다.
# ----------------------------------------------------------------------------
_CHECKPOINT_VERSION = 1


def _stable_repr(obj) -> str:
    """지문용 안정적 문자열화: dict 키 순서·tuple 키에 무관하게 동일 출력."""
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: repr(kv[0]))
        return "{" + ",".join(
            f"{_stable_repr(k)}:{_stable_repr(v)}" for k, v in items) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_stable_repr(v) for v in obj) + "]"
    return repr(obj)


def _sha_short(obj) -> str:
    return hashlib.sha256(_stable_repr(obj).encode("utf-8")).hexdigest()


def build_checkpoint_fingerprint(
    questions: list[dict],
    n: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int | None = None,
    provider_id: str = "",
    model: str = "",
    persona_source: str = "manual",
    persona_dims: dict | None = None,
    learned: dict | None = None,
    transcripts: list[dict] | None = None,
    use_precision: bool = False,
    nemotron_quotas: dict | None = None,
    nemotron_count_quotas: dict | None = None,
    nemotron_region_level: str = "region",
    nemotron_facets=None,
) -> dict:
    """재개 가능 여부를 판정하는 생성 조건 지문.

    문항 정의(내용·유형·보기·분기), 응답자 수·배치 크기·seed·모델·
    페르소나 출처·페르소나 차원(전체)·학습 분포(문항별 분포 값 전체)·
    트랜스크립트(이름+본문 전체)·Nemotron 쿼터·facets가 하나라도
    다르면 다른 실행으로 보고 이어서 진행하지 않는다.
    (C-1 수정: persona_dims·학습 분포 값·트랜스크립트 본문·facets는
    이전 지문에서 빠져 있어, 조건이 바뀐 채로 이어서 진행하면 앞 배치는
    구 조건·뒷 배치는 신 조건으로 섞이는 조용한 오염이 생겼다.)
    """
    qsig = [{"id": q.get("id"), "text": q.get("text"), "type": q.get("type"),
             "options": list(q.get("options") or []),
             "branching": q.get("branching") or ""}
            for q in (questions or [])]
    n_int = int(n)
    bs_int = int(batch_size)
    total_batches = (n_int + bs_int - 1) // bs_int if n_int > 0 else 0
    return {
        "version": _CHECKPOINT_VERSION,
        "n": n_int,
        "batch_size": bs_int,
        "total_batches": total_batches,
        "seed": seed,
        "provider_id": str(provider_id),
        "model": str(model),
        "persona_source": str(persona_source),
        "use_precision": bool(use_precision),
        "learned_qids": sorted((learned or {}).keys()),
        "learned_sha": _sha_short(learned or {}),
        "persona_dims_sha": _sha_short(persona_dims or {}),
        "transcript_names": [str(t.get("name", ""))
                             for t in (transcripts or [])],
        "transcripts_sha": _sha_short(transcripts or []),
        "question_sha": _sha_short(qsig),
        "nemotron_sha": _sha_short({
            "quotas": nemotron_quotas,
            "count_quotas": nemotron_count_quotas,
            "region_level": nemotron_region_level,
            "facets": sorted(nemotron_facets or []),
        }),
    }


def _checkpoint_problems(checkpoint, fingerprint: dict) -> list[str]:
    """체크포인트가 지문과 맞지 않는 이유 목록 (빈 목록 = 재개 가능)."""
    if not isinstance(checkpoint, dict):
        return ["체크포인트가 dict가 아닙니다"]
    problems: list[str] = []
    if checkpoint.get("version") != _CHECKPOINT_VERSION:
        problems.append(
            f"체크포인트 버전 불일치: {checkpoint.get('version')!r}")
    stored_fp = checkpoint.get("fingerprint")
    if not isinstance(stored_fp, dict):
        problems.append("체크포인트에 지문(fingerprint)이 없습니다")
    elif stored_fp != fingerprint:
        diff_keys = sorted(
            {*stored_fp.keys(), *fingerprint.keys()}
            - {k for k in stored_fp.keys() & fingerprint.keys()
               if stored_fp[k] == fingerprint[k]})
        problems.append(
            "생성 조건이 바뀌었습니다 (지문 불일치: "
            + ", ".join(diff_keys) + ")")
    completed = checkpoint.get("completed_batches")
    if (not isinstance(completed, int)) or isinstance(completed, bool) \
            or completed < 0:
        problems.append(
            f"완료 배치 수가 올바르지 않습니다: {completed!r}")
        return problems
    total = fingerprint.get("total_batches", 0)
    if completed > total:
        problems.append(
            f"완료 배치 수({completed})가 전체 배치 수({total})를 넘습니다")
    rows = checkpoint.get("rows")
    if not isinstance(rows, list) \
            or any(not isinstance(r, dict) for r in rows):
        problems.append("체크포인트 행(rows)이 dict 목록이 아닙니다")
    else:
        expected_rows = min(completed * fingerprint.get("batch_size", 0),
                            fingerprint.get("n", 0))
        if len(rows) != expected_rows:
            problems.append(
                f"체크포인트 행 수({len(rows)})가 완료 배치 수에 맞지 "
                f"않습니다 (예상 {expected_rows}건)")
    bb = checkpoint.get("branch_blanked")
    if bb is not None and not isinstance(bb, dict):
        problems.append("분기 공란 기록(branch_blanked)이 dict가 아닙니다")
    return problems


def checkpoint_matches(checkpoint, fingerprint: dict) -> bool:
    """체크포인트로 이어서 진행할 수 있는지 (app.py 표시용)."""
    try:
        return not _checkpoint_problems(checkpoint, fingerprint)
    except Exception:
        return False


def format_checkpoint_status(checkpoint) -> str:
    """앱 표시용 재개 상태 문구 — "N배치 완료, 이어서 진행" 형태."""
    try:
        completed = int(checkpoint.get("completed_batches", 0))
        total = int(checkpoint.get("total_batches", 0))
        n_rows = len(checkpoint.get("rows") or [])
        return (f"{completed}배치 완료, 이어서 진행 "
                f"(전체 {total}배치 중 {completed}배치 · {n_rows}건 저장됨)")
    except Exception:
        return "이어서 진행 가능한 중단 기록이 있습니다."


def _make_checkpoint(fingerprint: dict, completed_batches: int,
                     rows: list[dict], branch_blanked: dict,
                     cache_misses: int) -> dict:
    """배치 완료 시점 스냅샷 (행은 복사해 이후 append에 오염되지 않게 한다)."""
    return {
        "version": _CHECKPOINT_VERSION,
        "fingerprint": dict(fingerprint),
        "completed_batches": int(completed_batches),
        "total_batches": int(fingerprint["total_batches"]),
        "rows": [dict(r) for r in rows],
        "branch_blanked": dict(branch_blanked),
        "cache_misses": int(cache_misses),
        "status": "partial",
    }


# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 추정 확률 계측 헬퍼 (2차 캘리브레이션 C0+C1, 워커 C)
#
# 워커 B가 GenerationContext에 추가하는 필드(batch_estimates·
# batch_unmatched·last_call_attempts)를 방어적으로 읽는다 — 베이스
# strategies.py에는 이 필드가 없어 getattr 기본값으로 동작한다.
# 필드 형태 가정 (NOTES.md 참조): batch_estimates는 응답자별
# {qid: {보기: 확률}} dict 목록, batch_unmatched는 {raw키: 질량} dict,
# last_call_attempts는 이번 호출의 시도 횟수(1 = 재시도 없음).
# ----------------------------------------------------------------------------
def _ctx_estimates_snapshot(ctx) -> tuple[int, dict | int]:
    """배치 전 ctx 추정 계측 스냅샷 (델타 계산용)."""
    est = getattr(ctx, "batch_estimates", None)
    n_est = len(est) if isinstance(est, list) else 0
    unm = getattr(ctx, "batch_unmatched", None)
    if isinstance(unm, dict):
        unm_snap: dict | int = {k: float(v) for k, v in unm.items()}
    elif isinstance(unm, list):
        unm_snap = len(unm)
    else:
        unm_snap = {}
    return n_est, unm_snap


def _ctx_estimates_delta(ctx, snapshot) -> tuple[list[dict], dict[str, float]]:
    """스냅샷 이후 ctx에 새로 쌓인 추정·미매칭 델타를 반환한다."""
    n_before, unm_before = snapshot
    est = getattr(ctx, "batch_estimates", None)
    new_est: list[dict] = []
    if isinstance(est, list):
        new_est = [e for e in est[n_before:] if isinstance(e, dict)]
    elif isinstance(est, dict):
        # 배치 단위 dict 1건으로 쌓는 형태면 통째로 1표본으로 본다.
        new_est = [est]
    unm = getattr(ctx, "batch_unmatched", None)
    delta: dict[str, float] = {}
    if isinstance(unm, dict) and isinstance(unm_before, dict):
        for k, v in unm.items():
            d = float(v) - float(unm_before.get(k, 0.0))
            if d:
                delta[str(k)] = d
    elif isinstance(unm, list) and isinstance(unm_before, int):
        for e in unm[unm_before:]:
            if isinstance(e, dict):
                for k, v in e.items():
                    # 워커 B 형태: {qid: {raw키: 질량}} (중첩). 평탄 {raw키: 질량}도 허용.
                    if isinstance(v, dict):
                        for rk, rv in v.items():
                            delta[str(rk)] = delta.get(str(rk), 0.0) + float(rv)
                    else:
                        delta[str(k)] = delta.get(str(k), 0.0) + float(v)
    return new_est, delta


def _merge_estimates(sums: dict[str, dict[str, float]],
                     counts: dict[str, int],
                     estimates: list[dict]) -> None:
    """응답자별 추정 dict 목록을 문항별 합/건수에 누적한다."""
    for row in estimates:
        for qid, probs in row.items():
            if not isinstance(probs, dict) or not probs:
                continue
            counts[str(qid)] = counts.get(str(qid), 0) + 1
            acc = sums.setdefault(str(qid), {})
            for opt, p in probs.items():
                try:
                    acc[str(opt)] = acc.get(str(opt), 0.0) + float(p)
                except (TypeError, ValueError):
                    continue


def _read_execution_info() -> dict:
    """providers.get_last_call_info() 방어적 읽기 (워커 D).

    함수가 아직 없으면(ImportError/AttributeError/예외) 기본값
    {"fallback_used": False, "key_start_index": None}을 반환한다 —
    여기서 False는 '폴백 미사용 확인'이 아니라 '보고 없음'과 같은
    뜻이다 (NOTES.md 참조).
    """
    try:
        from . import providers as _providers
        info = _providers.get_last_call_info()
    except Exception:
        return {"fallback_used": False, "key_start_index": None}
    if not isinstance(info, dict):
        return {"fallback_used": False, "key_start_index": None}
    try:
        ksi = info.get("start_key_index")
        ksi = int(ksi) if ksi is not None else None
    except (TypeError, ValueError):
        ksi = None
    return {"fallback_used": bool(info.get("fallback_used", False)),
            "key_start_index": ksi}


# ----------------------------------------------------------------------------
# 가상 응답 생성 (전략 오케스트레이션)
# ----------------------------------------------------------------------------
class GenerationService:
    def __init__(self, strategy_order: list[str] | None = None) -> None:
        self.strategy_order = strategy_order or STRATEGY_ORDER
        # 마지막 생성의 페르소나 메타 (출처·쿼터 달성률·완화 기록).
        # generate_responses()는 DataFrame만 반환하고, 메타는 여기에 둔다.
        self.last_gen_meta: dict | None = None
        # Nemotron 모드의 원본 샘플링 메타 (쿼터 달성률 상세·완화 목록 포함)
        self.last_nemotron_meta: dict | None = None

        # 분기(branching) 적용 메타 (규칙·공란 처리 건수·문항별 유효 n).
        # generate_responses()가 채운다.
        self.last_branch_meta: dict | None = None

        # 배치 체크포인트 (워커 D): 마지막으로 완료된 배치의 스냅샷.
        # generate_responses()가 배치마다 갱신한다. 중간에 예외로 끊겨도
        # 마지막 완료 배치까지는 여기에 남아 app.py가 세션 상태에 옮길 수 있다.
        self.last_checkpoint: dict | None = None
        # 마지막 생성의 재개 정보: {"resumed_batches", "cache_misses",
        # "cache_covered", "status", "status_text"}.
        self.last_resume_info: dict | None = None
        # 추정 확률 계측 (2차 캘리브레이션 C0+C1, 워커 C):
        # generate_responses()가 채운다. {"mean_estimated_probs",
        # "unmatched": {"keys", "total_mass"}, "retry_counts"}.
        # 반환값이 DataFrame이라 dict로 바꿀 수 없어 (app.py 2634행·
        # _save_current_project·step5가 DataFrame을 기대함) 이 속성과
        # last_gen_meta 요약으로 노출한다.
        self.last_estimation: dict | None = None
        # 실행 로그 (워커 D의 providers.get_last_call_info() 방어적 읽기):
        # {"fallback_used": bool, "key_start_index": int | None}.
        self.last_execution: dict | None = None


    # Nemotron 모드의 프로필 컬럼 순서 (직접 모드의 persona_dims 대응)
    NEMOTRON_PROFILE_COLS = (
        "프로필_성별", "프로필_연령대", "프로필_지역", "프로필_세부지역",
        "프로필_직업", "프로필_학력", "프로필_결혼상태",
    )

    def _select_strategy(self, q: dict, ctx: GenerationContext):
        return select_strategy(q, ctx, self.strategy_order)

    def generate_responses(
        self,
        questions: list[dict],
        persona_dims: dict[str, dict[str, float]],
        n: int,
        *,
        provider_id: str,
        model: str,
        learned: dict | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress_cb=None,
        cache: dict | None = None,
        seed: int | None = None,
        transcripts: list[dict] | None = None,
        use_precision: bool = False,
        generate_fn=None,
        persona_source: str = "manual",
        nemotron_quotas: dict | None = None,
        nemotron_count_quotas: dict | None = None,
        nemotron_region_level: str = "region",
        nemotron_facets: tuple[str, ...] | list[str] | None = None,
        checkpoint: dict | None = None,
        checkpoint_cb=None,
    ) -> pd.DataFrame:
        """가상 응답 n건을 생성해 DataFrame으로 반환.

        문항마다 레지스트리 우선순위대로 전략을 매칭한다:
        - 학습된 분포가 있는 선택형 → 분포학습 (API 호출 없음)
        - 고정밀 컨텍스트의 선택형 → 고정밀 (인터뷰 기반)
        - 나머지 선택형 → 분포추출 (Prolific Gordon et al. 2026 방식)
        - 주관식 → 주관식 생성 (LLM 배치)

        persona_source="nemotron"이면 Nemotron 발췌본에서 쿼터대로
        페르소나를 추출한다 (응답 생성 로직은 동일). 메타(출처·쿼터
        달성률·완화 기록)는 self.last_gen_meta에 남긴다.

        branching 필드가 있는 문항은 생성 직후 분기 조건을 적용한다:
        원문항에 답하지 않았거나(공란) 트리거와 맞지 않으면 해당 문항은
        공란("")으로 둔다. 적용 내역은 self.last_branch_meta와
        self.last_gen_meta["분기 스킵"]에 남긴다.

        배치 중간 저장 / 재개 (checkpoint, checkpoint_cb):
        배치 하나가 끝날 때마다 체크포인트 스냅샷을 만들고
        self.last_checkpoint에 남긴 뒤 checkpoint_cb(스냅샷)을 부른다.
        app.py는 이 콜백으로 스냅샷을 세션 상태에 저장한다. 중간에
        예외로 끊기면 같은 seed·같은 조건으로 checkpoint를 넘겨 다시
        부르면 완료 배치는 재생성 스킵(행은 체크포인트 그대로, 난수
        순서만 재현)하고 마지막 완료 배치부터 이어간다. 재개 정보는
        self.last_resume_info에, 상태 문구는
        format_checkpoint_status(checkpoint)로 확인한다.
        seed가 None이면 재개가 불가해 ValueError를 낸다. 지문이 맞지
        않는 체크포인트도 ValueError를 낸다 (조용히 처음부터 다시
        돌리지 않는다 — 행이 섞이는 것을 막기 위함).

        """

        rng = random.Random(seed)
        learned = learned or {}
        transcripts = transcripts or []
        # --- 배치 체크포인트: 지문 계산·검증 (워커 D) ---
        total_batches = (n + batch_size - 1) // batch_size if n > 0 else 0
        fingerprint = build_checkpoint_fingerprint(
            questions, n, batch_size=batch_size, seed=seed,
            provider_id=provider_id, model=model,
            persona_source=persona_source, persona_dims=persona_dims,
            learned=learned,
            transcripts=transcripts, use_precision=use_precision,
            nemotron_quotas=nemotron_quotas,
            nemotron_count_quotas=nemotron_count_quotas,
            nemotron_region_level=nemotron_region_level,
            nemotron_facets=nemotron_facets)
        resume_from = 0
        checkpoint_rows: list[dict] = []
        branch_blanked: dict[str, int] = {}
        cache_misses = 0
        if checkpoint is not None:
            if seed is None:
                raise ValueError(
                    "체크포인트 이어서 진행에는 seed가 필요합니다 "
                    "(seed가 없으면 중단 전과 같은 페르소나·난수 순서를 "
                    "재현할 수 없습니다).")
            problems = _checkpoint_problems(checkpoint, fingerprint)
            if problems:
                raise ValueError(
                    "체크포인트가 현재 생성 조건과 맞지 않아 이어서 "
                    "진행할 수 없습니다: " + "; ".join(problems))
            resume_from = int(checkpoint["completed_batches"])
            checkpoint_rows = [dict(r) for r in checkpoint["rows"]]
            branch_blanked = dict(checkpoint.get("branch_blanked") or {})
        ctx = GenerationContext(
            provider_id=provider_id,
            model=model,
            learned=learned,
            transcripts=transcripts,
            use_precision=bool(use_precision and transcripts),
            batch_size=batch_size,
            cache=cache if cache is not None else {},
            generate_fn=generate_fn,
            persona_source=persona_source,
            nemotron_facets=(tuple(nemotron_facets)
                             if nemotron_facets is not None else None),
        )

        # Nemotron 모드: 전체 패널을 한 번에 쿼터대로 추출한다
        # (배치별 추출보다 쿼터 달성률이 정확하다).
        nemotron_panel: list[dict] | None = None
        nemotron_meta: dict | None = None
        if persona_source == "nemotron":
            records = load_nemotron_extract()
            nemotron_panel, nemotron_meta = sample_nemotron_personas(
                records, nemotron_quotas, n, rng,
                count_quotas=nemotron_count_quotas,
                region_level=nemotron_region_level)
            self.last_nemotron_meta = nemotron_meta
        else:
            self.last_nemotron_meta = None

        # 분기 규칙 컴파일 (문항 목록에서 한 번만 파싱 — 배치마다 반복하지 않음)
        questions_by_id = {q.get("id"): q for q in questions if q.get("id")}
        question_order = [q.get("id") for q in questions if q.get("id")]
        branch = _compile_branch_rules(questions)


        # 문항 → 전략 매칭 (전략별 그룹)
        groups: dict[str, list[dict]] = {}
        for q in questions:
            strategy = self._select_strategy(q, ctx)
            groups.setdefault(strategy.name, []).append(q)

        # 추정 확률 계측 누적기 (워커 C): fresh 배치의 델타만 합산한다.
        # 재개로 다시 실행되는 완료 배치는 행을 버리므로 추정치도
        # 합산하지 않는다 (이중 집계 방지). 재시도 횟수는 실행된
        # 모든 배치(재개 재생성 포함)에 대해 배치당 1개씩 기록한다.
        est_sums: dict[str, dict[str, float]] = {}
        est_counts: dict[str, int] = {}
        unmatched_keys: dict[str, float] = {}
        retry_counts: list[int] = []

        # rows는 체크포인트 행에서 이어받는다 (재개 시) — 새 배치를 뒤에 붙인다.
        rows: list[dict] = list(checkpoint_rows)
        if resume_from >= total_batches and total_batches > 0:
            # 이미 전부 완료된 체크포인트 — LLM 호출 없이 저장된 행 그대로 반환.
            if progress_cb:
                progress_cb(1.0)
            self.last_checkpoint = dict(checkpoint)
            resume_status = "already_complete"
        else:
            resume_status = "resumed" if resume_from else "fresh"
            for b in range(total_batches):
                bsize = min(batch_size, n - b * batch_size)
                if persona_source == "nemotron":
                    personas = nemotron_panel[b * batch_size:(b + 1) * batch_size]
                    ctx.nemotron_records = personas
                else:
                    personas = [sample_persona(persona_dims, rng)
                                for _ in range(bsize)]

                if b < resume_from:
                    # --- 이어서 진행: 완료 배치는 재생성 스킵 ---
                    # 같은 코드 경로로 다시 실행해 페르소나·전략의 난수 소비만
                    # 원래 순서대로 재현하고, 재생성된 행은 버린다.
                    # 캐시에 프롬프트가 있으면 LLM 호출이 없고, 없으면 재호출
                    # 건수를 cache_misses에 기록한다. 분기 적용은 난수를
                    # 쓰지 않으므로 생략하고 체크포인트의 공란 건수를 쓴다.
                    _keys_before = set(ctx.cache)
                    _est_snap = _ctx_estimates_snapshot(ctx)
                    _max_attempts = 0
                    for sname, qs in groups.items():
                        strategy = get_strategy(sname)
                        try:
                            ctx.last_call_attempts = None
                        except Exception:
                            pass
                        strategy.generate(ctx, qs, personas, rng)
                        _att = getattr(ctx, "last_call_attempts", None)
                        if (isinstance(_att, int)
                                and not isinstance(_att, bool)
                                and _att > _max_attempts):
                            _max_attempts = _att
                    # 재생성 배치는 행·추정치를 버리지만, 이번 실행에서
                    # 실제로 돌린 호출의 재시도 횟수는 기록한다.
                    retry_counts.append(max(0, _max_attempts - 1))
                    cache_misses += len(set(ctx.cache) - _keys_before)
                    if progress_cb:
                        progress_cb((b + 1) / total_batches)
                    continue

                sampled: list[dict] = [{} for _ in range(bsize)]
                _est_snap = _ctx_estimates_snapshot(ctx)
                _max_attempts = 0
                for sname, qs in groups.items():
                    strategy = get_strategy(sname)
                    try:
                        ctx.last_call_attempts = None
                    except Exception:
                        pass
                    results = strategy.generate(ctx, qs, personas, rng)
                    _att = getattr(ctx, "last_call_attempts", None)
                    if (isinstance(_att, int)
                            and not isinstance(_att, bool)
                            and _att > _max_attempts):
                        _max_attempts = _att
                    for i, ans in enumerate(results):
                        sampled[i].update(ans)
                # fresh 배치 델타만 계측에 합산한다.
                _new_est, _unm_delta = _ctx_estimates_delta(ctx, _est_snap)
                _merge_estimates(est_sums, est_counts, _new_est)
                for _k, _v in _unm_delta.items():
                    unmatched_keys[_k] = unmatched_keys.get(_k, 0.0) + _v
                retry_counts.append(max(0, _max_attempts - 1))


                # 분기 적용: 해당 없음 문항을 공란으로 둔다 (전략 무관 단일 지점).
                # 모든 전략 결과가 sampled에 합쳐진 뒤이므로 전략 종류와 무관하다.
                if branch["requires"] or branch["skip_to"]:

                    for i in range(bsize):

                        for qid, rule in branch["requires"].items():

                            if not _branch_rule_applies(rule, sampled[i]):

                                if not _is_blank_answer(sampled[i].get(qid, "")):

                                    sampled[i][qid] = ""
                                    branch_blanked[qid] = branch_blanked.get(qid, 0) + 1

                        for src_qid, rule in branch["skip_to"].items():

                            if _skip_trigger_matches(rule, sampled[i]):

                                for mid_qid in _questions_between(
                                        question_order, src_qid, rule["target"]):

                                    if not _is_blank_answer(sampled[i].get(mid_qid, "")):

                                        sampled[i][mid_qid] = ""
                                        branch_blanked[mid_qid] = branch_blanked.get(mid_qid, 0) + 1


                for i, persona in enumerate(personas):

                    if persona_source == "nemotron":
                        row = {
                            "프로필_성별": persona.get("sex", ""),
                            "프로필_연령대": persona.get("age_band", ""),
                            "프로필_지역": persona.get("region", ""),
                            "프로필_세부지역": persona.get("district", ""),
                            "프로필_직업": persona.get("occupation", ""),
                            "프로필_학력": persona.get("education_level", ""),
                            "프로필_결혼상태": persona.get("marital_status", ""),
                        }
                    else:
                        row = {f"프로필_{dim}": val
                               for dim, val in persona.items()}
                    for q in questions:
                        row[q["id"]] = sampled[i].get(q["id"], "")
                    rows.append(row)

                # --- 배치 완료 즉시 체크포인트 저장 ---
                # 예외로 끊겨도 마지막 완료 배치까지는 남는다. app.py는
                # checkpoint_cb로 이 스냅샷을 세션 상태에 저장한다.
                # progress_cb보다 먼저 저장한다 — 진행률 표시에서 예외가
                # 나도 막 완료된 배치는 잃지 않기 위함.
                snap = _make_checkpoint(
                    fingerprint, b + 1, rows, branch_blanked, cache_misses)
                self.last_checkpoint = snap
                if checkpoint_cb:
                    checkpoint_cb(snap)

                if progress_cb:
                    progress_cb((b + 1) / total_batches)

        self.last_resume_info = {
            "resumed_batches": resume_from,
            "cache_misses": cache_misses,
            "cache_covered": cache_misses == 0,
            "status": resume_status,
            "status_text": (format_checkpoint_status(self.last_checkpoint)
                            if self.last_checkpoint else ""),
        }

        if persona_source == "nemotron":
            profile_cols = list(self.NEMOTRON_PROFILE_COLS)
        else:
            profile_cols = [f"프로필_{dim}" for dim in persona_dims]
        col_order = profile_cols + [q["id"] for q in questions]
        df = pd.DataFrame(rows)
        df = df[[c for c in col_order if c in df.columns]]

        # 추정 확률 계측 확정 (워커 C) — fresh 배치 델타 합산 결과.
        mean_estimated_probs = {
            qid: {opt: total / est_counts[qid]
                  for opt, total in opts.items()}
            for qid, opts in est_sums.items()
            if est_counts.get(qid, 0) > 0
        }
        unmatched_total = float(sum(unmatched_keys.values()))
        self.last_estimation = {
            "mean_estimated_probs": mean_estimated_probs,
            "unmatched": {"keys": dict(unmatched_keys),
                          "total_mass": unmatched_total},
            "retry_counts": list(retry_counts),
        }
        self.last_execution = _read_execution_info()

        self.last_gen_meta = self._build_gen_meta(
            persona_source, nemotron_meta)
        # 추정 계측 요약 (화면·엑셀 메타에 그대로 뜬다 — app.py 수정 없이
        # gen_meta → 프로젝트 저장 → 엑셀 메타 경로를 탄다).
        if (mean_estimated_probs or unmatched_total > 0
                or sum(retry_counts) > 0):
            _top_unm = sorted(unmatched_keys.items(),
                              key=lambda kv: -kv[1])[:3]
            _unm_txt = "; ".join(f"'{k}' {v:.3f}" for k, v in _top_unm)
            self.last_gen_meta["추정 계측"] = (
                f"평균 추정확률 문항 {len(mean_estimated_probs)}개; "
                f"미매칭 질량 총 {unmatched_total:.3f}"
                + (f" ({_unm_txt})" if _unm_txt else "")
                + f"; 재시도 총 {sum(retry_counts)}회 / {len(retry_counts)}배치"
            )
        if (self.last_execution or {}).get("fallback_used"):
            self.last_gen_meta["실행"] = (
                "폴백 키 사용 "
                f"(시작 키 인덱스 {self.last_execution.get('key_start_index')})"
            )
        if resume_from:
            # 이어서 진행 표시를 메타에도 남긴다 (화면·엑셀 메타에 그대로 뜬다).
            self.last_gen_meta = dict(self.last_gen_meta or {})
            self.last_gen_meta["이어서 진행"] = (
                f"{resume_from}배치 완료 상태에서 이어서 진행"
                + ("" if cache_misses == 0
                   else f" (캐시 미커버 {cache_misses}건 LLM 재호출)"))

        # 분기 적용 메타: 적용된 규칙·공란 처리 건수·분기 문항별 유효 n.
        # "분기 스킵"은 app.py가 last_gen_meta를 그대로 화면·엑셀 메타에 합치므로
        # 별도 UI 수정 없이 표시된다.
        branched_qids = sorted(
            set(branch["requires"]) |
            {mid for rule in branch["skip_to"].values()
             for mid in _questions_between(question_order, rule["source"],
                                           rule["target"])})
        effective_n = {}
        for qid in branched_qids:
            if qid in df.columns:
                effective_n[qid] = int(
                    df[qid].map(lambda v: not _is_blank_answer(v)).sum())
        self.last_branch_meta = {
            "규칙": dict(branch["summaries"]),
            "공란 처리": dict(branch_blanked),
            "문항별 유효 n": effective_n,
            # 파싱에 실패한 branching 원문 — 적용 안 된 분기가 있으면
            # 여기서 확인한다 (무시이지 오류가 아니므로 경고로만 남긴다).
            "파싱 실패": dict(branch["unparsed"]),
        }
        if branch_blanked:
            self.last_gen_meta = dict(self.last_gen_meta or {})
            self.last_gen_meta["분기 스킵"] = "; ".join(
                f"{qid} {c}건 공란" for qid, c in sorted(branch_blanked.items()))

        return df


    @staticmethod
    def _build_gen_meta(persona_source: str,
                        nemotron_meta: dict | None) -> dict:
        """결과 화면·엑셀에 표시할 페르소나 출처 메타 (표시용 문자열 위주)."""
        if persona_source != "nemotron" or not nemotron_meta:
            return {
                "페르소나 출처": PERSONA_SOURCES["manual"],
            }
        relaxations = nemotron_meta.get("relaxations") or []
        relaxed = nemotron_meta.get("relaxed_count", 0)
        if relaxed == 0:
            quota_line = "100% (완화 없음)"
            relax_line = "없음"
        else:
            by_level: dict[str, int] = {}
            for r in relaxations:
                by_level[r["level"]] = by_level.get(r["level"], 0) + r["shortfall"]
            quota_line = (
                f"완화 {relaxed}건 "
                f"({', '.join(f'{lv} {c}건' for lv, c in sorted(by_level.items()))})"
            )
            relax_line = "; ".join(
                f"{r['cell'][0]} {r['cell'][1]} {r['cell'][2]} "
                f"{r['shortfall']}건 ({r['level']})"
                for r in relaxations[:5]
            )
            if len(relaxations) > 5:
                relax_line += f" 외 {len(relaxations) - 5}건"
        # 차원별 최대 편차 (목표 대비 실제)
        max_devs = []
        for dim, vals in (nemotron_meta.get("quota_achievement") or {}).items():
            dev = max((abs(v["actual_pct"] - v["target_pct"])
                       for v in vals.values()), default=0.0)
            max_devs.append(f"{dim} 최대 {dev:.1f}%p")
        meta = {
            "페르소나 출처": PERSONA_SOURCES["nemotron"],
            "Nemotron 발췌본": (
                f"{NEMOTRON_EXTRACT_VERSION} · "
                f"{NEMOTRON_EXTRACT_SIZE:,}건"
            ),
            "쿼터 달성": quota_line,
            "쿼터 편차": " · ".join(max_devs) if max_devs else "-",
            "완화 기록": relax_line,
            "쿼터 정규화": "예" if nemotron_meta.get("normalized") else "아니오",
            "쿼터 방식": ("건수 쿼터" if nemotron_meta.get("quota_mode") == "count"
                         else "비율 쿼터"),
            "지역 단위": ("시도 (17개)" if nemotron_meta.get("region_level") == "province"
                         else "권역 (5개)"),
        }
        warning = nemotron_meta.get("count_quota_warning")
        if warning:
            meta["건수 쿼터 경고"] = warning
            if nemotron_meta.get("count_quota_fill"):
                meta["건수 쿼터 채움"] = (
                    f"부족분 {nemotron_meta['count_quota_fill']}건을 "
                    "발췌본 분포대로 채움")
            truncated = nemotron_meta.get("count_quota_truncated") or []
            if truncated:
                meta["건수 쿼터 절단"] = "; ".join(
                    f"{'|'.join(map(str, t['cell']))} "
                    f"{t['from']}건→{t['to']}건" for t in truncated[:5])
        return meta

# ----------------------------------------------------------------------------
# 합성 표기 표준화 (Phase 2)
# ----------------------------------------------------------------------------
# 다운로드 산출물(xlsx/docx)과 수정 설문지 초안은 합성 응답 기반 사전 테스트
# 결과임을 표지에 명시한다. 표기는 추가만 하고 기존 시트·컬럼·표 구조는
# 그대로 둔다.
_SYNTHETIC_LABEL = "[합성 응답 · Pre-test]"

_SYNTHETIC_NOTE = (
    "본 산출물은 합성 응답(가상 응답) 기반 사전 테스트(Pre-test) 결과입니다. "
    "실제 조사 전 실측으로 확정해야 합니다."
)


def _normalize_finding_text(value) -> str:
    """finding/문항 텍스트 비교용 정규화: NFKC + 공백 정리."""
    return re.sub(r"\s+", " ",
                  unicodedata.normalize("NFKC", str(value or ""))).strip()


def _finding_matches_question(finding: dict, question: dict) -> bool:
    """진단 finding이 해당 문항에 대한 것인지 느슨하게 매칭한다.

    diagnose()는 '문항' 값으로 문항 텍스트(qtext)를 넣지만, 다른 호출부는
    문항 ID(Q12 등)를 넣을 수 있다. 그래서 ID가 일치하면 ID로, 아니면
    텍스트 포함 여부로 매칭한다. "(전체)"·"(패널 구성)" 같은 문항 무관
    finding은 어떤 문항과도 매칭하지 않는다 (요약에만 들어간다).
    """
    if not isinstance(finding, dict) or not isinstance(question, dict):
        return False
    f_q = _normalize_finding_text(finding.get("문항", ""))
    if not f_q:
        return False
    qid = _normalize_finding_text(question.get("id", ""))
    qtext = _normalize_finding_text(question.get("text", ""))
    if qid:
        # ID 정확 일치 또는 "Q12. ..."처럼 ID로 시작하는 표기
        if f_q == qid:
            return True
        if re.match(rf"{re.escape(qid)}(?![0-9])", f_q):
            return True
    if qtext and f_q:
        # 텍스트 정확 일치 또는 한쪽이 다른 쪽을 포함 (짧은 문자열의
        # 우연한 부분일치는 제외 — 8자 이상일 때만 포함 매칭).
        if f_q == qtext:
            return True
        if len(f_q) >= 8 and (f_q in qtext or qtext in f_q):
            return True
    return False


def _findings_for_question(question: dict,
                           findings: list[dict]) -> list[dict]:
    """한 문항에 매칭되는 findings만 골라낸다 (원본 순서 유지)."""
    return [f for f in (findings or [])
            if _finding_matches_question(f, question)]


def _improvement_rows(findings: list[dict]) -> list[dict]:
    """'개선 제안' 섹션용으로 findings를 정규화한다 (.get() 안전 접근)."""
    rows = []
    for f in (findings or []):
        if not isinstance(f, dict):
            continue
        rows.append({
            "문항": str(f.get("문항", "") or ""),
            "문제": str(f.get("문제", "") or ""),
            "근거": str(f.get("근거", "") or ""),
            "해결책": str(f.get("해결책", "") or ""),
            "심각도": str(f.get("심각도", "") or ""),
        })
    return rows


def _question_type_label(qtype: str) -> str:
    """문항 유형 코드를 표시용 라벨로 바꾼다 (미등록 유형은 코드 그대로)."""
    try:
        return get_question_type(qtype).label
    except Exception:
        return str(qtype or "")


def _setup_docx_document():
    """보고서용 DOCX 문서 뼈대 (맑은 고딕 기본 폰트 — to_word_bytes와 동일)."""
    from docx import Document
    from docx.shared import Pt
    from docx.oxml.ns import qn
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "맑은 고딕"
    style.font.size = Pt(10)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")
    return doc


# ----------------------------------------------------------------------------
# 엑셀 출력
# ----------------------------------------------------------------------------
class ExportService:
    @staticmethod
    def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "가상응답",
                       meta: dict | None = None,
                       diagnostics: list[dict] | None = None) -> bytes:
        from core.diagnostics import findings_to_dataframe
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            # 표지 (첫 시트) — 합성 표기 + 근거(문항/응답 ID 연결).
            # 기존 데이터 시트·컬럼은 건드리지 않고 표지 시트만 앞에 추가한다.
            qids = [str(c) for c in df.columns
                    if not str(c).startswith("프로필_")]
            cover_rows = [
                [_SYNTHETIC_LABEL],
                [_SYNTHETIC_NOTE],
                [""],
                ["근거 — 문항 ID", ", ".join(qids) if qids else "-"],
                ["근거 — 응답 건수", str(len(df))],
            ]
            pd.DataFrame(cover_rows).to_excel(
                writer, index=False, header=False, sheet_name="표지"
            )
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            info_rows = [["항목", "내용"]]
            for line in METHODOLOGY_NOTE.strip().split("\n"):
                info_rows.append([line])
            if meta:
                info_rows.append([""])
                for k, v in meta.items():
                    info_rows.append([k, str(v)])
            pd.DataFrame(info_rows).to_excel(
                writer, index=False, header=False, sheet_name="방법론_안내"
            )
            # 파일럿 진단 리포트 시트 (규칙 기반, 환각 없음)
            findings_to_dataframe(diagnostics or []).to_excel(
                writer, index=False, sheet_name="진단_리포트"
            )
            # 개선 제안 섹션 — findings가 있을 때만 별도 시트로 추가한다.
            if diagnostics:
                findings_to_dataframe(diagnostics).to_excel(
                    writer, index=False, sheet_name="개선_제안"
                )
        return buf.getvalue()

    @staticmethod
    def to_word_bytes(df: pd.DataFrame,
                      meta: dict | None = None,
                      diagnostics: list[dict] | None = None,
                      questions: list[dict] | None = None) -> bytes:
        """파일럿 진단 리포트를 Word(.docx)로 생성.

        깔끔한 디자인: 제목, 메타 정보, 진단 결과 표, 방법론 안내 섹션.

        """
        from docx import Document
        from docx.shared import Pt, Cm, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.oxml.ns import qn
        from datetime import datetime
        doc = Document()

        # 기본 폰트 설정
        style = doc.styles["Normal"]
        style.font.name = "맑은 고딕"
        style.font.size = Pt(10)
        style.element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")

        # 제목
        title = doc.add_heading("파일럿 테스트 진단 리포트", level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # 생성 일시
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        # 합성 표기 (Phase 2) — 표지 상단. 근거(문항/응답 ID)도 함께 표시한다.
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(_SYNTHETIC_LABEL)
        run.bold = True
        run.font.size = Pt(11)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(_SYNTHETIC_NOTE)
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        if df is not None and not df.empty:
            qids = [str(c) for c in df.columns
                    if not str(c).startswith("프로필_")]
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(
                f"근거 — 문항 {len(qids)}개"
                + (f" ({', '.join(qids)})" if qids else "")
                + f", 응답 {len(df)}건")
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        doc.add_paragraph()  # 여백

        # 메타 정보
        if meta:
            doc.add_heading("기본 정보", level=1)
            table = doc.add_table(rows=1, cols=2)
            table.style = "Light Grid Accent 1"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            hdr = table.rows[0].cells
            hdr[0].text = "항목"
            hdr[1].text = "내용"
            for k, v in meta.items():
                row = table.add_row().cells
                row[0].text = str(k)
                row[1].text = str(v)
            doc.add_paragraph()

        # 진단 결과
        doc.add_heading("진단 결과", level=1)
        findings = diagnostics or []
        if not findings:
            p = doc.add_paragraph()
            run = p.add_run("✅ 진단 결과: 문제가 발견되지 않았습니다. 설문지가 정상적으로 작동합니다.")
            run.font.color.rgb = RGBColor(0x1B, 0x7A, 0x34)
        else:
            # 요약
            high = sum(1 for f in findings if f.get("심각도") == "높음")
            mid = len(findings) - high
            p = doc.add_paragraph()
            p.add_run(f"총 {len(findings)}건 발견 (높음 {high}건, 중간 {mid}건)")
            table = doc.add_table(rows=1, cols=5)
            table.style = "Light Grid Accent 1"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            headers = ["문항", "문제", "근거", "해결책", "심각도"]
            for i, h in enumerate(headers):
                cell = table.rows[0].cells[i]
                cell.text = h
                for par in cell.paragraphs:
                    for r in par.runs:
                        r.bold = True
            for f in findings:
                row = table.add_row().cells
                row[0].text = str(f.get("문항", ""))
                row[1].text = str(f.get("문제", ""))
                row[2].text = str(f.get("근거", ""))
                row[3].text = str(f.get("해결책", ""))
                sev = str(f.get("심각도", ""))
                row[4].text = sev
                # 심각도에 따라 색상
                for par in row[4].paragraphs:
                    for r in par.runs:
                        r.bold = True
                        if sev == "높음":
                            r.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
                        else:
                            r.font.color.rgb = RGBColor(0xCC, 0x7A, 0x00)

            # 열 너비 조정
            for row in table.rows:
                row.cells[0].width = Cm(3)
                row.cells[1].width = Cm(4)
                row.cells[2].width = Cm(4)
                row.cells[3].width = Cm(4)
                row.cells[4].width = Cm(2)
        doc.add_paragraph()

        # 개선 제안 (Phase 2) — findings가 있을 때만 별도 섹션으로 붙인다.
        # 각 항목은 문항 ID, 문제, 근거, 해결책을 포함한다.
        if findings:
            doc.add_heading("개선 제안", level=1)
            table = doc.add_table(rows=1, cols=4)
            table.style = "Light Grid Accent 1"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            headers = ["문항", "문제", "근거", "해결책"]
            for i, h in enumerate(headers):
                cell = table.rows[0].cells[i]
                cell.text = h
                for par in cell.paragraphs:
                    for r in par.runs:
                        r.bold = True
            for f in findings:
                row = table.add_row().cells
                row[0].text = str(f.get("문항", ""))
                row[1].text = str(f.get("문제", ""))
                row[2].text = str(f.get("근거", ""))
                row[3].text = str(f.get("해결책", ""))
            for row in table.rows:
                row.cells[0].width = Cm(3)
                row.cells[1].width = Cm(4)
                row.cells[2].width = Cm(5)
                row.cells[3].width = Cm(5)
            doc.add_paragraph()

        # 응답 요약 통계
        if df is not None and not df.empty and questions:
            doc.add_heading("문항별 응답 요약", level=1)
            for q in questions:
                qid = q.get("id")
                if qid not in df.columns:
                    continue
                doc.add_heading(q.get("text", qid), level=2)
                vc = df[qid].astype(str).value_counts()
                # 공란("")·결측 표기("nan")는 하나의 응답이 아니라 분기 스킵
                # 등으로 답하지 않은 것이다 — 빈 라벨 행 대신 "해당 없음"
                # 범주 하나로 합쳐 명시한다 (리뷰 필수 조치 4). 비율의
                # 분모는 전체 응답자 수 그대로 둔다.
                blank_total = int(sum(
                    c for v, c in vc.items() if _is_blank_answer(v)))
                summary_rows = [
                    (str(v), c) for v, c in vc.items()
                    if not _is_blank_answer(v)]
                if blank_total:
                    summary_rows.append(("해당 없음", blank_total))
                table = doc.add_table(rows=1, cols=3)
                table.style = "Light Grid Accent 1"
                hdr = table.rows[0].cells
                hdr[0].text = "응답"
                hdr[1].text = "건수"
                hdr[2].text = "비율"
                for val, cnt in summary_rows:
                    row = table.add_row().cells
                    row[0].text = val
                    row[1].text = str(cnt)
                    row[2].text = f"{cnt/len(df)*100:.1f}%"
                doc.add_paragraph()

        # 방법론 안내
        doc.add_heading("방법론 안내", level=1)
        for line in METHODOLOGY_NOTE.strip().split("\n"):
            if line.strip():
                doc.add_paragraph(line.strip(), style="List Bullet")
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

    def to_improved_questionnaire_docx(self, questions: list[dict],
                                       findings: list[dict]) -> bytes:
        """진단 findings를 반영한 수정 설문지 초안 DOCX를 바이트로 반환.

        questions: 문항 dict 목록 (id, text, type, options, branching).
        findings: diagnose() 결과 dict 목록 (키: 문항, 문제, 근거, 해결책, 심각도).

        구성: ① 제목 + 합성 표기 ② 개선 제안 요약 ③ 수정 설문지 초안
        (원본 문항 그대로 + 해당 finding의 해결책을 '개선 제안:' 주석으로
        부착) ④ '실측으로 확정' 안내. LLM을 호출하지 않고 findings의
        해결책 텍스트를 그대로 활용하는 결정적 생성이다. findings가
        비어 있으면 문항 목록만 담은 DOCX를 반환한다.
        """
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.oxml.ns import qn
        from datetime import datetime

        questions = list(questions or [])
        rows = _improvement_rows(findings)

        doc = _setup_docx_document()

        # ① 제목 + 합성 표기
        title = doc.add_heading("수정 설문지 초안", level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(_SYNTHETIC_LABEL)
        run.bold = True
        run.font.size = Pt(11)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(_SYNTHETIC_NOTE)
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        doc.add_paragraph()  # 여백

        # ② 개선 제안 요약 — findings가 있을 때만.
        if rows:
            doc.add_heading("개선 제안 요약", level=1)
            high = sum(1 for f in rows if f.get("심각도") == "높음")
            p = doc.add_paragraph()
            p.add_run(f"총 {len(rows)}건 (높음 {high}건, "
                      f"기타 {len(rows) - high}건)")
            table = doc.add_table(rows=1, cols=4)
            table.style = "Light Grid Accent 1"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            headers = ["문항", "문제", "해결책", "심각도"]
            for i, h in enumerate(headers):
                cell = table.rows[0].cells[i]
                cell.text = h
                for par in cell.paragraphs:
                    for r in par.runs:
                        r.bold = True
            for f in rows:
                row = table.add_row().cells
                row[0].text = f.get("문항", "")
                row[1].text = f.get("문제", "")
                row[2].text = f.get("해결책", "")
                sev = f.get("심각도", "")
                row[3].text = sev
                for par in row[3].paragraphs:
                    for r in par.runs:
                        r.bold = True
                        if sev == "높음":
                            r.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
                        else:
                            r.font.color.rgb = RGBColor(0xCC, 0x7A, 0x00)
            doc.add_paragraph()

        # ③ 수정 설문지 초안 — 원본 문항을 그대로 싣고, 해당 finding이
        # 있는 문항 아래에 해결책을 '개선 제안:' 주석으로 붙인다.
        # 매칭이 안 된 finding은 요약(②)에만 있고 주석은 생략한다.
        doc.add_heading("수정 설문지 초안", level=1)
        if not questions:
            doc.add_paragraph("문항이 없습니다.")
        for q in questions:
            if not isinstance(q, dict):
                continue
            qid = str(q.get("id", "") or "")
            qtext = str(q.get("text", "") or "")
            heading_text = f"{qid}. {qtext}" if qid else qtext
            doc.add_heading(heading_text or "(문항 없음)", level=2)
            qtype = str(q.get("type", "") or "")
            label = _question_type_label(qtype)
            p = doc.add_paragraph()
            run = p.add_run(f"유형: {label}" if label else f"유형: {qtype}")
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
            options = q.get("options")
            if isinstance(options, list) and options:
                for opt in options:
                    doc.add_paragraph(str(opt), style="List Bullet")
            branching = str(q.get("branching", "") or "").strip()
            if branching:
                p = doc.add_paragraph()
                run = p.add_run(f"분기: {branching}")
                run.font.size = Pt(9)
                run.font.italic = True
            for f in _findings_for_question(q, rows):
                solution = f.get("해결책", "")
                problem = f.get("문제", "")
                note = f"개선 제안: {solution}" if solution else ""
                if problem and solution:
                    note = f"개선 제안 ({problem}): {solution}"
                if note:
                    p = doc.add_paragraph()
                    run = p.add_run(note)
                    run.bold = True
                    run.font.color.rgb = RGBColor(0x1B, 0x7A, 0x34)
            doc.add_paragraph()

        # ④ 실측으로 확정 안내
        doc.add_heading("실측으로 확정", level=1)
        doc.add_paragraph(
            "이 초안은 합성 응답(가상 응답) 기반 사전 테스트(Pre-test) 결과를 "
            "바탕으로 만든 것입니다. 합성 응답은 실제 응답자를 대신할 수 "
            "없으므로, 실제 조사 전에 반드시 실측 파일럿으로 문항을 확정하세요.",
            style="List Bullet")
        doc.add_paragraph(
            "개선 제안은 진단 규칙이 실제 합성 응답 데이터에서 계산한 통계에만 "
            "근거합니다. 제안대로 고친 문항도 실측에서 다시 검증해야 합니다.",
            style="List Bullet")
        doc.add_paragraph(
            "이 문서를 실제 조사용 설문지로 쓰기 전에는 문항 순서·분기 조건·"
            "보기 문구를 실측 응답으로 최종 확인하세요.",
            style="List Bullet")

        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

# ----------------------------------------------------------------------------
# PDF 설문지 자동 인식
# ----------------------------------------------------------------------------
class PdfQuestionnaireService:
    """PDF 설문지를 업로드하면 AI가 문항을 자동 파싱합니다.

    분기문항(스킵 로직)도 인식하여 branching 필드에 저장합니다.

    """

    # 마지막 자동 인식의 상태를 UI가 간단히 안내할 수 있도록 보관한다.
    last_parse_stats: dict = {}

    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        """PDF에서 텍스트를 추출합니다."""
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts)

    @staticmethod
    def extract_hwpx_text(hwpx_bytes: bytes) -> str:
        """HWPX(한글 문서)에서 텍스트를 추출합니다. (표준 라이브러리만 사용)"""
        import zipfile
        import xml.etree.ElementTree as ET
        texts = []
        with zipfile.ZipFile(io.BytesIO(hwpx_bytes)) as z:
            section_files = sorted(
                n for n in z.namelist()
                if n.startswith("Contents/section") and n.endswith(".xml")
            )
            if not section_files:
                raise ValueError("HWPX 섹션 파일을 찾을 수 없습니다.")
            for name in section_files:
                root = ET.fromstring(z.read(name))
                for elem in root.iter():
                    # hp:p = 문단, hp:t = 텍스트 런 (네임스페이스 무관)
                    if elem.tag.endswith("}p"):
                        para = "".join(
                            t.text for t in elem.iter()
                            if t.tag.endswith("}t") and t.text
                        )
                        if para.strip():
                            texts.append(para.strip())
        result = "\n".join(texts)
        if not result.strip():
            raise ValueError("HWPX에서 텍스트를 추출하지 못했습니다.")
        return result

    @staticmethod
    def extract_doc_text(doc_bytes: bytes) -> str:
        """구형 Word(.doc) 바이너리에서 텍스트를 추출합니다."""
        import re
        import struct
        import olefile
        def _scan_text_runs(data: bytes) -> list[str]:
            """UTF-16LE 한글/영문 텍스트 런 스캔 (fallback용)."""
            text = data.decode("utf-16-le", errors="ignore")
            # 원문자(①②③), 한글, 영문, 숫자, 기본 구두점 포함
            pattern = re.compile(
                r"[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9\u2460-\u24ff\u3200-\u32ff"
                r"\s.,!?;:()\[\]\"'·…—–\-]{2,}"
            )
            runs = []
            for m in pattern.finditer(text):
                s = m.group(0).strip()
                if len(s) >= 2 and re.search(r"[가-힣A-Za-z0-9①-⑳]", s):
                    runs.append(s)
            return runs

        def _fc_to_byte_offset(fc_val: int) -> tuple[int, bool]:
            is_unicode = bool(fc_val & 0x40000000)
            fc = fc_val & 0x3FFFFFFF
            return (fc * 2 if is_unicode else fc), is_unicode

        def _parse_structured(data: bytes) -> str:
            """FIB/피스테이블 기반 구조적 파싱. 실패 시 예외."""
            u16 = lambda o: struct.unpack("<H", data[o:o + 2])[0]
            u32 = lambda o: struct.unpack("<I", data[o:o + 4])[0]
            is_complex = bool(u16(0x0A) & 0x0004)
            ccp_text = u32(0x4C)
            if ccp_text <= 0 or ccp_text > 10_000_000:
                raise ValueError("ccpText 비정상")
            out_parts: list[str] = []
            if is_complex:
                # fcClx/lcbClx: FibRgFcLcb97 시작(0x94) + 35번째 쌍
                fc_clx, lcb_clx = u32(0x1AC), u32(0x1B0)
                if lcb_clx <= 0 or fc_clx + lcb_clx > len(data):
                    raise ValueError("CLX 범위 비정상")
                clx = data[fc_clx:fc_clx + lcb_clx]
                # CLX에서 Pcdt(0x02) 찾기
                pos, pcdt = 0, None
                while pos < len(clx):
                    clxt = clx[pos]
                    if clxt == 0x02:
                        lcb = struct.unpack("<I", clx[pos + 1:pos + 5])[0]
                        pcdt = clx[pos + 5:pos + 5 + lcb]
                        break
                    elif clxt == 0x01:
                        lcb = struct.unpack("<I", clx[pos + 1:pos + 5])[0]
                        pos += 5 + lcb
                    else:
                        raise ValueError("CLX 파싱 실패")
                if not pcdt or len(pcdt) < 8:
                    raise ValueError("Pcdt 없음")
                # 각 Pcd(8바이트)의 bytes 2-5 = FcCompressed
                fcs = []
                for i in range(len(pcdt) // 8):
                    fc_val = struct.unpack(
                        "<I", pcdt[i * 8 + 2:i * 8 + 6])[0]
                    fcs.append(_fc_to_byte_offset(fc_val))
                # piece 구간별로 텍스트 추출
                for i, (byte_off, is_unicode) in enumerate(fcs):
                    end_off = (fcs[i + 1][0] if i + 1 < len(fcs)
                               else len(data))
                    if byte_off >= len(data) or end_off <= byte_off:
                        continue
                    raw = data[byte_off:min(end_off, len(data))]
                    if is_unicode:
                        # 2바이트 경계 맞추기
                        if len(raw) % 2:
                            raw = raw[:-1]
                        out_parts.append(
                            raw.decode("utf-16-le", errors="ignore"))
                    else:
                        for enc in ("cp949", "cp1252"):
                            try:
                                out_parts.append(raw.decode(enc))
                                break
                            except Exception:
                                continue
            else:
                # 단순 문서: fallback 스캔에 맡김
                raise ValueError("단순 문서 — 스캔 방식 사용")
            text = "".join(out_parts)
            # 제어문자 정리
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
            if len(text.strip()) < 10:
                raise ValueError("구조적 파싱 결과 부족")
            return text

        ole = olefile.OleFileIO(io.BytesIO(doc_bytes))
        try:
            if not ole.exists("WordDocument"):
                raise ValueError("WordDocument 스트림이 없습니다.")
            data = ole.openstream("WordDocument").read()
        finally:
            ole.close()

        if len(data) < 512 or data[:2] != b"\xEC\xA5":
            raise ValueError("올바른 Word 97-2003 문서(.doc)가 아닙니다.")

        # 1차: 구조적 파싱 시도
        try:
            return _parse_structured(data)
        except Exception:
            pass

        # 2차: 텍스트 런 스캔 fallback
        seen: set[str] = set()
        uniq: list[str] = []
        for r in _scan_text_runs(data):
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        result = "\n".join(uniq).replace("\r\n", "\n").replace("\r", "\n")
        # 의미 없는 짧은 노이즈 줄 제거
        lines = []
        for ln in result.split("\n"):
            s = ln.strip()
            if len(s) < 2:
                continue
            # 한글이 있으면 유지, 없으면 4자 이상만 유지
            if re.search(r"[가-힣]", s) or len(s) >= 4:
                lines.append(ln)
        result = "\n".join(lines)
        if not result.strip():
            raise ValueError("DOC에서 텍스트를 추출하지 못했습니다.")
        return result

    @staticmethod
    def extract_docx_text(docx_bytes: bytes) -> str:
        """Word(.docx)에서 텍스트를 추출합니다."""
        from docx import Document
        doc = Document(io.BytesIO(docx_bytes))
        parts = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)
        # 표 안의 텍스트도 추출
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text = cell.text.strip()
                    if text:
                        parts.append(text)
        result = "\n".join(parts)
        if not result.strip():
            raise ValueError("DOCX에서 텍스트를 추출하지 못했습니다.")
        return result

    @staticmethod
    def extract_text_auto(file_bytes: bytes, filename: str) -> str:
        """파일 확장자에 따라 텍스트를 자동 추출합니다. (pdf/hwpx/doc/docx)"""
        from pathlib import Path
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return PdfQuestionnaireService.extract_text(file_bytes)
        if ext == ".hwpx":
            return PdfQuestionnaireService.extract_hwpx_text(file_bytes)
        if ext == ".doc":
            return PdfQuestionnaireService.extract_doc_text(file_bytes)
        if ext == ".docx":
            return PdfQuestionnaireService.extract_docx_text(file_bytes)
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext} "
                         "(pdf, hwpx, doc, docx만 지원)")

    @staticmethod
    def _split_text(text: str, max_chars: int,
                    overlap_chars: int) -> list[str]:
        """긴 설문 텍스트를 줄 경계를 우선해 겹치는 구간으로 나눈다.

        문항과 보기 목록이 경계에서 분리되는 것을 줄이기 위해 overlap을
        둔다. 입력이 짧으면 원문을 그대로 한 구간으로 반환한다.

        """
        normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\f", "\n").strip()
        if not normalized:
            return []
        if max_chars <= 0:
            raise ValueError("QUESTIONNAIRE_PARSE_CHUNK_CHARS는 1 이상이어야 합니다.")
        if len(normalized) <= max_chars:
            return [normalized]
        overlap = max(0, min(overlap_chars, max_chars - 1))
        chunks: list[str] = []
        start = 0
        text_len = len(normalized)
        while start < text_len:
            target_end = min(start + max_chars, text_len)
            end = target_end
            if target_end < text_len:
                # 절반 이후의 마지막 줄바꿈에서 끊어 문항 중간 절단을 줄인다.
                boundary_start = start + max_chars // 2
                boundary = normalized.rfind("\n", boundary_start, target_end)
                if boundary > start:
                    end = boundary
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_len:
                break
            next_start = max(end - overlap, start + 1)
            if next_start <= start:
                next_start = end
            start = next_start
        return chunks

    @staticmethod
    def _parse_prompt(text: str, chunk_index: int = 1,
                      chunk_total: int = 1) -> str:
        segment_note = ""
        if chunk_total > 1:
            segment_note = f"""

이 텍스트는 전체 설문 중 {chunk_index}/{chunk_total}번째 구간입니다.
구간 경계에서 잘린 문항은 추측해서 만들지 말고 제외하세요. 겹치는 다음 구간에
완전한 문항이 다시 나타날 수 있습니다. 이 구간에서 완전히 확인되는 문항만 반환하세요."""
        return f"""다음은 설문지에서 추출한 텍스트입니다. 설문 문항을 JSON 배열로 파싱하세요.
규칙:
1. 각 문항은 {{"text": "문항 내용", "type": "single|multi|likert|open", "options": ["보기1", "보기2", ...], "branching": "분기 설명 또는 빈 문자열"}} 형태입니다.
2. type 판단 기준:
   - single: 보기 중 하나만 선택 (단일선택)
   - multi: 복수 선택 가능 (다중선택, "모두 선택" 등)
   - likert: 5점 척도 (매우 그렇다~전혀 그렇지 않다 등)
   - open: 주관식 서술형
3. 분기문항(스킵 로직)이 있으면 branching에 설명을 적으세요. 예: "문3에서 ① 응답 시 문5로 이동", "문2의 ② 응답자만 응답"
   분기가 없으면 빈 문자열("")을 넣으세요.
4. 보기 번호(①②③, 1. 2. 3. 등)는 제거하고 보기 내용만 추출하세요.
5. 설문 안내문, 표지, 인구통계 안내 등은 문항에서 제외하세요.
6. JSON 배열만 출력하고 다른 설명은 붙이지 마세요.{segment_note}
--- 설문지 텍스트 ---
{text}
--- 끝 ---"""

    @staticmethod
    def _clean_questions(questions) -> list[dict]:
        """AI 결과를 앱이 사용하는 문항 구조로 정규화한다."""
        if not isinstance(questions, list):
            return []
        valid_types = {"single", "multi", "likert", "open"}
        cleaned = []
        for q in questions:
            if not isinstance(q, dict) or not str(q.get("text", "")).strip():
                continue
            qtype = q.get("type", "single")
            if qtype not in valid_types:
                qtype = "single"
            options = q.get("options", [])
            if not isinstance(options, list):
                options = []
            options = [str(o).strip() for o in options if str(o).strip()]
            cleaned.append({
                "text": str(q["text"]).strip(),
                "type": qtype,
                "options": options,
                "branching": str(q.get("branching", "")).strip(),
            })
        return cleaned

    @staticmethod
    def _question_key(question: dict) -> str:
        """겹치는 구간에서 반복 인식된 같은 문항을 식별할 키."""
        text = unicodedata.normalize("NFKC", str(question.get("text", "")))
        text = re.sub(r"^\s*(?:문항\s*)?[0-9０-９]+\s*[.)、:：-]?\s*", "", text)
        return re.sub(r"\s+", " ", text).strip().casefold()

    @classmethod
    def _parse_chunk(cls, chunk_text: str, provider_id: str,
                     chunk_index: int, chunk_total: int) -> list[dict]:
        from .providers import generate
        result = generate(
            provider_id,
            cls._parse_prompt(chunk_text, chunk_index, chunk_total),
        ).strip()
        # JSON 코드블록을 허용하되, 최종 출력은 배열만 처리한다.
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
        try:
            questions = json.loads(result)
        except json.JSONDecodeError:
            return []
        return cls._clean_questions(questions)

    @classmethod
    def parse_questions(cls, pdf_text: str,
                        provider_id: str = "gemini") -> list[dict]:
        """AI를 이용해 설문 텍스트 전체에서 문항을 파싱합니다.

        Returns:
            [{"text": ..., "type": "single|multi|likert|open",
              "options": [...], "branching": "..."}, ...]
            branching은 분기 로직 설명 문자열 (없으면 "")
        """
        chunks = cls._split_text(
            pdf_text,
            QUESTIONNAIRE_PARSE_CHUNK_CHARS,
            QUESTIONNAIRE_PARSE_OVERLAP_CHARS,
        )
        cls.last_parse_stats = {
            "total_chunks": len(chunks),
            "successful_chunks": 0,
            "failed_chunks": 0,
            "parsed_questions": 0,
            "unique_questions": 0,
            "duplicates_removed": 0,
        }
        if not chunks:
            return []
        parsed: list[dict] = []
        last_error: Exception | None = None
        for index, chunk in enumerate(chunks, start=1):
            try:
                chunk_questions = cls._parse_chunk(
                    chunk, provider_id, index, len(chunks)
                )
                cls.last_parse_stats["successful_chunks"] += 1
                cls.last_parse_stats["parsed_questions"] += len(chunk_questions)
                parsed.extend(chunk_questions)
            except Exception as exc:
                # 한 구간의 일시적 실패가 전체 설문 인식을 막지 않게 한다.
                cls.last_parse_stats["failed_chunks"] += 1
                last_error = exc
        if not parsed and last_error is not None:
            raise last_error
        # 겹치는 구간에서 중복 인식된 문항 제거. 최초 등장 순서는 유지한다.
        unique: list[dict] = []
        seen: set[str] = set()
        for question in parsed:
            key = cls._question_key(question)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(question)
        cls.last_parse_stats["unique_questions"] = len(unique)
        cls.last_parse_stats["duplicates_removed"] = len(parsed) - len(unique)
        return unique
