"""Synthetic Respondent Studio 핵심 로직.

방법론 (UI와 README에 투명하게 고지):
- '파인튜닝'이 아님: GPU 가중치 학습을 하지 않는다.
- 분포 학습: 업로드된 실제 응답 엑셀에서 문항별 응답 분포(경험분포)를 추출한다.
- 페르소나 시뮬레이션: LLM에 페르소나 프로필을 주고 가상 응답을 생성하게 한다.
"""

import hashlib
import io
import json
import random

import pandas as pd

from providers import generate

LIKERT_OPTIONS = ["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"]

QUESTION_TYPES = {
    "single": "단일선택",
    "multi": "다중선택",
    "likert": "5점 리커트",
    "open": "주관식",
}

MULTI_SEP = ";"

# 결과 화면·리포트에 고정 표시하는 정직한 한계 고지
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


# ----------------------------------------------------------------------------
# 설문 문항
# ----------------------------------------------------------------------------
def new_question_id(questions: list[dict]) -> str:
    nums = [int(q["id"][1:]) for q in questions if q["id"].startswith("Q") and q["id"][1:].isdigit()]
    return f"Q{(max(nums) + 1) if nums else 1}"


def question_summary(q: dict) -> str:
    base = f"{q['id']}. {q['text']} ({QUESTION_TYPES[q['type']]})"
    if q["type"] in ("single", "multi") and q.get("options"):
        base += " — " + ", ".join(q["options"])
    return base


# ----------------------------------------------------------------------------
# 페르소나
# ----------------------------------------------------------------------------
DEFAULT_PERSONA = {
    "성별": {"남성": 50.0, "여성": 50.0},
    "연령대": {"20대": 20.0, "30대": 25.0, "40대": 25.0, "50대": 20.0, "60대 이상": 10.0},
    "지역": {"수도권": 50.0, "영남권": 25.0, "호남권": 15.0, "충청권": 5.0, "강원·제주": 5.0},
    "직업": {"직장인": 45.0, "자영업": 15.0, "학생": 10.0, "주부": 15.0, "기타": 15.0},
}


def weighted_choice(dist: dict[str, float], rng: random.Random) -> str:
    total = sum(dist.values())
    if total <= 0:
        return next(iter(dist))
    r = rng.uniform(0, total)
    upto = 0.0
    for k, v in dist.items():
        upto += v
        if r <= upto:
            return k
    return next(reversed(dist))


def sample_persona(persona_dims: dict[str, dict[str, float]], rng: random.Random) -> dict[str, str]:
    return {dim: weighted_choice(dist, rng) for dim, dist in persona_dims.items()}


def persona_to_text(persona: dict[str, str]) -> str:
    return ", ".join(f"{dim} {val}" for dim, val in persona.items())


# ----------------------------------------------------------------------------
# 분포 학습
# ----------------------------------------------------------------------------
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


def learn_distributions(df: pd.DataFrame, questions: list[dict]) -> dict:
    """실제 응답 엑셀에서 문항별 응답 분포를 추출한다.

    Returns: {qid: {"matched_col": str, "type": str, "dist": {option: prob}, "n": int}}
    주관식은 분포를 만들지 않고 n만 기록한다.
    """
    learned = {}
    for q in questions:
        col = _match_column(df, q)
        if col is None:
            continue
        series = df[col].dropna()
        n = len(series)
        if n == 0:
            continue
        if q["type"] in ("single", "likert"):
            counts = series.astype(str).str.strip().value_counts(normalize=True)
            learned[q["id"]] = {
                "matched_col": str(col),
                "type": q["type"],
                "dist": {str(k): float(v) for k, v in counts.items()},
                "n": n,
            }
        elif q["type"] == "multi":
            exploded = []
            for val in series.astype(str):
                parts = [p.strip() for p in val.split(MULTI_SEP) if p.strip()]
                exploded.extend(parts)
            if exploded:
                counts = pd.Series(exploded).value_counts(normalize=True)
                learned[q["id"]] = {
                    "matched_col": str(col),
                    "type": q["type"],
                    "dist": {str(k): float(v) for k, v in counts.items()},
                    "n": n,
                }
        else:  # open
            learned[q["id"]] = {"matched_col": str(col), "type": "open", "dist": {}, "n": n}
    return learned


def sample_from_dist(dist: dict[str, float], rng: random.Random) -> str:
    return weighted_choice(dist, rng)


# ----------------------------------------------------------------------------
# 프롬프트 생성
# ----------------------------------------------------------------------------
def _question_block(questions: list[dict], include_open: bool = True) -> str:
    lines = []
    for q in questions:
        if q["type"] == "open" and not include_open:
            continue
        t = QUESTION_TYPES[q["type"]]
        if q["type"] in ("single", "multi"):
            opts = "/".join(q["options"])
            extra = " (복수 선택 가능, ';'로 구분)" if q["type"] == "multi" else ""
            lines.append(f"{q['id']}. {q['text']} [{t}: {opts}]{extra}")
        elif q["type"] == "likert":
            lines.append(f"{q['id']}. {q['text']} [{t}: {'/'.join(LIKERT_OPTIONS)}]")
        else:
            lines.append(f"{q['id']}. {q['text']} [주관식: 1-2문장으로 자연스럽게]")
    return "\n".join(lines)


def build_open_prompt(open_questions: list[dict], persona_texts: list[str]) -> str:
    profiles = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(persona_texts))
    return f"""당신은 설문조사 가상 응답자 시뮬레이터입니다.
아래 {len(persona_texts)}명의 응답자가 주관식 문항에 답한 것처럼 응답을 생성하세요.

[응답자 프로필]
{profiles}

[주관식 문항]
{_question_block(open_questions)}

[출력 형식]
JSON 배열로만 출력하세요. 설명이나 코드펜스 없이 아래 형식 그대로:
[{{"Q5": "..."}}, ...]
- 키는 문항 번호를 그대로 사용하세요.
- 구어체 1-2문장, 각 응답자의 프로필에 맞는 자연스러운 말투로 작성하세요.
- 모든 응답자가 비슷하게 답하지 마세요."""


def build_dist_prompt(choice_questions: list[dict], persona_texts: list[str]) -> str:
    """Prolific 2026 방식: 강제 단일 응답이 아닌 보기별 선택 확률분포를 추출한다."""
    profiles = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(persona_texts))
    lines = []
    for q in choice_questions:
        t = QUESTION_TYPES[q["type"]]
        if q["type"] in ("single", "likert"):
            opts = "/".join(q["options"] if q["type"] == "single" else LIKERT_OPTIONS)
            lines.append(f"{q['id']}. {q['text']} [{t} | 보기: {opts}]")
        else:  # multi: 각 보기별 독립 선택 확률
            opts = "/".join(q["options"])
            lines.append(f"{q['id']}. {q['text']} [{t} | 보기: {opts}] — 각 보기별 독립 선택확률")
    qblock = "\n".join(lines)
    return f"""당신은 설문조사 응답 분포 추정 전문가입니다.
각 응답자 프로필을 보고, 각 문항의 보기별 선택 확률(0~1 사이 실수)을 추정하세요.
특정 보기를 하나로 찍지 말고 반드시 확률분포 형태로 답하세요.

[응답자 프로필]
{profiles}

[문항]
{qblock}

[출력 형식]
JSON 배열로만 출력하세요. 설명이나 코드펜스 없이 아래 형식 그대로:
[{{"Q1": {{"보기1": 0.5, "보기2": 0.3}}, "Q3": {{"전혀 그렇지 않다": 0.1, ...}}}}, ...]
- 키는 문항 번호를 그대로 사용하세요.
- 단일선택/리커트는 한 문항의 확률 합이 1이 되게 하세요.
- 다중선택은 보기마다 독립적인 선택 확률(합이 1이 아니어도 됨)을 주세요.
- 프로필(연령·직업·지역 등)에 따라 현실적인 차이를 반영하세요."""


def normalize_dist(raw: dict, options: list[str]) -> dict[str, float]:
    """LLM이 준 확률을 정규화. 누락된 보기는 0, 합이 0이면 균등분포."""
    probs = {opt: max(0.0, float(raw.get(opt, 0) or 0)) for opt in options}
    total = sum(probs.values())
    if total <= 0:
        return {opt: 1.0 / len(options) for opt in options}
    return {opt: v / total for opt, v in probs.items()}


def sample_multi(pick_probs: dict[str, float], rng: random.Random) -> str:
    """다중선택: 보기별 독립 확률로 Bernoulli 샘플링. 최소 1개 보장."""
    picks = [opt for opt, p in pick_probs.items() if rng.random() < min(1.0, max(0.0, p))]
    if not picks:
        picks = [max(pick_probs, key=lambda o: pick_probs[o])]
    return MULTI_SEP.join(sorted(picks))


def parse_json_array(text: str) -> list:
    """LLM 출력에서 JSON 배열을 추출한다."""
    t = text.strip()
    for fence in ("```json", "```"):
        if t.startswith(fence):
            t = t[len(fence):].strip()
            if t.endswith("```"):
                t = t[: -3].strip()
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON 배열을 찾을 수 없습니다.")
    return json.loads(t[start : end + 1])


# ----------------------------------------------------------------------------
# 가상 응답 생성
# ----------------------------------------------------------------------------
def _cache_key(provider_id: str, model: str, prompt: str) -> str:
    return hashlib.md5(f"{provider_id}|{model}|{prompt}".encode("utf-8")).hexdigest()


def cached_generate(provider_id: str, model: str, prompt: str, cache: dict) -> str:
    key = _cache_key(provider_id, model, prompt)
    if key not in cache:
        cache[key] = generate(provider_id, prompt, model)
    return cache[key]


def _persona_texts(personas: list[dict], transcripts: list[dict] | None,
                 rng: random.Random) -> list[str]:
    """응답자별 프로필 텍스트. 고정밀 모드에서는 인터뷰 트랜스크립트 기반."""
    texts = []
    for p in personas:
        if transcripts:
            t = rng.choice(transcripts)
            excerpt = t["text"][:2500]
            texts.append(f"인터뷰 기반 페르소나 '{t['name']}' — 다음 실제 인터뷰 내용을 반영:\n{excerpt}")
        else:
            texts.append(persona_to_text(p))
    return texts


def generate_responses(
    questions: list[dict],
    persona_dims: dict[str, dict[str, float]],
    n: int,
    provider_id: str,
    model: str,
    learned: dict | None,
    batch_size: int = 5,
    progress_cb=None,
    cache: dict | None = None,
    seed: int | None = None,
    transcripts: list[dict] | None = None,
) -> pd.DataFrame:
    """가상 응답 n건을 생성해 DataFrame으로 반환.

    - 학습된 분포가 있는 선택형 문항: 경험분포에서 직접 샘플링 (API 호출 없음)
    - 학습된 분포가 없는 선택형 문항: Prolific 2026 방식 —
      LLM으로부터 페르소나별 보기별 선택 확률분포를 추출한 뒤 샘플링
    - 주관식: LLM 배치 생성
    - transcripts가 주어지면 고정밀 모드(인터뷰 기반 페르소나)로 동작
    """
    rng = random.Random(seed)
    cache = cache if cache is not None else {}
    learned = learned or {}
    transcripts = transcripts or None

    choice_qs = [q for q in questions if q["type"] in ("single", "multi", "likert")]
    open_qs = [q for q in questions if q["type"] == "open"]
    use_dist = {q["id"] for q in choice_qs if q["id"] in learned and learned[q["id"]].get("dist")}
    llm_dist_qs = [q for q in choice_qs if q["id"] not in use_dist]  # 분포 추출 대상

    rows: list[dict] = []
    total_batches = (n + batch_size - 1) // batch_size
    done = 0

    for b in range(total_batches):
        bsize = min(batch_size, n - b * batch_size)
        personas = [sample_persona(persona_dims, rng) for _ in range(bsize)]
        ptexts = _persona_texts(personas, transcripts, rng)

        # 1) 경험분포 샘플링 (학습된 선택형)
        sampled: list[dict] = []
        for _ in range(bsize):
            ans = {}
            for q in choice_qs:
                if q["id"] in use_dist:
                    dist = learned[q["id"]]["dist"]
                    if q["type"] == "multi":
                        picks = {sample_from_dist(dist, rng) for _ in range(rng.randint(1, min(3, len(dist))))}
                        ans[q["id"]] = MULTI_SEP.join(sorted(picks))
                    else:
                        ans[q["id"]] = sample_from_dist(dist, rng)
            sampled.append(ans)

        # 2) LLM 분포 추출 → 샘플링 (학습되지 않은 선택형, Prolific 2026 방식)
        if llm_dist_qs:
            raw = cached_generate(provider_id, model, build_dist_prompt(llm_dist_qs, ptexts), cache)
            parsed = parse_json_array(raw)
            if len(parsed) != bsize:
                raise ValueError(
                    f"LLM이 {bsize}건 요청에 {len(parsed)}건을 반환했습니다. 다시 시도해 주세요."
                )
            for i in range(bsize):
                for q in llm_dist_qs:
                    opts = q["options"] if q["type"] == "single" else (
                        LIKERT_OPTIONS if q["type"] == "likert" else q["options"])
                    praw = parsed[i].get(q["id"], {}) if isinstance(parsed[i], dict) else {}
                    if not isinstance(praw, dict):
                        continue
                    if q["type"] == "multi":
                        probs = {opt: min(1.0, max(0.0, float(praw.get(opt, 0) or 0))) for opt in opts}
                        if sum(probs.values()) <= 0:
                            probs = {opt: 0.5 for opt in opts}
                        sampled[i][q["id"]] = sample_multi(probs, rng)
                    else:
                        dist = normalize_dist(praw, opts)
                        sampled[i][q["id"]] = sample_from_dist(dist, rng)

        # 3) 주관식 LLM 생성
        if open_qs:
            raw = cached_generate(provider_id, model, build_open_prompt(open_qs, ptexts), cache)
            parsed = parse_json_array(raw)
            if len(parsed) != bsize:
                raise ValueError(
                    f"LLM이 {bsize}건 요청에 {len(parsed)}건을 반환했습니다. 다시 시도해 주세요."
                )
            for i in range(bsize):
                for q in open_qs:
                    if isinstance(parsed[i], dict) and q["id"] in parsed[i]:
                        sampled[i][q["id"]] = str(parsed[i][q["id"]]).strip()

        for i, persona in enumerate(personas):
            row = {f"프로필_{dim}": val for dim, val in persona.items()}
            for q in questions:
                row[q["id"]] = sampled[i].get(q["id"], "")
            rows.append(row)

        done += 1
        if progress_cb:
            progress_cb(done / total_batches)

    col_order = [f"프로필_{dim}" for dim in persona_dims] + [q["id"] for q in questions]
    df = pd.DataFrame(rows)
    return df[[c for c in col_order if c in df.columns]]


def to_excel_bytes(
    df: pd.DataFrame,
    sheet_name: str = "가상응답",
    meta: dict | None = None,
) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
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
    return buf.getvalue()
