"""파일럿 테스트 진단 리포트 — 규칙 기반 (환각 방지)


모든 진단은 실제 데이터에서 계산된 통계에만 근거합니다.

LLM이 문제를 '추측'하지 않으며, 임계값을 넘은 경우에만 보고합니다.

각 진단 결과는 (문제, 근거_수치, 해결책) 튜플로 반환됩니다.

"""

from __future__ import annotations


import re

import pandas as pd

from config import PANEL_POPULATION_DIFF_THRESHOLD_PP


# 임계값 (조정 가능)
CEILING_THRESHOLD = 0.85      # 한 보기가 85% 이상이면 천장/바닥 효과
LOW_VARIANCE_UNIQUE_RATIO = 0.1  # 고유 응답 비율 10% 미만이면 저분산
IDENTICAL_THRESHOLD = 0.3     # 30% 이상이 완전 동일 응답이면 의심
SHORT_ANSWER_LEN = 5          # 서술형 주관식 5자 미만은 부실 응답 (숫자·금액 응답 제외)

SHORT_ANSWER_RATIO = 0.3      # 부실 응답 30% 이상이면 문제

# 구조적 0 판정 임계값 — 추정확률이 이 값 이상인데 출현 0건이면 플래그.
# question_types.UNMATCHED_MASS_THRESHOLD(= 0.02, 워커 A)와 동일 값.
STRUCTURAL_ZERO_THRESHOLD = 0.02


# 숫자·금액형 주관식 응답에 붙을 수 있는 단위 접미사.
# "4명", "12000원", "12,000원", "3회", "1.5시간"처럼 숫자에 단위가 붙은 답은
# 길이가 짧아도 정상 응답이다. 접미사를 떼고 남은 부분이 숫자면 숫자 응답으로
# 본다. 텍스트 답이 우연히 단위로 끝나는 경우(예: "지원")는 남은 부분이 숫자가
# 아니라서 숫자 응답으로 오판하지 않는다.
# 복합 단위는 반복해서 뗀다 ("2만원" → "원" → "2만" → "만" → "2",
# "1천원" → "1", "3시" → "3"). 긴 접미사부터 검사해야 한다
# ("500ml"의 "ml"을 "m"보다 먼저, "3시간"의 "시간"을 "시"보다 먼저
# 떼기 위해).
_NUMERIC_UNIT_SUFFIXES = (
    "개월", "시간", "마리", "kg", "km", "cm", "mm", "ml",
    "원", "명", "개", "회", "번", "일", "분", "초", "세", "살", "점",
    "%", "주", "건", "대", "채", "권", "장", "줄", "년", "g", "L", "m",
    "만", "천", "억", "시",
)

_NUMERIC_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")

# astype(str)로 바뀐 결측 표기 — 무응답이지 짧은 답변이 아니다.
_MISSING_ANSWER_STRS = {"", "nan", "none", "null", "nat"}


def _is_missing_answer(value: str) -> bool:
    """무응답 판정: 빈 문자열·공백만·결측 표기('nan'/'None' 등)."""
    return str(value).strip().lower() in _MISSING_ANSWER_STRS


def _is_numeric_answer(value: str) -> bool:
    """숫자·금액 응답 판정: 숫자만, 또는 숫자+단위(명/원/회/시간 등).

    "4명" → "4", "12,000원" → "12000", "1.5시간" → "1.5"처럼 공백·쉼표·
    단위 접미사를 제거한 뒤 남은 부분이 숫자인지 본다. 복합 단위는
    반복 제거한다 ("2만원" → "원" → "2만" → "만" → "2", "1천원" → "1",
    "3시" → "3"). 긴 접미사부터 떼어 "500ml"의 "ml"을 "m"보다 먼저,
    "3시간"의 "시간"을 "시"보다 먼저 뗀다.
    """
    s = str(value).strip().replace(",", "").replace(" ", "")
    if not s:
        return False
    units = sorted(_NUMERIC_UNIT_SUFFIXES, key=len, reverse=True)
    while True:
        stripped = False
        for unit in units:
            if s.endswith(unit) and len(s) > len(unit):
                s = s[: -len(unit)]
                stripped = True
                break
        if not stripped:
            break
    return bool(_NUMERIC_RE.match(s))



def diagnose(df: pd.DataFrame, questions: list[dict],
               panel_reference: dict[str, dict[str, float]] | None = None,
               panel_threshold_pp: float = PANEL_POPULATION_DIFF_THRESHOLD_PP,
               estimated: dict[str, dict[str, float]] | None = None
               ) -> list[dict]:
    """데이터프레임과 문항 목록을 받아 진단 결과 리스트를 반환.

    각 결과: {"문항": str, "문제": str, "근거": str, "해결책": str, "심각도": str}

    문제가 없으면 빈 리스트를 반환합니다 (환각 방지: 없는 문제를 만들지 않음).

    panel_reference (Phase 2, 선택 사항): 패널 구성과 비교할 모집단 분포.
        {"성별": {"여성": 50.4, "남성": 49.6},
         "연령대": {"20대": 14.6, ...},
         "지역": {"수도권": 50.6, ...}} — 값은 % 단위.
        발췌본 분포는 reference_distribution_from_records()로 만든다.
        None이면 패널-모집단 비교 진단을 건너뛴다 (기존 호출부 동작 유지).
        차이가 panel_threshold_pp(기본 5%p) 이상인 셀이 있으면
        심각도 "참고"로 보고한다 — 응답 품질 판단이 아니라 패널 구성의
        투명성을 위한 것이다.

    estimated (2차 캘리브레이션 C0+C1, 선택 사항): 문항 id →
        {보기: 평균 추정확률}. None이면 이 진단을 건너뛴다 (기존 동작
        그대로). 주어지면 선택형 문항마다 추정확률이
        STRUCTURAL_ZERO_THRESHOLD(0.02) 이상인데 df에서 출현 0건인
        보기에 대해 '보기 미출현(구조적 0)' finding을 추가한다 —
        모델 회피가 아니라 매칭 손실(정확 일치 실패 시 0 반환 후
        재정규화)일 수 있다는 신호다.

    """
    findings: list[dict] = []
    if df is None or df.empty:
        return findings
    for q in questions:
        qid = q.get("id")
        if qid not in df.columns:
            continue
        qtext = q.get("text", qid)
        qtype = q.get("type", "")

        series = df[qid].astype(str)

        # 공란(무응답)은 하나의 응답 범주가 아니다 — 분기 스킵 등으로 답하지
        # 않은 문항을 천장/저분산으로 오진하지 않도록 §1·§2·§3 모두에서
        # 공란을 제외한 응답만으로 판정한다 (리뷰 필수 조치 4).
        is_missing = series.map(_is_missing_answer)
        answered = series[~is_missing]
        blank_count = int(is_missing.sum())
        blank_note = f" (공란 {blank_count}건 제외)" if blank_count else ""

        # 1. 천장/바닥 효과 (선택형)
        if qtype in ("single", "likert", "multiple"):
            vc = answered.value_counts(normalize=True)
            if not vc.empty and vc.iloc[0] >= CEILING_THRESHOLD:
                findings.append({
                    "문항": qtext,
                    "문제": "천장/바닥 효과 — 한 보기에 응답이 과도하게 쏠림",
                    "근거": f"'{vc.index[0]}' 보기가 {vc.iloc[0]*100:.1f}% (임계값 {CEILING_THRESHOLD*100:.0f}%){blank_note}",
                    "해결책": "보기 문구를 더 변별력 있게 나누거나, 척도 단계를 세분화하세요. 예: 5점→7점 척도.",
                    "심각도": "높음",
                })

        # 2. 저분산 (선택형: 사용된 보기가 전체 보기의 절반 미만일 때)
        if qtype in ("single", "likert", "multiple"):
            n_unique = answered.nunique()
            n_options = len(q.get("options", [])) or n_unique
            if n_options >= 4 and n_unique < n_options / 2 and len(answered) >= 20:
                findings.append({
                    "문항": qtext,
                    "문제": "저분산 — 보기가 충분히 활용되지 않음",
                    "근거": f"사용된 보기 {n_unique}개 / 전체 보기 {n_options}개{blank_note}",
                    "해결책": "사용되지 않은 보기가 왜 선택되지 않는지 검토하세요. 보기 문구를 현실적으로 조정하세요.",
                    "심각도": "중간",
                })

        # 3. 주관식 부실 응답
        #    숫자·금액 응답(예: "4명", "12000원")과 무응답은 부실 판정에서 제외한다.
        #    숫자 응답은 길이가 짧아도 정상이고, 무응답은 '짧은 답변'이 아니라
        #    분기 스킵 등으로 답하지 않은 것이다. 비율의 분모는 무응답을 제외한
        #    응답 건수다.

        if qtype in ("open", "text"):

            is_numeric = answered.map(_is_numeric_answer)
            text_answers = answered[~is_numeric]
            short_count = (text_answers.str.len() < SHORT_ANSWER_LEN).sum()
            short_ratio = short_count / max(len(answered), 1)

            if short_ratio >= SHORT_ANSWER_RATIO:

                findings.append({
                    "문항": qtext,
                    "문제": "부실 주관식 응답 — 너무 짧은 답변이 많음",
                    "근거": (f"{SHORT_ANSWER_LEN}자 미만 서술 답변이 {short_count}건 "
                             f"({short_ratio*100:.1f}%, 분모 {len(answered)}건) — "
                             f"숫자·금액 응답 {int(is_numeric.sum())}건과 "
                             f"무응답 {int(is_missing.sum())}건은 판정에서 제외"),
                    "해결책": "질문을 더 구체적으로 바꾸거나 예시를 제시하세요. 예: '구체적으로 어떤 점이...'",
                    "심각도": "중간",
                })

        # 3b. 보기 미출현(구조적 0) — 추정확률은 있는데 df에서 출현 0건.
        #     모델 회피가 아니라 매칭 손실일 수 있다. estimated가 None이면
        #     건너뛴다 (기존 호출부 동작 유지). 다중선택은 "A, B" 형태의
        #     결합 답 안에 보기가 토큰으로 들어 있으면 출현으로 본다.
        if estimated and qtype in ("single", "likert", "multiple"):
            est = estimated.get(qid)
            if isinstance(est, dict):
                if qtype == "multiple":
                    present: set[str] = set()
                    for v in answered:
                        present.update(
                            t.strip() for t in re.split(r"[,;]", str(v))
                            if t.strip())
                else:
                    present = set(answered.unique())
                for opt in (q.get("options") or []):
                    try:
                        p = float(est.get(str(opt), 0.0))
                    except (TypeError, ValueError):
                        continue
                    if (p >= STRUCTURAL_ZERO_THRESHOLD
                            and str(opt) not in present):
                        findings.append({
                            "문항": qtext,
                            "문제": "보기 미출현(구조적 0)",
                            "근거": (f"'{opt}' 보기 평균 추정확률 {p*100:.1f}%인데 "
                                     f"출현 0건 (응답 {len(answered)}건 중) — 매칭 손실 의심"),
                            "해결책": ("추정 확률과 매칭 결과를 확인하세요. 미매칭 질량(unmatched)과 "
                                       "재시도 기록을 보고, 보기 표기(공백·괄호·번호)가 LLM 출력과 "
                                       "일치하는지 점검하세요."),
                            "심각도": "높음",
                        })


    # 4. 전체 동일 응답 패턴 (2개 이상 문항이 있을 때만 의미 있음)
    if len(df) >= 10 and len(df.columns) >= 2:
        dup_ratio = 1 - len(df.drop_duplicates()) / len(df)
        if dup_ratio >= IDENTICAL_THRESHOLD:
            findings.append({
                "문항": "(전체)",
                "문제": "동일 응답 패턴 — 너무 많은 응답자가 완전히 같은 답을 함",
                "근거": f"중복 응답 비율 {dup_ratio*100:.1f}% (임계값 {IDENTICAL_THRESHOLD*100:.0f}%)",
                "해결책": "생성 다양성을 높이거나, 실제 파일럿에서는 응답자 모집 채널을 다양화하세요.",
                "심각도": "높음",
            })

    # 5. 패널-모집단 비교 (Phase 2 — panel_reference가 주어진 경우만)
    if panel_reference:
        findings.extend(_diagnose_panel_composition(
            df, panel_reference, panel_threshold_pp))
    return findings

# ----------------------------------------------------------------------------
# 패널-모집단 비교 진단 (Phase 2)
# ----------------------------------------------------------------------------
# 프로필 컬럼 → 참조 분포 차원 매핑
# 생성 결과 DataFrame의 프로필_ 컬럼과, 샘플링된 레코드 그대로의
# DataFrame(sex/age_band/region 컬럼)을 둘 다 받는다.
_PANEL_DIMS = {
    "프로필_성별": "성별",
    "프로필_연령대": "연령대",
    "프로필_지역": "지역",
    "sex": "성별",
    "age_band": "연령대",
    "region": "지역",
}


def reference_distribution_from_records(
        records: list[dict]) -> dict[str, dict[str, float]]:
    """Nemotron 발췌본 레코드에서 차원별 분포(%)를 만든다.

    diagnose()의 panel_reference 인자로 그대로 쓴다.

    발췌본 분포 ≒ 모집단 분포라는 전제는 설계안 §10과 같다.

    """
    dims: dict[str, dict[str, float]] = {
        "성별": {}, "연령대": {}, "지역": {}}
    for r in records:
        for dim, field in (("성별", "sex"), ("연령대", "age_band"),
                           ("지역", "region")):
            val = str(r.get(field, ""))
            if not val:
                continue
            dims[dim][val] = dims[dim].get(val, 0.0) + 1.0
    total = max(sum(dims["성별"].values()), 1.0)
    return {dim: {k: v / total * 100.0 for k, v in vals.items()}
            for dim, vals in dims.items()}


def _diagnose_panel_composition(
        df: pd.DataFrame,
        panel_reference: dict[str, dict[str, float]],
        threshold_pp: float) -> list[dict]:
    """패널의 연령대·성별·지역 분포와 참조(모집단) 분포를 비교한다.

    차이가 threshold_pp 이상인 셀이 있는 차원마다 1건씩,
    심각도 "참고"로 보고한다.

    """
    findings: list[dict] = []
    n = len(df)
    if n == 0:
        return findings
    seen_dims: set[str] = set()
    for col, dim in _PANEL_DIMS.items():
        if dim in seen_dims:
            continue
        ref = panel_reference.get(dim)
        if not ref or col not in df.columns:
            continue
        seen_dims.add(dim)
        actual = df[col].astype(str).value_counts(normalize=True) * 100
        deviating: list[str] = []
        for val in sorted(set(ref) | set(actual.index)):
            diff = actual.get(val, 0.0) - float(ref.get(val, 0.0))
            if abs(diff) >= threshold_pp:
                sign = "+" if diff >= 0 else ""
                deviating.append(
                    f"{val} {actual.get(val, 0.0):.1f}% "
                    f"(모집단 {float(ref.get(val, 0.0)):.1f}%, {sign}{diff:.1f}%p)")
        if deviating:
            findings.append({
                "문항": "(패널 구성)",
                "문제": f"패널-모집단 분포 차이 — {dim} 구성이 모집단과 다름",
                "근거": "; ".join(deviating),
                "해결책": "의도한 쿼터(특정 집단 과대표집 등)가 아니라면 쿼터를 "
                          "모집단 분포에 가깝게 조정하세요. 이 진단은 참고용이며 "
                          "응답 품질 판정이 아닙니다.",
                "심각도": "참고",
            })
    return findings


def findings_to_dataframe(findings: list[dict]) -> pd.DataFrame:
    """진단 결과를 엑셀 시트용 데이터프레임으로 변환."""
    if not findings:
        return pd.DataFrame(
            [["진단 결과: 문제가 발견되지 않았습니다."]],
            columns=["파일럿 진단 리포트"],
        )
    return pd.DataFrame(findings, columns=["문항", "문제", "근거", "해결책", "심각도"])
