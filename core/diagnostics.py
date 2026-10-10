"""파일럿 테스트 진단 리포트 — 규칙 기반 (환각 방지)

모든 진단은 실제 데이터에서 계산된 통계에만 근거합니다.
LLM이 문제를 '추측'하지 않으며, 임계값을 넘은 경우에만 보고합니다.
각 진단 결과는 (문제, 근거_수치, 해결책) 튜플로 반환됩니다.
"""
from __future__ import annotations

import pandas as pd


# 임계값 (조정 가능)
CEILING_THRESHOLD = 0.85      # 한 보기가 85% 이상이면 천장/바닥 효과
LOW_VARIANCE_UNIQUE_RATIO = 0.1  # 고유 응답 비율 10% 미만이면 저분산
IDENTICAL_THRESHOLD = 0.3     # 30% 이상이 완전 동일 응답이면 의심
SHORT_ANSWER_LEN = 5          # 주관식 5자 미만은 부실 응답
SHORT_ANSWER_RATIO = 0.3      # 부실 응답 30% 이상이면 문제


def diagnose(df: pd.DataFrame, questions: list[dict]) -> list[dict]:
    """데이터프레임과 문항 목록을 받아 진단 결과 리스트를 반환.

    각 결과: {"문항": str, "문제": str, "근거": str, "해결책": str, "심각도": str}
    문제가 없으면 빈 리스트를 반환합니다 (환각 방지: 없는 문제를 만들지 않음).
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

        # 1. 천장/바닥 효과 (선택형)
        if qtype in ("single", "likert", "multiple"):
            vc = series.value_counts(normalize=True)
            if not vc.empty and vc.iloc[0] >= CEILING_THRESHOLD:
                findings.append({
                    "문항": qtext,
                    "문제": "천장/바닥 효과 — 한 보기에 응답이 과도하게 쏠림",
                    "근거": f"'{vc.index[0]}' 보기가 {vc.iloc[0]*100:.1f}% (임계값 {CEILING_THRESHOLD*100:.0f}%)",
                    "해결책": "보기 문구를 더 변별력 있게 나누거나, 척도 단계를 세분화하세요. 예: 5점→7점 척도.",
                    "심각도": "높음",
                })

        # 2. 저분산 (선택형: 사용된 보기가 전체 보기의 절반 미만일 때)
        if qtype in ("single", "likert", "multiple"):
            n_unique = series.nunique()
            n_options = len(q.get("options", [])) or n_unique
            if n_options >= 4 and n_unique < n_options / 2 and len(series) >= 20:
                findings.append({
                    "문항": qtext,
                    "문제": "저분산 — 보기가 충분히 활용되지 않음",
                    "근거": f"사용된 보기 {n_unique}개 / 전체 보기 {n_options}개",
                    "해결책": "사용되지 않은 보기가 왜 선택되지 않는지 검토하세요. 보기 문구를 현실적으로 조정하세요.",
                    "심각도": "중간",
                })

        # 3. 주관식 부실 응답
        if qtype in ("open", "text"):
            short_count = (series.str.len() < SHORT_ANSWER_LEN).sum()
            short_ratio = short_count / max(len(series), 1)
            if short_ratio >= SHORT_ANSWER_RATIO:
                findings.append({
                    "문항": qtext,
                    "문제": "부실 주관식 응답 — 너무 짧은 답변이 많음",
                    "근거": f"{SHORT_ANSWER_LEN}자 미만 답변이 {short_count}건 ({short_ratio*100:.1f}%)",
                    "해결책": "질문을 더 구체적으로 바꾸거나 예시를 제시하세요. 예: '구체적으로 어떤 점이...'",
                    "심각도": "중간",
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

    return findings


def findings_to_dataframe(findings: list[dict]) -> pd.DataFrame:
    """진단 결과를 엑셀 시트용 데이터프레임으로 변환."""
    if not findings:
        return pd.DataFrame(
            [["진단 결과: 문제가 발견되지 않았습니다."]],
            columns=["파일럿 진단 리포트"],
        )
    return pd.DataFrame(findings, columns=["문항", "문제", "근거", "해결책", "심각도"])
