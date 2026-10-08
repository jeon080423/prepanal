"""샘플 설문 + 샘플 실제응답 엑셀 생성기. (데모용, 배포 산출물 아님)"""
import json
import random
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
rng = random.Random(42)

questions = [
    {"id": "Q1", "text": "주로 이용하는 온라인 쇼핑몰은 어디인가요?",
     "type": "single",
     "options": ["쿠팡", "네이버쇼핑", "G마켓", "11번가", "기타"]},
    {"id": "Q2", "text": "쇼핑몰 선택 시 중요하게 생각하는 요소는 무엇인가요? (복수 선택)",
     "type": "multi",
     "options": ["가격", "배송 속도", "상품 구색", "리뷰·평점", "고객 서비스"]},
    {"id": "Q3", "text": "전반적으로 온라인 쇼핑에 만족하시나요?",
     "type": "likert", "options": []},
    {"id": "Q4", "text": "해당 쇼핑몰을 다시 이용할 의향이 있으신가요?",
     "type": "likert", "options": []},
    {"id": "Q5", "text": "온라인 쇼핑에서 개선되었으면 하는 점을 자유롭게 적어주세요.",
     "type": "open", "options": []},
    {"id": "Q6", "text": "월 평균 온라인 쇼핑 횟수는 어떻게 되나요?",
     "type": "single",
     "options": ["1회 이하", "2~3회", "4~5회", "6회 이상"]},
]

LIKERT = ["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"]
Q1_DIST = [("쿠팡", 0.42), ("네이버쇼핑", 0.28), ("G마켓", 0.12), ("11번가", 0.08), ("기타", 0.10)]
Q2_P = {"가격": 0.75, "배송 속도": 0.68, "상품 구색": 0.45, "리뷰·평점": 0.52, "고객 서비스": 0.30}
Q3_DIST = [0.03, 0.07, 0.25, 0.40, 0.25]
Q4_DIST = [0.02, 0.06, 0.22, 0.42, 0.28]
Q6_DIST = [("1회 이하", 0.15), ("2~3회", 0.35), ("4~5회", 0.30), ("6회 이상", 0.20)]
OPEN_SAMPLES = [
    "배송은 빠른데 포장이 과한 것 같아요.",
    "가격 비교가 쉬워서 좋아요.",
    "반품 절차가 좀 더 간편했으면 좋겠어요.",
    "새벽배송이 안 되는 지역이라 아쉬워요.",
    "리뷰가 광고성 같아서 신뢰가 안 가요.",
    "쿠폰 알림이 너무 자주 와요.",
    "상품 설명이 실제와 다를 때가 있어요.",
    "전반적으로 만족합니다.",
]


def wpick(pairs):
    r = rng.random()
    upto = 0.0
    for k, p in pairs:
        upto += p
        if r <= upto:
            return k
    return pairs[-1][0]


rows = []
for _ in range(200):
    q2 = [o for o, p in Q2_P.items() if rng.random() < p] or ["가격"]
    rows.append({
        "주로 이용하는 온라인 쇼핑몰은 어디인가요?": wpick(Q1_DIST),
        "쇼핑몰 선택 시 중요하게 생각하는 요소는 무엇인가요? (복수 선택)": ";".join(sorted(q2)),
        "전반적으로 온라인 쇼핑에 만족하시나요?": rng.choices(LIKERT, weights=Q3_DIST)[0],
        "해당 쇼핑몰을 다시 이용할 의향이 있으신가요?": rng.choices(LIKERT, weights=Q4_DIST)[0],
        "온라인 쇼핑에서 개선되었으면 하는 점을 자유롭게 적어주세요.": rng.choice(OPEN_SAMPLES),
        "월 평균 온라인 쇼핑 횟수는 어떻게 되나요?": wpick(Q6_DIST),
    })

(BASE / "sample_questionnaire.json").write_text(
    json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
pd.DataFrame(rows).to_excel(BASE / "sample_real_data.xlsx", index=False)
print("samples written:", len(rows), "rows")
