"""Nemotron 페르소나 연동 테스트 (Phase 1).

LLM 호출 없음 — 샘플링·텍스트 합성·메타 경로는 generate_fn 주입으로 검증한다.
E2E 실제 LLM 생성은 API 키가 없는 이 환경에서 수행하지 않는다.
"""
import json
import random
import re
from pathlib import Path

import pandas as pd
import pytest

from config import (NEMOTRON_EXTRACT_PATH, NEMOTRON_TEXT_MAX_CHARS)
from core import (GenerationService, build_persona_texts,
                  load_nemotron_extract, nemotron_persona_to_text,
                  sample_nemotron_personas)

EXTRACT_METRICS_PATH = Path(__file__).parent.parent / "data" / "extract_metrics.json"


@pytest.fixture(scope="module")
def records():
    return load_nemotron_extract()


# ----------------------------------------------------------------------------
# 발췌본
# ----------------------------------------------------------------------------
def test_extract_loads_10000_records(records):
    assert len(records) == 10_000
    assert NEMOTRON_EXTRACT_PATH.exists()


def test_extract_required_fields_present(records):
    required = ("pid", "sex", "age", "age_band", "province", "district",
                "region", "education_level", "occupation", "marital_status",
                "persona", "professional_persona")
    pids = set()
    for r in records:
        for col in required:
            assert r.get(col) not in (None, ""), f"{r.get('pid')}: {col} 비어 있음"
        assert r["pid"] not in pids
        pids.add(r["pid"])
    assert len(pids) == 10_000  # pid 고유


def test_extract_distribution_close_to_original(records):
    """발췌본 주변분포가 원본과 ±1%p 안에 드는지 (빌드 메트릭 대조)."""
    metrics = json.loads(EXTRACT_METRICS_PATH.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    sex_map = {"남자": "남성", "여자": "여성"}
    for dim, field in (("age_band", "age_band"), ("sex", "sex"),
                       ("region", "region")):
        orig = metrics["distribution"].get(dim)
        if not orig:
            continue
        actual = df[field].value_counts(normalize=True) * 100
        for raw_val, info in orig.items():
            val = sex_map.get(raw_val, raw_val)
            diff = abs(actual.get(val, 0.0) - info["orig_pct"])
            assert diff <= 1.0, f"{dim} {val}: 원본 대비 {diff:.2f}%p 차이"


# ----------------------------------------------------------------------------
# 샘플러
# ----------------------------------------------------------------------------
def test_sampler_reproducible(records):
    quotas = {"연령대": {"20대": 50.0, "30대": 50.0},
              "성별": {"여성": 100.0},
              "지역": {"수도권": 100.0}}
    a, _ = sample_nemotron_personas(records, quotas, 50, random.Random(42))
    b, _ = sample_nemotron_personas(records, quotas, 50, random.Random(42))
    assert [r["pid"] for r in a] == [r["pid"] for r in b]
    c, _ = sample_nemotron_personas(records, quotas, 50, random.Random(7))
    assert [r["pid"] for r in a] != [r["pid"] for r in c]


def test_sampler_quota_achievement(records):
    quotas = {"연령대": {"20대": 50.0, "30대": 50.0},
              "성별": {"여성": 80.0, "남성": 20.0},
              "지역": {"수도권": 70.0, "영남권": 30.0}}
    sampled, meta = sample_nemotron_personas(records, quotas, 200,
                                             random.Random(3))
    assert len(sampled) == 200
    assert len({r["pid"] for r in sampled}) == 200  # 비복원
    assert meta["relaxed_count"] == 0
    df = pd.DataFrame(sampled)
    assert abs(df["age_band"].value_counts(normalize=True).get("20대", 0) * 100 - 50) < 2
    assert abs(df["sex"].value_counts(normalize=True).get("여성", 0) * 100 - 80) < 2
    assert abs(df["region"].value_counts(normalize=True).get("수도권", 0) * 100 - 70) < 2


def test_sampler_no_quota_uses_extract_distribution(records):
    sampled, meta = sample_nemotron_personas(records, None, 200,
                                             random.Random(5))
    assert len(sampled) == 200
    assert meta["relaxed_count"] == 0
    df = pd.DataFrame(sampled)
    full = pd.DataFrame(records)
    for field in ("age_band", "sex", "region"):
        a = df[field].value_counts(normalize=True) * 100
        b = full[field].value_counts(normalize=True) * 100
        for val in b.index:
            assert abs(a.get(val, 0.0) - b[val]) < 5


def test_sampler_relaxation_recorded(records):
    """채울 수 없는 셀은 완화 규칙으로 채우고 메타에 기록한다."""
    quotas = {"연령대": {"70대 이상": 100.0},
              "성별": {"여성": 100.0},
              "지역": {"강원·제주": 100.0}}
    sampled, meta = sample_nemotron_personas(records, quotas, 200,
                                             random.Random(11))
    assert len(sampled) == 200
    assert len({r["pid"] for r in sampled}) == 200
    assert meta["relaxed_count"] > 0
    assert meta["relaxations"]  # 완화 기록 존재
    assert all("level" in r and "cell" in r for r in meta["relaxations"])
    # 완화 없이 채울 수 있는 셀은 33건 수준 — 나머지는 완화로 채워짐
    exact = sum(1 for r in sampled
                if r["age_band"] == "70대 이상" and r["sex"] == "여성"
                and r["region"] == "강원·제주")
    assert exact < 200
    assert exact + meta["relaxed_count"] == 200


def test_sampler_quota_normalization_recorded(records):
    quotas = {"연령대": {"20대": 25.0, "30대": 25.0}}  # 합계 50% → 정규화
    _, meta = sample_nemotron_personas(records, quotas, 20, random.Random(2))
    assert meta["normalized"] is True
    assert abs(sum(meta["quotas"]["연령대"].values()) - 100.0) < 0.01


# ----------------------------------------------------------------------------
# 텍스트 합성
# ----------------------------------------------------------------------------
def test_persona_text_contains_facets(records):
    text = nemotron_persona_to_text(records[0])
    assert "[인구통계]" in text
    assert "[요약]" in text
    assert "[직업]" in text
    assert records[0]["district"] in text
    assert records[0]["occupation"] in text


def test_persona_text_length_budget(records):
    rng = random.Random(9)
    batch = rng.sample(records, 50)
    texts = [nemotron_persona_to_text(r) for r in batch]
    assert all(len(t) <= NEMOTRON_TEXT_MAX_CHARS for t in texts)
    # 배치 5명 기준 프롬프트 프로필 부분 3,000자 이내
    for i in range(0, 50, 5):
        assert sum(len(t) for t in texts[i:i + 5]) < 3000


def test_build_persona_texts_priority(records):
    transcripts = [{"name": "t1", "text": "인터뷰 내용 " * 100}]
    recs = records[:2]
    # transcripts가 있으면 고정밀 우선
    out = build_persona_texts(recs, random.Random(1),
                              transcripts=transcripts,
                              persona_source="nemotron")
    assert all("인터뷰 기반" in t for t in out)
    # 없으면 Nemotron 서술
    out = build_persona_texts(recs, random.Random(1), persona_source="nemotron")
    assert all("[요약]" in t for t in out)
    # manual이면 평면 텍스트
    manual = [{"성별": "여성", "연령대": "30대"}]
    out = build_persona_texts(manual, random.Random(1))
    assert out == ["성별 여성, 연령대 30대"]


# ----------------------------------------------------------------------------
# 생성 서비스 (가짜 LLM — test_core.py와 같은 방식)
# ----------------------------------------------------------------------------
def _profile_count(prompt: str) -> int:
    return len(re.findall(r"(?m)^\d+\.\s", prompt))


def fake_generate(provider_id, prompt, model):
    n = _profile_count(prompt)
    assert n > 0
    if "[주관식 문항]" in prompt:
        return json.dumps([{"Q3": "배송이 빨라서 좋아요"} for _ in range(n)],
                          ensure_ascii=False)
    return json.dumps([{"Q2": {"예": 0.7, "아니오": 0.3}} for _ in range(n)],
                      ensure_ascii=False)


@pytest.fixture
def mixed_questions():
    return [
        {"id": "Q1", "text": "학습된 문항", "type": "single",
         "options": ["A", "B"]},
        {"id": "Q2", "text": "미학습 문항", "type": "single",
         "options": ["예", "아니오"]},
        {"id": "Q3", "text": "주관식 문항", "type": "open", "options": []},
    ]


def test_generate_nemotron_mode(mixed_questions):
    persona_dims = {"성별": {"남성": 50.0, "여성": 50.0},
                    "연령대": {"20대": 50.0, "30대": 50.0}}
    svc = GenerationService()
    df = svc.generate_responses(
        mixed_questions, persona_dims, 10,
        provider_id="gemini", model="dummy",
        batch_size=5, seed=42,
        generate_fn=fake_generate,
        persona_source="nemotron",
        nemotron_quotas=None,
    )
    assert list(df.columns) == [
        "프로필_성별", "프로필_연령대", "프로필_지역", "프로필_세부지역",
        "프로필_직업", "프로필_학력", "프로필_결혼상태",
        "Q1", "Q2", "Q3",
    ]
    assert len(df) == 10
    assert set(df["프로필_성별"].unique()) <= {"여성", "남성"}
    assert set(df["프로필_연령대"].unique()) <= {
        "20대", "30대", "40대", "50대", "60대", "70대 이상"}
    assert (df["Q3"] == "배송이 빨라서 좋아요").all()
    meta = svc.last_gen_meta
    assert meta is not None
    assert "Nemotron" in meta["페르소나 출처"]
    assert meta["쿼터 달성"] == "100% (완화 없음)"
    assert svc.last_nemotron_meta is not None
    assert svc.last_nemotron_meta["n_sampled"] == 10


def test_generate_nemotron_reproducible(mixed_questions):
    persona_dims = {"성별": {"남성": 50.0, "여성": 50.0},
                    "연령대": {"20대": 50.0, "30대": 50.0}}
    kw = dict(provider_id="gemini", model="dummy", batch_size=5, seed=42,
              generate_fn=fake_generate, persona_source="nemotron")
    df1 = GenerationService().generate_responses(mixed_questions, persona_dims, 10, **kw)
    df2 = GenerationService().generate_responses(mixed_questions, persona_dims, 10, **kw)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_manual_mode_unchanged(mixed_questions):
    """직접 분포 모드 회귀: 출력 형식이 기존과 동일하다."""
    persona_dims = {"성별": {"남성": 50.0, "여성": 50.0},
                    "연령대": {"20대": 50.0, "30대": 50.0}}
    learned = {"Q1": {"matched_col": "c", "type": "single",
                      "dist": {"A": 1.0, "B": 0.0}, "n": 50}}
    svc = GenerationService()
    df = svc.generate_responses(
        mixed_questions, persona_dims, 4,
        provider_id="gemini", model="dummy",
        learned=learned, batch_size=4, seed=42,
        generate_fn=fake_generate,
    )
    assert list(df.columns) == ["프로필_성별", "프로필_연령대",
                                "Q1", "Q2", "Q3"]
    assert len(df) == 4
    assert svc.last_gen_meta["페르소나 출처"] == "직접 분포 설정"


def test_excel_export_nemotron_meta_in_methodology_sheet(mixed_questions):
    """엑셀 방법론 시트에 Nemotron 출처·어필 문구가 들어간다."""
    import io

    from core import ExportService
    from openpyxl import load_workbook

    persona_dims = {"성별": {"남성": 50.0, "여성": 50.0},
                    "연령대": {"20대": 50.0, "30대": 50.0}}
    svc = GenerationService()
    df = svc.generate_responses(
        mixed_questions, persona_dims, 4,
        provider_id="gemini", model="dummy",
        batch_size=4, seed=42,
        generate_fn=fake_generate,
        persona_source="nemotron",
    )
    meta = {"페르소나 출처": svc.last_gen_meta["페르소나 출처"],
            "쿼터 달성": svc.last_gen_meta["쿼터 달성"]}
    data = ExportService.to_excel_bytes(df, meta=meta)
    wb = load_workbook(io.BytesIO(data))
    ws = wb["방법론_안내"]
    texts = " ".join(str(c.value) for row in ws.iter_rows() for c in row
                     if c.value)
    texts = re.sub(r"\s+", " ", texts)  # 셀·줄바꿈 경계의 공백 정규화
    assert "Nemotron-Personas-Korea" in texts
    assert "실측 분포" in texts
    assert "합성 데이터의 일부이며 실존 인물과 무관" in texts
    assert "Nemotron" in texts  # meta(출처) 포함
