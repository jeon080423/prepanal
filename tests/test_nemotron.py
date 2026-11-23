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
from core.diagnostics import diagnose, reference_distribution_from_records
from core.persona import (nemotron_default_quotas, nemotron_province_list,
                          normalize_count_quotas, normalize_nemotron_facets,
                          provinces_in_region)

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


# ----------------------------------------------------------------------------
# Phase 2 — 건수 쿼터
# ----------------------------------------------------------------------------
COUNT_QUOTAS_40 = {
    ("20대", "여성", "수도권"): 10,
    ("30대", "여성", "수도권"): 10,
    ("20대", "남성", "영남권"): 10,
    ("30대", "남성", "영남권"): 10,
}


def test_count_quota_achievement(records):
    sampled, meta = sample_nemotron_personas(
        records, None, 40, random.Random(21),
        count_quotas=COUNT_QUOTAS_40)
    assert len(sampled) == 40
    assert len({r["pid"] for r in sampled}) == 40  # 비복원
    assert meta["quota_mode"] == "count"
    assert meta["count_quota_warning"] is None
    assert meta["count_quota_total"] == 40
    assert meta["count_quota_fill"] == 0
    assert meta["relaxed_count"] == 0
    for key, vals in meta["count_quota_achievement"].items():
        assert vals["actual_n"] == vals["target_n"], key
    df = pd.DataFrame(sampled)
    assert ((df["age_band"] == "20대") & (df["sex"] == "여성")
            & (df["region"] == "수도권")).sum() == 10


def test_count_quota_string_keys(records):
    """건수 쿼터 키는 "연령대|성별|지역" 문자열도 받는다."""
    counts = normalize_count_quotas({"20대|여성|수도권": 10,
                                     "30대|여성|수도권": 10})
    assert counts == {("20대", "여성", "수도권"): 10,
                      ("30대", "여성", "수도권"): 10}
    assert normalize_count_quotas({"20대|여성|수도권": 0,
                                   "20대|여성": 5}) == {}


def test_count_quota_overrides_ratio(records):
    """비율 쿼터와 건수 쿼터가 함께 오면 건수가 우선한다."""
    ratio = {"연령대": {"70대 이상": 100.0},
             "성별": {"여성": 100.0},
             "지역": {"강원·제주": 100.0}}
    sampled, meta = sample_nemotron_personas(
        records, ratio, 40, random.Random(21),
        count_quotas=COUNT_QUOTAS_40)
    assert meta["quota_mode"] == "count"
    assert set(pd.DataFrame(sampled)["age_band"]) <= {"20대", "30대"}
    assert meta["count_quota_achievement"]["20대|여성|수도권"]["actual_n"] == 10


def test_count_quota_total_below_n_fills_from_distribution(records):
    """건수 합계가 n보다 적으면 경고를 남기고 부족분을 분포대로 채운다."""
    counts = {("20대", "여성", "수도권"): 5, ("30대", "여성", "수도권"): 5}
    sampled, meta = sample_nemotron_personas(
        records, None, 20, random.Random(22), count_quotas=counts)
    assert len(sampled) == 20
    assert meta["count_quota_warning"] is not None
    assert "10건" in meta["count_quota_warning"]
    assert meta["count_quota_fill"] == 10
    # 채움은 발췌본 분포대로 무작위라 목표 셀에 더 들어갈 수 있다
    # (목표 건수 이상 보장, 초과분은 채움의 일부).
    assert meta["count_quota_achievement"]["20대|여성|수도권"]["actual_n"] >= 5
    assert meta["count_quota_achievement"]["30대|여성|수도권"]["actual_n"] >= 5
    assert sum(v["actual_n"]
               for v in meta["count_quota_achievement"].values()) >= 10


def test_count_quota_total_above_n_truncates(records):
    """건수 합계가 n보다 많으면 경고를 남기고 가장 큰 셀부터 줄인다."""
    counts = {("20대", "여성", "수도권"): 10,
              ("30대", "여성", "수도권"): 10,
              ("20대", "남성", "영남권"): 10}
    sampled, meta = sample_nemotron_personas(
        records, None, 20, random.Random(23), count_quotas=counts)
    assert len(sampled) == 20
    assert meta["count_quota_warning"] is not None
    assert "30건" in meta["count_quota_warning"]
    assert meta["count_quota_truncated"]
    assert sum(t["to"] for t in meta["count_quota_truncated"]) + sum(
        v["target_n"] for k, v in meta["count_quota_achievement"].items()
        if k not in {("|".join(map(str, t["cell"])))
                     for t in meta["count_quota_truncated"]}) == 20


def test_count_quota_reproducible(records):
    a, _ = sample_nemotron_personas(records, None, 40, random.Random(42),
                                    count_quotas=COUNT_QUOTAS_40)
    b, _ = sample_nemotron_personas(records, None, 40, random.Random(42),
                                    count_quotas=COUNT_QUOTAS_40)
    assert [r["pid"] for r in a] == [r["pid"] for r in b]


# ----------------------------------------------------------------------------
# Phase 2 — 시도 단위 쿼터
# ----------------------------------------------------------------------------
def test_province_default_quotas(records):
    quotas = nemotron_default_quotas(records, region_level="province")
    assert sorted(quotas["지역"]) == nemotron_province_list(records)
    assert len(quotas["지역"]) == 17
    assert abs(sum(quotas["지역"].values()) - 100.0) < 0.01
    # 권역 단위는 기존과 동일 (5개)
    assert len(nemotron_default_quotas(records)["지역"]) == 5


def test_province_level_ratio_sampling(records):
    quotas = {"연령대": {"20대": 50.0, "30대": 50.0},
              "성별": {"여성": 100.0},
              "지역": {"서울": 100.0}}
    sampled, meta = sample_nemotron_personas(
        records, quotas, 50, random.Random(31), region_level="province")
    assert len(sampled) == 50
    assert meta["region_level"] == "province"
    assert meta["quota_mode"] == "ratio"
    assert {r["province"] for r in sampled} == {"서울"}
    assert set(meta["quota_achievement"]["지역"]) == {"서울"}
    assert meta["relaxed_count"] == 0


def test_province_level_relaxation_to_region(records):
    """시도 단위 셀 부족 시 권역으로 완화하고 메타에 기록한다."""
    quotas = {"연령대": {"70대 이상": 100.0},
              "성별": {"여성": 100.0},
              "지역": {"제주": 100.0}}
    sampled, meta = sample_nemotron_personas(
        records, quotas, 200, random.Random(32), region_level="province")
    assert len(sampled) == 200
    assert len({r["pid"] for r in sampled}) == 200
    assert meta["relaxed_count"] > 0
    assert any(r["level"] == "province" for r in meta["relaxations"])
    # 제주 70대 여성은 발췌본에 11건뿐 — 나머지는 완화로 채워짐
    exact = sum(1 for r in sampled
                if r["province"] == "제주" and r["age_band"] == "70대 이상"
                and r["sex"] == "여성")
    assert exact == 11
    assert exact + meta["relaxed_count"] == 200


def test_province_count_quota(records):
    """건수 쿼터도 시도 단위 셀을 받을 수 있다 (서울 20대 여성 10명)."""
    counts = {("20대", "여성", "서울"): 10, ("30대", "여성", "서울"): 10}
    sampled, meta = sample_nemotron_personas(
        records, None, 20, random.Random(33),
        count_quotas=counts, region_level="province")
    assert len(sampled) == 20
    assert meta["count_quota_achievement"]["20대|여성|서울"]["actual_n"] == 10
    assert meta["count_quota_achievement"]["30대|여성|서울"]["actual_n"] == 10
    assert meta["relaxed_count"] == 0


def test_provinces_in_region():
    assert provinces_in_region("수도권") == ["서울", "경기", "인천"]
    assert "제주" in provinces_in_region("강원·제주")
    assert provinces_in_region("없는권역") == []


# ----------------------------------------------------------------------------
# Phase 2 — 측면 선택
# ----------------------------------------------------------------------------
def test_facet_selection_in_text(records):
    rec = next(r for r in records if r.get("family_persona")
               and r.get("culinary_persona"))
    text = nemotron_persona_to_text(
        rec, facets=("persona", "family_persona", "culinary_persona"))
    assert "[요약]" in text
    assert "[가족]" in text
    assert "[식생활]" in text
    assert "[직업]" not in text
    # 기본값은 기존과 동일하게 요약+직업만
    default = nemotron_persona_to_text(rec)
    assert "[요약]" in default and "[직업]" in default
    assert "[가족]" not in default and "[식생활]" not in default


def test_facet_normalization():
    assert normalize_nemotron_facets(None) == ("persona", "professional_persona")
    assert normalize_nemotron_facets(
        ["family_persona", "family_persona", "unknown_facet",
         "culinary_persona"]) == ("family_persona", "culinary_persona")
    assert normalize_nemotron_facets(()) == ()
    # 발췌본에 없는 측면도 목록에는 남는다 (확장 시 자동 활성화)
    assert "travel_persona" in normalize_nemotron_facets(("travel_persona",))


def test_facet_missing_in_extract_is_skipped(records):
    """발췌본에 없는 측면(travel)은 텍스트에서 그냥 빠진다."""
    text = nemotron_persona_to_text(
        records[0], facets=("persona", "travel_persona", "sports_persona"))
    assert "[요약]" in text
    assert "[여행]" not in text and "[스포츠]" not in text


def test_build_persona_texts_facets(records):
    recs = records[:2]
    out = build_persona_texts(
        recs, random.Random(1), persona_source="nemotron",
        facets=("persona", "family_persona"))
    assert all("[가족]" in t for t in out)
    assert all("[직업]" not in t for t in out)
    # facets를 안 주면 기본값(요약+직업)
    out = build_persona_texts(recs, random.Random(1), persona_source="nemotron")
    assert all("[직업]" in t for t in out)


def test_generate_nemotron_facets_reach_prompt(mixed_questions):
    """GenerationService → GenerationContext → 프롬프트까지 측면이 전달된다."""
    prompts: list[str] = []

    def capturing_generate(provider_id, prompt, model):
        prompts.append(prompt)
        return fake_generate(provider_id, prompt, model)

    persona_dims = {"성별": {"남성": 50.0, "여성": 50.0},
                    "연령대": {"20대": 50.0, "30대": 50.0}}
    svc = GenerationService()
    svc.generate_responses(
        mixed_questions, persona_dims, 5,
        provider_id="gemini", model="dummy",
        batch_size=5, seed=42,
        generate_fn=capturing_generate,
        persona_source="nemotron",
        nemotron_facets=("persona", "professional_persona",
                         "family_persona", "culinary_persona"),
    )
    joined = "\n".join(prompts)
    assert "[가족]" in joined
    assert "[식생활]" in joined


def test_generate_nemotron_count_and_province(mixed_questions):
    """서비스 레벨: 건수 쿼터 + 시도 단위 쿼터 전달."""
    persona_dims = {"성별": {"남성": 50.0, "여성": 50.0},
                    "연령대": {"20대": 50.0, "30대": 50.0}}
    svc = GenerationService()
    df = svc.generate_responses(
        mixed_questions, persona_dims, 20,
        provider_id="gemini", model="dummy",
        batch_size=5, seed=42,
        generate_fn=fake_generate,
        persona_source="nemotron",
        nemotron_count_quotas={("20대", "여성", "서울"): 10,
                               ("30대", "여성", "서울"): 10},
        nemotron_region_level="province",
    )
    assert len(df) == 20
    meta = svc.last_nemotron_meta
    assert meta["quota_mode"] == "count"
    assert meta["region_level"] == "province"
    assert svc.last_gen_meta["쿼터 방식"] == "건수 쿼터"
    assert svc.last_gen_meta["지역 단위"] == "시도 (17개)"
    assert (df["프로필_세부지역"].str.startswith("서울-")).all()


# ----------------------------------------------------------------------------
# Phase 2 — 패널-모집단 비교 진단
# ----------------------------------------------------------------------------
def test_diagnose_panel_population_skewed(records):
    """쏠린 패널은 '참고' 수준으로 보고된다."""
    quotas = {"연령대": {"20대": 100.0},
              "성별": {"여성": 100.0},
              "지역": {"수도권": 100.0}}
    sampled, _ = sample_nemotron_personas(records, quotas, 100,
                                          random.Random(41))
    df = pd.DataFrame(sampled)
    ref = reference_distribution_from_records(records)
    findings = diagnose(df, [], panel_reference=ref)
    panel = [f for f in findings if f["문항"] == "(패널 구성)"]
    assert len(panel) == 3  # 성별·연령대·지역 모두 5%p 이상 차이
    assert all(f["심각도"] == "참고" for f in panel)
    assert any("성별" in f["문제"] for f in panel)
    assert any("연령대" in f["문제"] for f in panel)
    assert any("지역" in f["문제"] for f in panel)
    # 근거에 편차 수치가 들어간다
    assert "20대 100.0%" in panel[0]["근거"] or "20대 100.0%" in " ".join(
        f["근거"] for f in panel)


def test_diagnose_panel_population_no_reference_no_finding(records):
    """참조 분포를 안 주면 패널 진단은 나오지 않는다 (기존 호출부 유지)."""
    sampled, _ = sample_nemotron_personas(
        records, {"연령대": {"20대": 100.0}, "성별": {"여성": 100.0},
                  "지역": {"수도권": 100.0}}, 100, random.Random(41))
    df = pd.DataFrame(sampled)
    findings = diagnose(df, [])
    assert not [f for f in findings if f["문항"] == "(패널 구성)"]


def test_diagnose_panel_population_matching_reference(records):
    """참조와 같은 분포면 패널 진단은 나오지 않는다."""
    sampled, _ = sample_nemotron_personas(records, None, 100,
                                          random.Random(42))
    df = pd.DataFrame(sampled)
    ref = reference_distribution_from_records(sampled)
    findings = diagnose(df, [], panel_reference=ref)
    assert not [f for f in findings if f["문항"] == "(패널 구성)"]


def test_diagnose_panel_population_threshold(records):
    """임계값을 높이면 작은 차이는 보고되지 않는다."""
    quotas = {"연령대": {"20대": 100.0},
              "성별": {"여성": 100.0},
              "지역": {"수도권": 100.0}}
    sampled, _ = sample_nemotron_personas(records, quotas, 100,
                                          random.Random(41))
    df = pd.DataFrame(sampled)
    ref = reference_distribution_from_records(records)
    findings = diagnose(df, [], panel_reference=ref, panel_threshold_pp=90.0)
    assert not [f for f in findings if f["문항"] == "(패널 구성)"]


def test_reference_distribution_from_records(records):
    ref = reference_distribution_from_records(records)
    assert set(ref) == {"성별", "연령대", "지역"}
    for dim in ref:
        assert abs(sum(ref[dim].values()) - 100.0) < 0.01
    assert abs(ref["성별"]["여성"] - 50.4) < 0.1
