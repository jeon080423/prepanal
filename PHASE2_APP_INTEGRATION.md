# Phase 2 — app.py 통합 지시서 (갈래 1 작업분)

작성: 2026-11-23 (갈래 1 워커). core 변경은 완료됐고 테스트 62개 통과.
이 문서는 **app.py에만** 필요한 변경을 적는다. app.py는 다른 워커가 작업 중이라
갈래 1은 직접 고치지 않았다. 팀장이 아래대로 이어 붙이면 된다.

## 0. 전제 — 이미 구현된 core API (시그니처 그대로 쓸 것)

```python
# core/persona.py
sample_nemotron_personas(records, quotas, n, rng,
                         *, count_quotas=None, region_level="region")
#   count_quotas: {(연령대, 성별, 지역): 건수} 또는 {"20대|여성|서울": 10}
#     - 비율 쿼터와 함께 오면 건수가 우선, 비율은 무시됨 (meta["quota_mode"]="count")
#     - 합계 != n 이면 meta["count_quota_warning"]에 경고 기록
#       합계 < n → 부족분을 발췌본 분포대로 채움 (meta["count_quota_fill"])
#       합계 > n → 가장 큰 셀부터 줄여 n에 맞춤 (meta["count_quota_truncated"])
#   region_level: "region"(권역 5개, 기본값) | "province"(시도 17개)
#     - 시도 단위 셀 부족 시 권역으로 완화, 완화 level="province"로 메타 기록
#   meta 추가 키: quota_mode, region_level, count_quota_warning,
#     count_quota_total, count_quota_fill, count_quota_truncated,
#     count_quota_achievement (셀별 목표/실제 건수)

nemotron_default_quotas(records, region_level="region")
#   region_level="province"면 "지역" 차원이 17개 시도 키가 된다.

normalize_count_quotas(count_quotas)   # 문자열 키도 튜플 키로 정규화
nemotron_province_list(records=None)   # 시도 목록 (기본값: config 순서 17개)
provinces_in_region(region)            # 권역 → 시도 목록

normalize_nemotron_facets(facets)     # None → 기본값(요약+직업), 알 수 없는 측면 제거
nemotron_persona_to_text(record, facets=("persona", "professional_persona"))
build_persona_texts(personas, rng, transcripts=None,
                    persona_source="manual", facets=None)

# core/strategies.py — GenerationContext에 필드 추가됨
#   nemotron_facets: tuple[str, ...] | None = None (None이면 기본값 요약+직업)

# core/services.py — GenerationService.generate_responses()에 키워드 인자 추가됨
#   nemotron_count_quotas=None, nemotron_region_level="region",
#   nemotron_facets=None
#   last_gen_meta에 "쿼터 방식"("건수 쿼터"/"비율 쿼터"), "지역 단위"
#   ("시도 (17개)"/"권역 (5개)"), 건수 경고 시 "건수 쿼터 경고"/"건수 쿼터 채움"/
#   "건수 쿼터 절단" 키가 추가됨 — gen_meta.update()로 자동 표시됨.

# core/diagnostics.py
diagnose(df, questions, panel_reference=None, panel_threshold_pp=5.0)
reference_distribution_from_records(records)  # 발췌본 → {"성별": {...}, ...} (%)
#   panel_reference를 주면 ±5%p 이상 차이 나는 차원마다 심각도 "참고" 1건씩 보고.
#   None이면 패널 진단을 건너뛴다 (기존 호출부 동작 유지).
#   df는 생성 결과(프로필_성별/연령대/지역 컬럼) 또는 샘플링 레코드 그대로
#   (sex/age_band/region 컬럼) 둘 다 받는다.
```

```python
# config.py 추가 상수
NEMOTRON_ALL_FACETS       # 7종 전체 (persona, professional_persona, family_persona,
                          #   culinary_persona, travel_persona, sports_persona, arts_persona)
NEMOTRON_AVAILABLE_FACETS # 발췌본에 실제로 있는 4종 (위에서 travel/sports/arts 제외)
NEMOTRON_FACET_LABELS     # {"persona": "요약", "professional_persona": "직업",
                          #  "family_persona": "가족", "culinary_persona": "식생활",
                          #  "travel_persona": "여행", "sports_persona": "스포츠",
                          #  "arts_persona": "예술"}
NEMOTRON_PROVINCES        # 17개 시도 (발췌본 값과 일치: "경상북/경상남/전라남/전북/
                          #   충청북/충청남"은 약어 형태 — REGION_MAP과 동일 표기)
PANEL_POPULATION_DIFF_THRESHOLD_PP = 5.0
```

주의: `travel_persona`/`sports_persona`/`arts_persona`는 원본 스키마에는 있으나
10,000건 발췌본 컬럼에는 없다. UI에서 선택해도 텍스트에 포함되지 않으니,
선택지에 이 3종을 보여줄 때는 "(발췌본 미포함 — 선택해도 반영 안 됨)" 안내를 붙일 것.
발췌본 확대 시 코드 변경 없이 자동 활성화된다.

## 1. 스토어 키 추가 (app.py ~74-75행 set_default, ~729-730행·~788-789행 초기화)

```python
store.set_default("nemotron_quota_mode", "ratio")      # "ratio" | "count"
store.set_default("nemotron_count_quotas", None)       # {"20대|여성|서울": 10, ...}
store.set_default("nemotron_region_level", "region")   # "region" | "province"
store.set_default("nemotron_facets", None)             # None이면 기본값(요약+직업)
```

초기화/리셋 블록(기존 `persona_source`·`nemotron_quotas` 리셋 옆)에도 같은 키를
기본값으로 되돌리는 코드를 추가한다.

## 2. Step 2 페르소나 설계 화면 — `_render_nemotron_quota_ui()` 근처 (app.py ~1048행)

### (a) 쿼터 방식 선택 (함수 맨 앞)

```python
mode = st.radio("쿼터 방식", options=["ratio", "count"],
                format_func=lambda k: {"ratio": "비율 쿼터 (%)",
                                       "count": "건수 쿼터 (셀별 목표 건수)"}[k],
                index=0 if store.get("nemotron_quota_mode") != "count" else 1,
                horizontal=True, key="nemotron_quota_mode_radio")
store.set("nemotron_quota_mode", mode)
```

`mode == "count"`면 기존 비율 입력 UI는 접거나 비활성화하고 아래 건수 UI를 보여준다.
비율 UI를 숨기지 않고 그대로 둘 경우 "건수 쿼터가 비율 쿼터보다 우선합니다" 캡션을
반드시 붙일 것 (core는 건수만 쓰고 비율을 무시한다).

### (b) 건수 쿼터 입력 UI (mode == "count"일 때)

셀 추가 행 패턴 (세션 스토어는 JSON 직렬화 가능해야 하므로 **문자열 키**로 저장):

```python
from core.persona import normalize_count_quotas  # 파일 상단 import에 추가
counts = store.get("nemotron_count_quotas") or {}
c1, c2, c3, c4 = st.columns(4)
age = c1.selectbox("연령대", [label for label, _, _ in AGE_BANDS],
                   key="count_age")
sex = c2.selectbox("성별", ["여성", "남성"], key="count_sex")
# 지역 선택지는 region_level에 따라 달라진다 (아래 (c) 참조)
region_key = c3.selectbox("지역", <권역 5개 또는 시도 17개>, key="count_region")
n_cell = c4.number_input("건수", min_value=1, max_value=200, value=10,
                         step=1, key="count_n")
if st.button("셀 추가", key="count_add"):
    counts[f"{age}|{sex}|{region_key}"] = int(n_cell)
    store.set("nemotron_count_quotas", counts)
    st.rerun()
# 등록된 셀 목록 + 삭제 버튼 + 합계 표시
for key in sorted(counts):
    st.write(f"{key.replace('|', ' · ')} — {counts[key]}건")
total = sum(counts.values())
st.write(f"건수 쿼터 합계: {total}건")
st.caption("합계가 생성 인원과 다르면 생성 시 경고가 기록되고, 부족분은 발췌본 "
           "분포대로 채우거나 가장 큰 셀부터 줄여 맞춥니다. 생성 인원은 생성 단계에서 정합니다.")
```

삭제 버튼은 각 셀 행에 `key=f"count_del_{key}"`로 두고, 클릭 시 해당 키를 지우고 rerun.
`normalize_count_quotas()`는 문자열 키를 받아들이므로 저장 형태는 문자열 키 그대로 두면 된다.

### (c) 시도 단위 쿼터 선택 UI (비율·건수 공통, (a) 바로 아래)

```python
level = st.radio("지역 단위", options=["region", "province"],
                 format_func=lambda k: {"region": "권역 (5개)",
                                        "province": "시도 (17개)"}[k],
                 index=0 if store.get("nemotron_region_level") != "province" else 1,
                 horizontal=True, key="nemotron_region_level_radio")
if level != store.get("nemotron_region_level"):
    store.set("nemotron_region_level", level)
    # 지역 차원 키가 바뀌므로 비율 쿼터 초기값을 다시 만든다
    store.set("nemotron_quotas",
              nemotron_default_quotas(records, region_level=level))
    st.rerun()
```

- `_NEMOTRON_QUOTA_DIMS`의 "지역" 카테고리는 `region_level`에 따라
  `list(REGION_MAP.keys())`(권역) 또는 `list(NEMOTRON_PROVINCES)`(시도)로 바꾼다.
  (`NEMOTRON_PROVINCES`를 config import에 추가해야 한다.)
- "쿼터 초기화" 버튼은 `nemotron_default_quotas(records, region_level=level)`로 바꾼다.
- 건수 UI의 지역 selectbox도 같은 목록을 쓴다. 시도 단위 셀 부족 시 권역으로
  완화된다는 캡션을 붙인다 ("예: 제주 70대 여성은 발췌본에 11건뿐이라
  강원·제주 권역에서 채우고 완화 기록을 남깁니다").

### (d) 측면 선택 UI (Step 2 Nemotron 섹션 하단 — `_render_nemotron_quota_ui()` 끝 또는 step2() 안)

```python
from config import NEMOTRON_ALL_FACETS, NEMOTRON_FACET_LABELS, NEMOTRON_FACETS
facets = st.multiselect(
    "프롬프트에 쓸 페르소나 측면 (조사 주제별 선택)",
    options=list(NEMOTRON_ALL_FACETS),
    format_func=lambda f: NEMOTRON_FACET_LABELS[f],
    default=list(store.get("nemotron_facets") or NEMOTRON_FACETS),
    key="nemotron_facets_select",
)
store.set("nemotron_facets", facets or None)
st.caption("기본값은 요약+직업입니다. 예: 식생활 조사는 식생활·가족을 추가하세요. "
           "여행·스포츠·예술은 10,000건 발췌본에 없어 지금은 반영되지 않습니다.")
```

`facets`가 빈 리스트면 `None`으로 저장해 기본값(요약+직업)으로 돌아가게 한다.
(`normalize_nemotron_facets(())`는 인구통계만 쓴다는 뜻이라, 빈 선택을
기본값 복귀와 구분하기 위해 UI 단에서 None으로 변환한다.)

## 3. 생성 호출부 (app.py ~1469행 `gen_service.generate_responses(...)`)

기존 인자 뒤에 3개만 추가한다:

```python
df = gen_service.generate_responses(
    ...
    persona_source=store.get("persona_source") or "manual",
    nemotron_quotas=store.get("nemotron_quotas"),
    nemotron_count_quotas=(store.get("nemotron_count_quotas")
                           if store.get("nemotron_quota_mode") == "count"
                           else None),
    nemotron_region_level=store.get("nemotron_region_level") or "region",
    nemotron_facets=store.get("nemotron_facets"),
)
```

`last_gen_meta`는 `gen_meta.update()`로 이미 표시되므로 메타 표시 UI 변경은 필요 없다.
("쿼터 방식"·"지역 단위"·"건수 쿼터 경고" 행이 자동으로 추가된다.)

## 4. 진단 호출부 (app.py ~1556행 `findings = diagnose(...)`)

Nemotron 출처일 때만 발췌본 분포를 참조로 넘긴다:

```python
from core.diagnostics import diagnose, reference_distribution_from_records
panel_ref = None
if store.get("persona_source") == "nemotron":
    try:
        panel_ref = reference_distribution_from_records(load_nemotron_extract())
    except ValueError:
        panel_ref = None  # 발췌본 로드 실패 시 패널 진단은 생략
findings = diagnose(df, store.get("questions") or [], panel_reference=panel_ref)
```

- `load_nemotron_extract`는 `_render_nemotron_quota_ui()`에서 이미 쓰고 있어 import돼 있다.
- 심각도 "참고"는 기존 렌더링에서 `🔴`이 아닌 `🟡`로 표시된다 (높음만 🔴).
  그대로 두면 된다.
- 직접 분포 모드에서는 `panel_reference=None`이라 기존과 동일하게 동작한다.

## 5. 검증 체크리스트 (팀장용)

- [ ] 비율 쿼터 기존 동작: 직접 모드·Nemotron 비율 모드 회귀 테스트 통과
      (`python3 -m pytest tests/test_nemotron.py tests/test_core.py tests/test_registries.py -q`)
- [ ] 건수 쿼터 UI에서 "서울 20대 여성 10명"을 지정해 생성 → 결과 화면
      "쿼터 방식: 건수 쿼터", 프로필_세부지역이 "서울-"인 20대 여성 10건 확인
- [ ] 지역 단위를 시도로 바꿔 비율 쿼터 생성 → "지역 단위: 시도 (17개)" 표시
- [ ] 측면 선택에서 가족·식생활 추가 → 생성 프롬프트에 [가족]·[식생활] 포함
      (LLM 호출 로그나 fake 테스트 `test_generate_nemotron_facets_reach_prompt` 참조)
- [ ] 쏠린 쿼터(예: 여성 100%)로 생성 → 진단 리포트에 심각도 "참고"의
      "(패널 구성)" 항목 표시

## 6. 갈래 1이 건드리지 않은 것 (팀장 확인용)

- `app.py` 미수정 (다른 워커 작업 중 — 이 문서대로 팀장이 통합).
- `core/providers.py`, `core/variance.py`, `benchmarks/`, `tests/test_variance.py`,
  `UPDATE_LOG.md`, `PROJECT_MANAGEMENT_WALKTHROUGH.md` 미수정.
- 기존 함수 시그니처는 기본값 인자 추가로만 확장 (직접 분포 모드 기본값 유지).
