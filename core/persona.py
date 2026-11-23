"""페르소나 샘플링 (core/persona.py)

인구통계 분포 → 가상 응답자 프로필 생성.
고정밀 모드용 인터뷰 트랜스크립트 기반 텍스트도 여기서 만든다.
Nemotron 출처(Phase 1): 발췌본에서 쿼터대로 추출해 서술 기반 프로필을 만든다.
"""

import random

from config import (AGE_BANDS, NEMOTRON_ALL_FACETS,
                    NEMOTRON_EXTRACT_PATH, NEMOTRON_FACETS,
                    NEMOTRON_FACET_LABELS, NEMOTRON_PROVINCES,
                    NEMOTRON_TEXT_MAX_CHARS, REGION_MAP,
                    TRANSCRIPT_EXCERPT_CHARS)


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


def sample_persona(persona_dims: dict[str, dict[str, float]],
                   rng: random.Random) -> dict[str, str]:
    """차원별 분포에서 가중 샘플링해 1명의 페르소나를 만든다."""
    return {dim: weighted_choice(dist, rng) for dim, dist in persona_dims.items()}


def persona_to_text(persona: dict[str, str]) -> str:
    return ", ".join(f"{dim} {val}" for dim, val in persona.items())


# ----------------------------------------------------------------------------
# Nemotron 출처 (Phase 1)
# ----------------------------------------------------------------------------
_extract_cache: dict[str, list[dict]] = {}

# 발췌본 레코드에 반드시 있어야 하는 컬럼 (쿼터·프로필 생성용)
_NEMOTRON_REQUIRED_COLUMNS = (
    "pid", "sex", "age", "age_band", "province", "district", "region",
    "education_level", "occupation", "marital_status",
    "persona", "professional_persona",
)

# 쿼터 차원 → 레코드 필드 매핑 (Phase 1: 비율 쿼터, 권역 단위 지역)
_NEMOTRON_QUOTA_FIELDS = {
    "연령대": "age_band",
    "성별": "sex",
    "지역": "region",
}


def clear_nemotron_cache() -> None:
    """발췌본 캐시를 비운다 (테스트용)."""
    _extract_cache.clear()


def load_nemotron_extract(path=None) -> list[dict]:
    """Nemotron 발췌본(Parquet)을 읽어 레코드 목록으로 반환한다.

    프로세스 내 캐시를 쓰므로 앱에서는 한 번만 읽는다.
    파일이 없거나 필수 컬럼이 빠지면 ValueError를 낸다.
    """
    import pandas as pd

    p = str(path) if path is not None else str(NEMOTRON_EXTRACT_PATH)
    if p not in _extract_cache:
        try:
            df = pd.read_parquet(p)
        except FileNotFoundError:
            raise ValueError(
                f"Nemotron 발췌본을 찾을 수 없습니다: {p}"
            ) from None
        except Exception as exc:
            raise ValueError(
                f"Nemotron 발췌본을 읽지 못했습니다: {p} ({exc})"
            ) from None
        missing = [c for c in _NEMOTRON_REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Nemotron 발췌본에 필수 컬럼이 없습니다: {missing}"
            )
        _extract_cache[p] = df.to_dict("records")
    return _extract_cache[p]


def age_band_of(age: int) -> str:
    """나이 → 연령대 밴드 (config.AGE_BANDS 기준)."""
    for label, lo, hi in AGE_BANDS:
        if lo <= int(age) <= hi:
            return label
    return AGE_BANDS[-1][0]


def region_of(province: str) -> str:
    """시도 → 권역 (config.REGION_MAP 기준)."""
    for region, provinces in REGION_MAP.items():
        if province in provinces:
            return region
    return "수도권"


def nemotron_default_quotas(records: list[dict],
                            region_level: str = "region"
                            ) -> dict[str, dict[str, float]]:
    """발췌본 분포 그대로의 비율 쿼터 (소수 첫째 자리, 차원별 합계 100%).

    UI에서 Nemotron 쿼터의 초기값으로 쓴다. 사용자가 바꾸지 않으면
    패널은 모집단 분포를 그대로 닮는다.
    region_level="province"면 지역 차원이 17개 시도 단위가 된다.
    """
    if region_level not in ("region", "province"):
        raise ValueError(
            f"region_level은 'region' 또는 'province'여야 합니다: {region_level}")
    quotas: dict[str, dict[str, float]] = {}
    for dim, field in _NEMOTRON_QUOTA_FIELDS.items():
        if dim == "지역" and region_level == "province":
            field = "province"
        shares = _extract_shares(records, field)
        rounded = {k: round(v, 1) for k, v in shares.items()}
        diff = round(100.0 - sum(rounded.values()), 1)
        if diff:
            biggest = max(rounded, key=lambda k: rounded[k])
            rounded[biggest] = round(rounded[biggest] + diff, 1)
        quotas[dim] = rounded
    return quotas


def provinces_in_region(region: str) -> list[str]:
    """권역에 속한 시도 목록 (config.REGION_MAP 기준)."""
    return list(REGION_MAP.get(region, []))


def nemotron_province_list(records: list[dict] | None = None) -> list[str]:
    """시도 목록. 레코드가 주어지면 발췌본에 실제 있는 시도만 정렬해 반환한다."""
    if records:
        present = sorted({str(r.get("province", "")) for r in records
                          if r.get("province")})
        if present:
            return present
    return list(NEMOTRON_PROVINCES)


def _record_cell(record: dict,
                 region_level: str = "region") -> tuple[str, str, str]:
    """레코드의 쿼터 셀 (연령대, 성별, 지역).

    region_level="region"이면 세 번째 원소가 권역(5개),
    "province"면 시도(17개)이다. 없는 값은 age/province에서 유도한다.
    """
    age_band = record.get("age_band") or age_band_of(record.get("age", 30))
    sex = record.get("sex", "")
    if region_level == "province":
        region_key = str(record.get("province", ""))
    else:
        region_key = record.get("region") or region_of(record.get("province", ""))
    return (str(age_band), str(sex), str(region_key))


def normalize_count_quotas(
        count_quotas: dict | None) -> dict[tuple[str, str, str], int]:
    """건수 쿼터를 {(연령대, 성별, 지역): 건수} 형태로 정규화한다.

    키는 튜플 ("20대", "여성", "서울") 또는 "20대|여성|서울" 문자열을 받는다.
    세 번째 원소는 권역(예: 수도권) 또는 시도(예: 서울) — region_level과
    맞춰서 쓰는 것은 호출부의 책임이다. 0 이하 건수는 버린다.
    """
    if not count_quotas:
        return {}
    out: dict[tuple[str, str, str], int] = {}
    for key, val in count_quotas.items():
        try:
            count = int(val)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        if isinstance(key, (tuple, list)) and len(key) == 3:
            cell = (str(key[0]), str(key[1]), str(key[2]))
        elif isinstance(key, str) and key.count("|") == 2:
            a, s, r = (p.strip() for p in key.split("|"))
            cell = (a, s, r)
        else:
            continue
        out[cell] = out.get(cell, 0) + count
    return out


def _count_quota_marginal_shares(
        counts: dict[tuple[str, str, str], int]
        ) -> dict[str, dict[str, float]]:
    """건수 쿼터에서 차원별 주변분포(%)를 유도한다 (달성률 계산용)."""
    total = sum(counts.values()) or 1
    marginals: dict[str, dict[str, float]] = {
        "연령대": {}, "성별": {}, "지역": {}}
    for (a, s, r), c in counts.items():
        for dim, val in (("연령대", a), ("성별", s), ("지역", r)):
            marginals[dim][val] = marginals[dim].get(val, 0.0) + c
    return {dim: {k: v / total * 100.0 for k, v in vals.items()}
            for dim, vals in marginals.items()}


def _normalize_quota(dist: dict[str, float]) -> tuple[dict[str, float], bool]:
    """비율 쿼터를 합계 100%로 정규화한다. (정규화된 값, 정규화 여부) 반환."""
    cleaned = {str(k): max(0.0, float(v)) for k, v in (dist or {}).items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {}, False
    if abs(total - 100.0) <= 0.01:
        return cleaned, False
    return {k: v / total * 100.0 for k, v in cleaned.items()}, True


def _extract_shares(records: list[dict], field: str) -> dict[str, float]:
    """발췌본 내 해당 필드의 분포(%) — 쿼터를 정하지 않은 차원의 기본값."""
    counts: dict[str, float] = {}
    for r in records:
        if field == "age_band":
            key = _record_cell(r)[0]
        elif field == "sex":
            key = str(r.get("sex", ""))
        elif field == "region":
            key = _record_cell(r)[2]
        else:
            key = str(r.get(field, ""))
        counts[key] = counts.get(key, 0.0) + 1.0
    total = sum(counts.values()) or 1.0
    return {k: v / total * 100.0 for k, v in counts.items()}


def _allocate_targets(cell_shares: dict[tuple[str, str, str], float],
                      n: int) -> dict[tuple[str, str, str], int]:
    """셀별 목표 비율을 정수 건수로 배분한다 (최대 나머지 방식, 합계 정확히 n건).

    동점 시 셀 키 오름차순으로 결정해 seed 재현성을 보장한다.
    """
    raw = {cell: share / 100.0 * n for cell, share in cell_shares.items()}
    targets = {cell: int(v) for cell, v in raw.items()}
    remainder = n - sum(targets.values())
    order = sorted(raw, key=lambda c: (-(raw[c] - int(raw[c])), c))
    i = 0
    while remainder > 0 and order:
        targets[order[i % len(order)]] += 1
        remainder -= 1
        i += 1
    return targets


def sample_nemotron_personas(records: list[dict],
                             quotas: dict[str, dict[str, float]] | None,
                             n: int,
                             rng: random.Random,
                             *,
                             count_quotas: dict | None = None,
                             region_level: str = "region"
                             ) -> tuple[list[dict], dict]:
    """Nemotron 발췌본에서 쿼터대로 비복원 추출한다.

    Args:
        records: load_nemotron_extract() 반환 레코드 목록.
        quotas: {"연령대": {"20대": 20.0, ...}, "성별": {...}, "지역": {...}}
            비율 쿼터. 차원을 생략하면 발췌본 분포대로 추출한다.
            None이면 전부 발췌본 분포대로 추출한다.
            region_level="province"면 "지역" 차원의 키가 시도명(17개)이다.
        n: 추출할 인원.
        rng: random.Random(seed) — 같은 seed·같은 쿼터면 같은 패널이 나온다.
        count_quotas: 셀별 목표 건수 (Phase 2).
            {(연령대, 성별, 지역): 건수} 또는 {"20대|여성|서울": 10} 형태.
            세 번째 원소는 region_level에 맞춰 권역 또는 시도명이다.
            비율 쿼터와 함께 오면 건수 쿼터가 우선하고 비율 쿼터는 무시한다.
            합계가 n과 다르면 메타(count_quota_warning)에 경고를 기록한다.
            합계가 n보다 적으면 부족분을 발췌본 분포대로 채우고,
            많으면 가장 큰 셀부터 줄여 n에 맞춘다(잘린 내역 메타 기록).
        region_level: "region"(권역 5개, 기본값) 또는 "province"(시도 17개).
            시도 단위 셀 부족 시 권역으로 완화한다.

    Returns:
        (추출된 레코드 목록, 메타 dict).
        메타: 출처, 쿼터 달성률, 정규화 여부, 완화 기록(셀 부족 시),
        quota_mode("ratio"/"count"), region_level, 건수 쿼터 경고·채움·절단 내역.
    """
    if region_level not in ("region", "province"):
        raise ValueError(
            f"region_level은 'region' 또는 'province'여야 합니다: {region_level}")
    if not records:
        raise ValueError("Nemotron 발췌본 레코드가 비어 있습니다.")
    if n <= 0:
        return [], {
            "source": "nemotron", "n_requested": n, "n_sampled": 0,
            "quotas": {}, "normalized": False,
            "quota_achievement": {}, "relaxations": [], "relaxed_count": 0,
            "quota_mode": "count" if count_quotas else "ratio",
            "region_level": region_level,
            "count_quota_warning": None, "count_quota_total": 0,
            "count_quota_fill": 0, "count_quota_truncated": [],
            "count_quota_achievement": {},
        }
    n = min(n, len(records))

    counts = normalize_count_quotas(count_quotas)
    count_mode = bool(counts)

    quotas = quotas or {}
    # 차원별 목표 비율 결정 (정하지 않은 차원은 발췌본 분포)
    target_shares: dict[str, dict[str, float]] = {}
    normalized_any = False
    for dim, field in _NEMOTRON_QUOTA_FIELDS.items():
        if dim == "지역" and region_level == "province":
            field = "province"
        if dim in quotas and quotas[dim]:
            normed, was_normed = _normalize_quota(quotas[dim])
            if normed:
                target_shares[dim] = normed
                normalized_any = normalized_any or was_normed
            else:
                target_shares[dim] = _extract_shares(records, field)
        else:
            target_shares[dim] = _extract_shares(records, field)

    # 셀별 목표 건수
    count_warning: str | None = None
    count_total = 0
    count_fill = 0
    count_truncated: list[dict] = []
    if count_mode:
        count_total = sum(counts.values())
        if count_total != n:
            count_warning = (
                f"건수 쿼터 합계 {count_total}건이 생성 인원 {n}건과 다릅니다. "
                f"({'부족분은 발췌본 분포대로 채웁니다' if count_total < n else '가장 큰 셀부터 줄여 맞춥니다'})"
            )
        if count_total > n:
            # 가장 큰 셀부터 줄여 n에 맞춘다 (동점은 셀 키 오름차순).
            targets = dict(counts)
            order = sorted(targets, key=lambda c: (-targets[c], c))
            excess = count_total - n
            i = 0
            while excess > 0:
                cell = order[i % len(order)]
                if targets[cell] > 0:
                    targets[cell] -= 1
                    excess -= 1
                i += 1
                if all(v == 0 for v in targets.values()):
                    break
            for cell in sorted(counts):
                if targets.get(cell, 0) != counts[cell]:
                    count_truncated.append({
                        "cell": cell, "from": counts[cell],
                        "to": targets.get(cell, 0)})
            targets = {c: v for c, v in targets.items() if v > 0}
        else:
            targets = dict(counts)
        # 달성률 계산용 주변분포는 (절단 후) 건수 쿼터에서 유도한다.
        target_shares = _count_quota_marginal_shares(targets)
    else:
        age_bands = sorted(target_shares["연령대"])
        sexes = sorted(target_shares["성별"])
        regions = sorted(target_shares["지역"])
        cell_shares = {
            (a, s, r): (target_shares["연령대"][a] / 100.0
                        * target_shares["성별"][s] / 100.0
                        * target_shares["지역"][r] / 100.0 * 100.0)
            for a in age_bands for s in sexes for r in regions
        }
        targets = _allocate_targets(cell_shares, n)

    # 셀별 인덱스 (발췌본 순서 유지 — seed 재현성의 기반)
    by_cell: dict[tuple[str, str, str], list[int]] = {}
    for idx, rec in enumerate(records):
        by_cell.setdefault(_record_cell(rec, region_level), []).append(idx)

    used: set[int] = set()
    sampled_idx: list[int] = []
    relaxations: list[dict] = []

    def _take(pool: list[int], k: int) -> list[int]:
        """풀에서 아직 안 쓴 인덱스 k개를 비복원 추출한다."""
        avail = [i for i in pool if i not in used]
        take = rng.sample(avail, min(k, len(avail)))
        used.update(take)
        return take

    # 완화 풀 — region_level에 따라 1·2·3단계의 의미가 달라진다.
    # region: (1) 지역 완화 — 같은 연령대·성별, 다른 지역
    #         (2) 연령 완화 — 같은 성별·지역, 다른 연령대
    #         (3) 같은 성별 내 임의
    # province: (1) 시도→권역 완화 — 같은 연령대·성별, 같은 권역의 다른 시도
    #           (2) 연령 완화 — 같은 성별·시도, 다른 연령대
    #           (3) 같은 성별·같은 권역 내 임의
    def _pool_relax1(cell):
        if region_level == "province":
            region = region_of(cell[2])
            return [i for c, idxs in by_cell.items()
                    if c[0] == cell[0] and c[1] == cell[1]
                    and region_of(c[2]) == region for i in idxs]
        return [i for c, idxs in by_cell.items()
                if c[0] == cell[0] and c[1] == cell[1] for i in idxs]

    def _pool_relax2(cell):
        return [i for c, idxs in by_cell.items()
                if c[1] == cell[1] and c[2] == cell[2] for i in idxs]

    def _pool_relax3(cell):
        if region_level == "province":
            region = region_of(cell[2])
            return [i for c, idxs in by_cell.items()
                    if c[1] == cell[1] and region_of(c[2]) == region
                    for i in idxs]
        return [i for c, idxs in by_cell.items()
                if c[1] == cell[1] for i in idxs]

    all_idx = list(range(len(records)))
    relax1_level = "province" if region_level == "province" else "region"
    # 셀 키 오름차순으로 처리해 rng 호출 순서를 고정한다.
    for cell in sorted(targets):
        want = targets[cell]
        if want <= 0:
            continue
        got = _take(by_cell.get(cell, []), want)
        shortfall = want - len(got)
        if shortfall > 0:
            # 1단계 완화: 지역(권역) 또는 시도→권역 완화
            extra = _take(_pool_relax1(cell), shortfall)
            if extra:
                relaxations.append({"cell": cell, "shortfall": len(extra),
                                    "level": relax1_level})
            got += extra
            shortfall = want - len(got)
        if shortfall > 0:
            # 2단계 완화: 연령 완화 (같은 성별·지역, 다른 연령대)
            extra = _take(_pool_relax2(cell), shortfall)
            if extra:
                relaxations.append({"cell": cell, "shortfall": len(extra),
                                    "level": "age"})
            got += extra
            shortfall = want - len(got)
        if shortfall > 0:
            # 3단계 완화: 같은 성별 내 임의
            extra = _take(_pool_relax3(cell), shortfall)
            if extra:
                relaxations.append({"cell": cell, "shortfall": len(extra),
                                    "level": "sex"})
            got += extra
            shortfall = want - len(got)
        if shortfall > 0:
            # 4단계 완화: 전체 분포에서 채움
            extra = _take(all_idx, shortfall)
            if extra:
                relaxations.append({"cell": cell, "shortfall": len(extra),
                                    "level": "overall"})
            got += extra
        sampled_idx.extend(got)

    # 건수 쿼터 합계가 n보다 적으면 부족분을 발췌본 분포대로 채운다.
    if count_mode and len(sampled_idx) < n:
        fill = _take(all_idx, n - len(sampled_idx))
        count_fill = len(fill)
        sampled_idx.extend(fill)

    # 셀별 그룹 순서를 섞어 배치(기본 5명)마다 다양한 패널이 들어가게 한다.
    # rng 상태가 고정된 시점의 셔플이라 seed 재현성은 유지된다.
    rng.shuffle(sampled_idx)
    sampled = [records[i] for i in sampled_idx]

    # 쿼터 달성률 (차원별 목표 대비 실제)
    achievement: dict[str, dict[str, dict[str, float]]] = {}
    for dim, field in _NEMOTRON_QUOTA_FIELDS.items():
        if dim == "지역" and region_level == "province":
            field = "province"
        actual = _extract_shares(sampled, field) if sampled else {}
        achievement[dim] = {
            val: {"target_pct": round(target_shares[dim].get(val, 0.0), 2),
                  "actual_pct": round(actual.get(val, 0.0), 2)}
            for val in sorted(set(target_shares[dim]) | set(actual))
        }

    # 건수 쿼터 모드: 셀별 목표 건수 대비 실제 건수 (완화되면 실제가 적다).
    count_achievement: dict[str, dict[str, int]] = {}
    if count_mode:
        actual_cells: dict[tuple[str, str, str], int] = {}
        for r in sampled:
            cell = _record_cell(r, region_level)
            actual_cells[cell] = actual_cells.get(cell, 0) + 1
        for cell in sorted(targets):
            key = "|".join(cell)
            count_achievement[key] = {
                "target_n": targets[cell],
                "actual_n": actual_cells.get(cell, 0),
            }

    meta = {
        "source": "nemotron",
        "n_requested": n,
        "n_sampled": len(sampled),
        "quotas": ({dim: target_shares[dim] for dim in target_shares}
                   if not count_mode else
                   {"건수": {"|".join(c): v for c, v in sorted(counts.items())}}),
        "normalized": normalized_any if not count_mode else False,
        "quota_achievement": achievement,
        "relaxations": relaxations,
        "relaxed_count": sum(r["shortfall"] for r in relaxations),
        "quota_mode": "count" if count_mode else "ratio",
        "region_level": region_level,
        "count_quota_warning": count_warning,
        "count_quota_total": count_total,
        "count_quota_fill": count_fill,
        "count_quota_truncated": count_truncated,
        "count_quota_achievement": count_achievement,
    }
    return sampled, meta


def normalize_nemotron_facets(
        facets: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """측면 목록을 정규화한다 (Phase 2).

    None이면 기본값(요약+직업). 알 수 없는 측면은 버리고, 중복은
    첫 등장 순서만 남긴다. 빈 튜플이면 인구통계만 쓴다는 뜻이다.
    발췌본에 없는 측면(travel/sports/arts)은 목록에는 남지만
    텍스트 합성 시 레코드에 값이 없어 자동으로 빠진다.
    """
    if facets is None:
        return NEMOTRON_FACETS
    seen: set[str] = set()
    out: list[str] = []
    for f in facets:
        f = str(f)
        if f in NEMOTRON_ALL_FACETS and f not in seen:
            seen.add(f)
            out.append(f)
    return tuple(out)


def nemotron_persona_to_text(record: dict,
                             facets: tuple[str, ...] = NEMOTRON_FACETS) -> str:
    """Nemotron 레코드 1건을 프롬프트용 프로필 텍스트로 합성한다.

    [인구통계] 한 줄 + 측면별 서술. 전체 길이가 NEMOTRON_TEXT_MAX_CHARS를
    넘으면 뒤쪽 측면부터 잘라 상한 안에 넣는다 (인구통계·요약 우선 보존).
    facets로 쓸 측면을 고른다 (기본값 요약+직업, Phase 2에서 주제별 선택).
    레코드에 없는 측면(발췌본 미포함 travel/sports/arts)은 건너뛴다.
    """
    demo = (
        f"{record.get('sex', '')}, {record.get('age', '')}세"
        f"({record.get('age_band') or age_band_of(record.get('age', 30))}), "
        f"{record.get('district', '')}, {record.get('education_level', '')}, "
        f"{record.get('occupation', '')}, {record.get('marital_status', '')}"
    )
    parts = [f"[인구통계] {demo}"]
    for facet in normalize_nemotron_facets(facets):
        text = (record.get(facet) or "").strip()
        if not text:
            continue
        label = NEMOTRON_FACET_LABELS.get(facet, facet)
        chunk = f"[{label}] {text}"
        current_len = sum(len(p) + 1 for p in parts)
        budget = NEMOTRON_TEXT_MAX_CHARS - current_len
        if budget <= 0:
            break
        parts.append(chunk[:budget] if len(chunk) > budget else chunk)
    return "\n".join(parts)


def build_persona_texts(personas: list[dict], rng: random.Random,
                        transcripts: list[dict] | None = None,
                        persona_source: str = "manual",
                        facets: tuple[str, ...] | list[str] | None = None
                        ) -> list[str]:
    """응답자별 프로필 텍스트.

    우선순위: transcripts(고정밀) → nemotron → manual.
    transcripts가 주어지면 고정밀 모드(스탠포드 Generative Agents 방식):
    인터뷰 발췌문을 페르소나 컨텍스트로 사용한다.
    facets는 Nemotron 모드에서만 쓰인다 (None이면 기본값 요약+직업).
    """
    texts = []
    for p in personas:
        if transcripts:
            t = rng.choice(transcripts)
            excerpt = t["text"][:TRANSCRIPT_EXCERPT_CHARS]
            texts.append(
                f"인터뷰 기반 페르소나 '{t['name']}' — 다음 실제 인터뷰 내용을 반영:\n{excerpt}"
            )
        elif persona_source == "nemotron":
            texts.append(nemotron_persona_to_text(p, facets=facets))
        else:
            texts.append(persona_to_text(p))
    return texts
