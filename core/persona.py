"""페르소나 샘플링 (core/persona.py)

인구통계 분포 → 가상 응답자 프로필 생성.
고정밀 모드용 인터뷰 트랜스크립트 기반 텍스트도 여기서 만든다.
Nemotron 출처(Phase 1): 발췌본에서 쿼터대로 추출해 서술 기반 프로필을 만든다.
"""

import random

from config import (AGE_BANDS, NEMOTRON_EXTRACT_PATH, NEMOTRON_FACETS,
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


def nemotron_default_quotas(records: list[dict]) -> dict[str, dict[str, float]]:
    """발췌본 분포 그대로의 비율 쿼터 (소수 첫째 자리, 차원별 합계 100%).

    UI에서 Nemotron 쿼터의 초기값으로 쓴다. 사용자가 바꾸지 않으면
    패널은 모집단 분포를 그대로 닮는다.
    """
    quotas: dict[str, dict[str, float]] = {}
    for dim, field in _NEMOTRON_QUOTA_FIELDS.items():
        shares = _extract_shares(records, field)
        rounded = {k: round(v, 1) for k, v in shares.items()}
        diff = round(100.0 - sum(rounded.values()), 1)
        if diff:
            biggest = max(rounded, key=lambda k: rounded[k])
            rounded[biggest] = round(rounded[biggest] + diff, 1)
        quotas[dim] = rounded
    return quotas


def _record_cell(record: dict) -> tuple[str, str, str]:
    """레코드의 쿼터 셀 (연령대, 성별, 지역). 없는 값은 age/province에서 유도."""
    age_band = record.get("age_band") or age_band_of(record.get("age", 30))
    sex = record.get("sex", "")
    region = record.get("region") or region_of(record.get("province", ""))
    return (str(age_band), str(sex), str(region))


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
                             rng: random.Random) -> tuple[list[dict], dict]:
    """Nemotron 발췌본에서 비율 쿼터대로 비복원 추출한다.

    Args:
        records: load_nemotron_extract() 반환 레코드 목록.
        quotas: {"연령대": {"20대": 20.0, ...}, "성별": {...}, "지역": {...}}
            비율 쿼터. 차원을 생략하면 발췌본 분포대로 추출한다.
            None이면 전부 발췌본 분포대로 추출한다.
        n: 추출할 인원.
        rng: random.Random(seed) — 같은 seed·같은 쿼터면 같은 패널이 나온다.

    Returns:
        (추출된 레코드 목록, 메타 dict).
        메타: 출처, 쿼터 달성률, 정규화 여부, 완화 기록(셀 부족 시).
    """
    if not records:
        raise ValueError("Nemotron 발췌본 레코드가 비어 있습니다.")
    if n <= 0:
        return [], {
            "source": "nemotron", "n_requested": n, "n_sampled": 0,
            "quotas": {}, "normalized": False,
            "quota_achievement": {}, "relaxations": [], "relaxed_count": 0,
        }
    n = min(n, len(records))

    quotas = quotas or {}
    # 차원별 목표 비율 결정 (정하지 않은 차원은 발췌본 분포)
    target_shares: dict[str, dict[str, float]] = {}
    normalized_any = False
    for dim, field in _NEMOTRON_QUOTA_FIELDS.items():
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
        by_cell.setdefault(_record_cell(rec), []).append(idx)

    used: set[int] = set()
    sampled_idx: list[int] = []
    relaxations: list[dict] = []

    def _take(pool: list[int], k: int) -> list[int]:
        """풀에서 아직 안 쓴 인덱스 k개를 비복원 추출한다."""
        avail = [i for i in pool if i not in used]
        take = rng.sample(avail, min(k, len(avail)))
        used.update(take)
        return take

    # 완화 풀: (연령대, 성별) / (성별, 지역) / 성별 / 전체
    def _pool_age_sex(cell):
        return [i for c, idxs in by_cell.items()
                if c[0] == cell[0] and c[1] == cell[1] for i in idxs]

    def _pool_sex_region(cell):
        return [i for c, idxs in by_cell.items()
                if c[1] == cell[1] and c[2] == cell[2] for i in idxs]

    def _pool_sex(cell):
        return [i for c, idxs in by_cell.items()
                if c[1] == cell[1] for i in idxs]

    all_idx = list(range(len(records)))
    # 셀 키 오름차순으로 처리해 rng 호출 순서를 고정한다.
    for cell in sorted(targets):
        want = targets[cell]
        if want <= 0:
            continue
        got = _take(by_cell.get(cell, []), want)
        shortfall = want - len(got)
        if shortfall > 0:
            # 1단계 완화: 지역 완화 (같은 연령대·성별, 다른 지역)
            extra = _take(_pool_age_sex(cell), shortfall)
            if extra:
                relaxations.append({"cell": cell, "shortfall": len(extra),
                                    "level": "region"})
            got += extra
            shortfall = want - len(got)
        if shortfall > 0:
            # 2단계 완화: 연령 완화 (같은 성별·지역, 다른 연령대)
            extra = _take(_pool_sex_region(cell), shortfall)
            if extra:
                relaxations.append({"cell": cell, "shortfall": len(extra),
                                    "level": "age"})
            got += extra
            shortfall = want - len(got)
        if shortfall > 0:
            # 3단계 완화: 같은 성별 내 임의
            extra = _take(_pool_sex(cell), shortfall)
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

    # 셀별 그룹 순서를 섞어 배치(기본 5명)마다 다양한 패널이 들어가게 한다.
    # rng 상태가 고정된 시점의 셔플이라 seed 재현성은 유지된다.
    rng.shuffle(sampled_idx)
    sampled = [records[i] for i in sampled_idx]

    # 쿼터 달성률 (차원별 목표 대비 실제)
    achievement: dict[str, dict[str, dict[str, float]]] = {}
    for dim, field in _NEMOTRON_QUOTA_FIELDS.items():
        actual = _extract_shares(sampled, field) if sampled else {}
        achievement[dim] = {
            val: {"target_pct": round(target_shares[dim].get(val, 0.0), 2),
                  "actual_pct": round(actual.get(val, 0.0), 2)}
            for val in sorted(set(target_shares[dim]) | set(actual))
        }

    meta = {
        "source": "nemotron",
        "n_requested": n,
        "n_sampled": len(sampled),
        "quotas": {dim: target_shares[dim] for dim in target_shares},
        "normalized": normalized_any,
        "quota_achievement": achievement,
        "relaxations": relaxations,
        "relaxed_count": sum(r["shortfall"] for r in relaxations),
    }
    return sampled, meta


_FACET_LABELS = {
    "persona": "요약",
    "professional_persona": "직업",
    "family_persona": "가족",
    "culinary_persona": "식생활",
}


def nemotron_persona_to_text(record: dict,
                             facets: tuple[str, ...] = NEMOTRON_FACETS) -> str:
    """Nemotron 레코드 1건을 프롬프트용 프로필 텍스트로 합성한다.

    [인구통계] 한 줄 + 측면별 서술. 전체 길이가 NEMOTRON_TEXT_MAX_CHARS를
    넘으면 뒤쪽 측면부터 잘라 상한 안에 넣는다 (인구통계·요약 우선 보존).
    """
    demo = (
        f"{record.get('sex', '')}, {record.get('age', '')}세"
        f"({record.get('age_band') or age_band_of(record.get('age', 30))}), "
        f"{record.get('district', '')}, {record.get('education_level', '')}, "
        f"{record.get('occupation', '')}, {record.get('marital_status', '')}"
    )
    parts = [f"[인구통계] {demo}"]
    for facet in facets:
        text = (record.get(facet) or "").strip()
        if not text:
            continue
        label = _FACET_LABELS.get(facet, facet)
        chunk = f"[{label}] {text}"
        current_len = sum(len(p) + 1 for p in parts)
        budget = NEMOTRON_TEXT_MAX_CHARS - current_len
        if budget <= 0:
            break
        parts.append(chunk[:budget] if len(chunk) > budget else chunk)
    return "\n".join(parts)


def build_persona_texts(personas: list[dict], rng: random.Random,
                        transcripts: list[dict] | None = None,
                        persona_source: str = "manual") -> list[str]:
    """응답자별 프로필 텍스트.

    우선순위: transcripts(고정밀) → nemotron → manual.
    transcripts가 주어지면 고정밀 모드(스탠포드 Generative Agents 방식):
    인터뷰 발췌문을 페르소나 컨텍스트로 사용한다.
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
            texts.append(nemotron_persona_to_text(p))
        else:
            texts.append(persona_to_text(p))
    return texts
