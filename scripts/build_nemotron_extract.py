"""Nemotron-Personas-Korea 발췌본 빌드 스크립트 (Phase 0).

원본 100만 건(Parquet, `data/raw/data/*.parquet`)에서 계층추출로 10,000건을 뽑아
`data/nemotron_personas_10k.parquet`를 만든다.

설계 근거: `~/workspace/goals/synthetic-respondent-prepanel-platform-build-and-deployment/files/nemotron-personas-design.md`
- §5.2 포함 필드, §5.3 발췌본 10,000건, §6.4 연령 밴딩, §6.5 지역 매핑,
  §5.4 발췌본 검증, §15.3 이름은 고지로 대응(서술 속 이름은 건드리지 않음).

추출 방식 (2-pass, 메모리 절약):
- Pass 1: 인구통계 컬럼만 읽어 셀(연령대 x 성별 x 권역)별 건수를 센다.
- 목표 배분: 셀별 비례 배분 + 셀별 최소 건수(MIN_CELL) 보장, 합계가 정확히 N이 되도록 조정.
- Pass 2: 셀별 목표 건수만큼 비복원 무작위 추출 (seed 고정으로 재현 가능).

사용법:
    python3 scripts/build_nemotron_extract.py [--raw data/raw] [--out data/nemotron_personas_10k.parquet]
        [--n 10000] [--seed 42] [--min-cell 20] [--download]

    --download: 원본이 없으면 scripts/download_nemotron_raw.py를 먼저 실행한다.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 설계 상수 (설계안 §6.4, §6.5)
# ----------------------------------------------------------------------------
AGE_BANDS = [
    ("20대", 19, 29),
    ("30대", 30, 39),
    ("40대", 40, 49),
    ("50대", 50, 59),
    ("60대", 60, 69),
    ("70대 이상", 70, 200),
]

# 시도 -> 권역 매핑. 원본 province 표기는 "서울", "경기", "경상북" 등.
# "경북"/"경상북" 같은 이형도 함께 받는다.
REGION_MAP = {
    "수도권": {"서울", "경기", "인천"},
    "영남권": {"부산", "대구", "울산", "경북", "경남", "경상북", "경상남"},
    "호남권": {"광주", "전북", "전남", "전라북", "전라남"},
    "충청권": {"대전", "세종", "충북", "충남", "충청북", "충청남"},
    "강원·제주": {"강원", "제주", "강원도", "제주도"},
}
PROVINCE_TO_REGION = {p: r for r, ps in REGION_MAP.items() for p in ps}

SEX_LABEL = {"남자": "남성", "여자": "여성"}

# 원본에서 읽을 컬럼 (발췌본 포함 필드 + 셀 계산용)
READ_COLUMNS = [
    "uuid", "sex", "age", "province", "district",
    "education_level", "occupation", "marital_status",
    "persona", "professional_persona", "family_persona", "culinary_persona",
]

EXTRACT_COLUMNS = [
    "pid", "sex", "age", "age_band", "province", "district", "region",
    "education_level", "occupation", "marital_status",
    "persona", "professional_persona", "family_persona", "culinary_persona",
]


def age_band(age: int) -> str:
    for label, lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return label
    return "기타"


def allocate_targets(counts: dict, n: int, min_cell: int) -> dict:
    """셀별 목표 건수 배분: 비례 + 최소 건수 보장, 합계 정확히 n."""
    total = sum(counts.values())
    n_cells = len(counts)
    # 최소 건수 실현 가능성: n이 셀 수 x 최소 건수보다 작으면 최소 건수를 낮춘다.
    effective_min = min(min_cell, n // n_cells)
    if effective_min < 1:
        raise ValueError(f"추출 건수(n={n})가 셀 수({n_cells})보다 작아 배분 불가")
    if effective_min < min_cell:
        print(f"주의: 최소 건수를 {min_cell} -> {effective_min}건으로 낮춤 "
              f"(n={n}, 셀 {n_cells}개)", flush=True)
    targets = {}
    for cell, c in counts.items():
        targets[cell] = max(int(round(n * c / total)), effective_min)
    # 합계 조정: 초과분은 최소 건수를 초과하는 셀에서 비례 차감, 부족분은 가장 큰 셀에 배분
    diff = sum(targets.values()) - n
    order = sorted(targets, key=lambda k: targets[k], reverse=True)
    i = 0
    while diff > 0:
        cell = order[i % len(order)]
        if targets[cell] > effective_min:
            targets[cell] -= 1
            diff -= 1
        i += 1
        if i > 100 * len(order) and diff > 0:
            raise RuntimeError("목표 배분 조정 실패 (초과분 해소 불가)")
    while diff < 0:
        targets[order[0]] += 1
        diff += 1
    return targets


def find_parquet_files(raw_dir: Path) -> list[Path]:
    data_dir = raw_dir / "data"
    search = data_dir if data_dir.exists() else raw_dir
    files = sorted(search.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Parquet 파일을 찾을 수 없습니다: {search}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Nemotron 발췌본 빌드")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/nemotron_personas_10k.parquet")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-cell", type=int, default=20)
    ap.add_argument("--download", action="store_true",
                    help="원본이 없으면 다운로드 스크립트 실행")
    ap.add_argument("--metrics-out", default="data/extract_metrics.json")
    args = ap.parse_args()

    t_start = time.time()
    raw_dir = Path(args.raw)
    out_path = Path(args.out)
    metrics_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    files = []
    if raw_dir.exists():
        try:
            files = find_parquet_files(raw_dir)
        except FileNotFoundError:
            files = []
    if not files:
        if args.download:
            print("원본 없음 — 다운로드 스크립트 실행", flush=True)
            subprocess.run([sys.executable, "scripts/download_nemotron_raw.py",
                            "--out", str(raw_dir)], check=True)
            files = find_parquet_files(raw_dir)
        else:
            raise FileNotFoundError(
                f"원본 Parquet이 없습니다: {raw_dir} "
                "(--download 옵션을 쓰거나 scripts/download_nemotron_raw.py를 먼저 실행하세요)")
    print(f"원본 파일 {len(files)}개:", flush=True)
    for f in files:
        print(f"  {f} ({f.stat().st_size / 1e6:.1f} MB)", flush=True)

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Pass 1: 인구통계만 읽어 셀 분포 계산
    # ------------------------------------------------------------------
    t0 = time.time()
    demo_cols = ["sex", "age", "province"]
    demo_frames = []
    for f in files:
        demo_frames.append(pd.read_parquet(f, columns=demo_cols))
    demo = pd.concat(demo_frames, ignore_index=True)
    n_total = len(demo)
    print(f"Pass 1: 원본 {n_total:,}건 로드 ({time.time() - t0:.1f}s)", flush=True)
    if demo["sex"].isna().any() or demo["age"].isna().any() or demo["province"].isna().any():
        raise ValueError("필수 인구통계 컬럼에 결측치가 있습니다")

    demo["age_band"] = demo["age"].map(age_band)
    demo["region"] = demo["province"].map(PROVINCE_TO_REGION)
    unmapped = sorted(demo.loc[demo["region"].isna(), "province"].unique().tolist())
    if unmapped:
        raise ValueError(f"권역 매핑에 없는 시도 값: {unmapped}")
    demo["cell"] = list(zip(demo["age_band"], demo["sex"], demo["region"]))
    cell_counts = demo["cell"].value_counts().to_dict()
    print(f"셀 수: {len(cell_counts)}개 (연령대 6 x 성별 2 x 권역 5 = 60개 기대)", flush=True)
    print(f"가장 작은 셀: {min(cell_counts.values())}건, "
          f"가장 큰 셀: {max(cell_counts.values())}건", flush=True)

    targets = allocate_targets(cell_counts, args.n, args.min_cell)
    assert sum(targets.values()) == args.n, "목표 합계 불일치"
    print(f"목표 배분 완료: 셀별 최소 {min(targets.values())}건, "
          f"최대 {max(targets.values())}건, 합계 {sum(targets.values()):,}건", flush=True)

    # ------------------------------------------------------------------
    # Pass 2: 셀별 비복원 무작위 추출 (인덱스 선택 후 해당 행만 읽기)
    # ------------------------------------------------------------------
    t0 = time.time()
    cell_codes, uniques = pd.factorize(demo["cell"])
    cell_index = {cell: code for code, cell in enumerate(uniques)}
    selected_idx = np.zeros(n_total, dtype=bool)
    for cell, target in targets.items():
        idx = np.flatnonzero(cell_codes == cell_index[cell])
        if len(idx) < target:
            raise RuntimeError(f"셀 {cell}: 원본 {len(idx)}건 < 목표 {target}건")
        chosen = rng.choice(idx, size=target, replace=False)
        selected_idx[chosen] = True
    print(f"Pass 2a: 추출 인덱스 선택 완료 ({time.time() - t0:.1f}s, "
          f"선택 {selected_idx.sum():,}건)", flush=True)

    # 선택된 행만 전 컬럼으로 읽기 (파일별 오프셋 처리)
    t0 = time.time()
    parts = []
    offset = 0
    for f in files:
        df = pd.read_parquet(f, columns=READ_COLUMNS)
        mask = selected_idx[offset:offset + len(df)]
        if mask.any():
            parts.append(df.loc[mask])
        offset += len(df)
    assert offset == n_total
    ext = pd.concat(parts, ignore_index=True)
    print(f"Pass 2b: 선택 행 로드 완료 ({time.time() - t0:.1f}s, {len(ext):,}건)", flush=True)
    assert len(ext) == args.n, f"발췌본 건수 불일치: {len(ext)} != {args.n}"

    # ------------------------------------------------------------------
    # 발췌본 정리
    # ------------------------------------------------------------------
    ext = ext.rename(columns={"uuid": "pid"})
    ext["sex"] = ext["sex"].map(SEX_LABEL).fillna(ext["sex"])
    ext["age_band"] = ext["age"].map(age_band)
    ext["region"] = ext["province"].map(PROVINCE_TO_REGION)
    ext = ext[EXTRACT_COLUMNS]
    for col in ["persona", "professional_persona", "family_persona", "culinary_persona"]:
        n_missing = int(ext[col].isna().sum())
        if n_missing:
            print(f"경고: {col} 결측 {n_missing}건 (빈 문자열로 대체)", flush=True)
            ext[col] = ext[col].fillna("")
    ext = ext.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    ext["pid"] = [f"P{i + 1:05d}" for i in range(len(ext))]

    # ------------------------------------------------------------------
    # 저장 (Parquet 메타데이터에 출처 기록)
    # ------------------------------------------------------------------
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(ext, preserve_index=False)
    meta = {
        b"source": b"nvidia/Nemotron-Personas-Korea v1.0 (CC BY 4.0)",
        b"source_url": b"https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea",
        b"extract_n": str(args.n).encode(),
        b"extract_seed": str(args.seed).encode(),
        b"extract_min_cell": str(args.min_cell).encode(),
        b"build_date": time.strftime("%Y-%m-%d").encode(),
        b"note": ("Synthetic personas only. Names in persona texts are synthetic; "
                   "any resemblance to real persons is coincidental.").encode(),
    }
    table = table.replace_schema_metadata({**table.schema.metadata, **meta})
    pq.write_table(table, out_path, compression="zstd")
    size_mb = out_path.stat().st_size / 1e6
    print(f"저장 완료: {out_path} ({size_mb:.2f} MB)", flush=True)

    # ------------------------------------------------------------------
    # 검증: 분포 비교 + 텍스트 길이 + 로드 시간
    # ------------------------------------------------------------------
    def pct(series: pd.Series) -> dict:
        return (series.value_counts(normalize=True) * 100).round(2).to_dict()

    orig_age = pct(demo["age_band"])
    ext_age = pct(ext["age_band"])
    orig_sex = pct(demo["sex"])
    ext_sex = pct(ext["sex"].map({v: k for k, v in SEX_LABEL.items()}))
    orig_prov = pct(demo["province"])
    ext_prov = pct(ext["province"])
    orig_region = pct(demo["region"])
    ext_region = pct(ext["region"])

    def comp(orig: dict, extd: dict) -> dict:
        keys = sorted(set(orig) | set(extd))
        return {k: {"orig_pct": orig.get(k, 0.0), "ext_pct": extd.get(k, 0.0),
                    "diff_pp": round(extd.get(k, 0.0) - orig.get(k, 0.0), 2)}
                for k in keys}

    demo_line_len = (
        ext["sex"] + ", " + ext["age"].astype(str) + "세(" + ext["age_band"] + "), "
        + ext["province"] + "-" + ext["district"] + ", "
        + ext["education_level"].astype(str) + ", "
        + ext["occupation"].astype(str) + ", " + ext["marital_status"].astype(str)
    ).str.len()
    persona_len = ext["persona"].str.len()
    prof_len = ext["professional_persona"].str.len()
    combined_len = persona_len + prof_len + demo_line_len

    t0 = time.time()
    _reloaded = pd.read_parquet(out_path)
    load_s = time.time() - t0
    print(f"발췌본 로드 시간 실측: {load_s:.2f}s", flush=True)

    def len_stats(s: pd.Series) -> dict:
        return {"mean": round(float(s.mean()), 1), "median": float(s.median()),
                "min": int(s.min()), "max": int(s.max())}

    metrics = {
        "source": "nvidia/Nemotron-Personas-Korea v1.0",
        "source_url": "https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea",
        "license": "CC BY 4.0",
        "build_date": time.strftime("%Y-%m-%d"),
        "seed": args.seed,
        "min_cell": args.min_cell,
        "n_original": int(n_total),
        "n_extract": int(len(ext)),
        "extract_file": str(out_path),
        "extract_size_mb": round(size_mb, 2),
        "extract_load_seconds": round(load_s, 2),
        "n_cells": len(cell_counts),
        "min_cell_original": int(min(cell_counts.values())),
        "max_cell_original": int(max(cell_counts.values())),
        "min_cell_target": int(min(targets.values())),
        "max_cell_target": int(max(targets.values())),
        "distribution": {
            "age_band": comp(orig_age, ext_age),
            "sex": comp(orig_sex, ext_sex),
            "province": comp(orig_prov, ext_prov),
            "region": comp(orig_region, ext_region),
        },
        "text_length_chars": {
            "persona_summary": len_stats(persona_len),
            "professional_persona": len_stats(prof_len),
            "demographic_line": len_stats(demo_line_len),
            "combined_per_person": len_stats(combined_len),
            "combined_batch_of_5_estimate": {
                "mean": round(float(combined_len.mean()) * 5, 1),
                "max": round(float(combined_len.max()) * 5, 1),
            },
        },
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"검증 메트릭 저장: {metrics_path}", flush=True)

    max_diff = max(abs(v["diff_pp"])
                   for d in metrics["distribution"].values() for v in d.values())
    print(f"분포 최대 오차: {max_diff:.2f}%p", flush=True)
    print(f"텍스트 길이: 요약 평균 {persona_len.mean():.0f}자 / 직업 평균 {prof_len.mean():.0f}자 / "
          f"1인 합성 평균 {combined_len.mean():.0f}자 / 배치 5명 평균 {combined_len.mean() * 5:.0f}자",
          flush=True)
    print(f"전체 소요: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
