"""Nemotron-Personas-Korea 원본 다운로드 (Phase 0, 1회성).

Hugging Face `nvidia/Nemotron-Personas-Korea`의 `data/` 폴더(Parquet, 약 1.98GB)를
`~/workspace/prepanel/data/raw/` 로 내려받는다. hf_transfer 가속 사용.
중단 후 재실행하면 이미 받은 파일은 건너뛰고 이어받는다.

사용법:
    HF_HUB_ENABLE_HF_TRANSFER=1 python3 scripts/download_nemotron_raw.py \
        [--repo nvidia/Nemotron-Personas-Korea] [--out data/raw]
"""

import argparse
import os
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser(description="Nemotron-Personas-Korea 원본 다운로드")
    ap.add_argument("--repo", default="nvidia/Nemotron-Personas-Korea")
    ap.add_argument("--out", default="data/raw",
                    help="원본 저장 디렉토리 (기본: prepanel/data/raw)")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    print(f"repo: {args.repo}", flush=True)
    print(f"out:  {out_dir}", flush=True)
    t0 = time.time()
    path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=out_dir,
        allow_patterns=["data/*", "data/**"],
    )
    dt = time.time() - t0
    print(f"DONE in {dt:.1f}s ({dt / 60:.1f} min) -> {path}", flush=True)
    total = 0
    for root, _dirs, files in os.walk(os.path.join(out_dir, "data")):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    print(f"data/ total: {total / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
