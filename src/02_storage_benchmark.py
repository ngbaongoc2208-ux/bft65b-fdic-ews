"""
Part 1 - Storage benchmark: CSV vs Parquet (snappy / zstd / gzip).

Measures (each repeated N times, mean +- std):
  - on-disk size
  - read all columns
  - read 5 columns only
  - filter one quarter
Output: reports/storage_benchmark.csv and printed table.

Usage (from repo root):  python src/02_storage_benchmark.py
"""
from __future__ import annotations

import platform
import shutil
import statistics as st
import time
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

RAW = Path("data_raw/fdic/financials")
WORK = Path("data_raw/bench")          # inside data_raw -> ignored by Git
REPORTS = Path("reports")
N_REPEAT = 5
COLS5 = ["CERT", "REPDTE", "ASSET", "EQ", "ROA"]
ONE_QUARTER = "20091231"


def dir_size_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1024**2
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024**2


def timeit(fn, n=N_REPEAT):
    fn()  # warm-up
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return st.mean(ts), (st.stdev(ts) if len(ts) > 1 else 0.0)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    print("Machine:", platform.platform(), "| Python", platform.python_version(),
          "| polars", pl.__version__, "| pyarrow", pa.__version__)

    # ---- load the raw panel once (as downloaded) -------------------------
    files = sorted(RAW.glob("*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    # REPDTE is stored as string in the download; keep as-is for fairness
    print(f"Loaded panel: {df.shape[0]:,} rows x {df.shape[1]} cols\n")

    # ---- write each variant ---------------------------------------------
    csv_path = WORK / "panel.csv"
    df.write_csv(csv_path)

    variants = {}
    variants["CSV"] = csv_path
    for comp in ("snappy", "zstd", "gzip"):
        p = WORK / f"panel_{comp}.parquet"
        df.write_parquet(p, compression=comp)
        variants[f"Parquet-{comp}"] = p

    # partitioned by quarter (one file per REPDTE) = layout used by the downloader
    part_dir = WORK / "panel_part_quarter"
    part_dir.mkdir()
    for key, sub in df.partition_by("REPDTE", as_dict=True).items():
        k = key[0] if isinstance(key, tuple) else key
        sub.write_parquet(part_dir / f"REPDTE={k}.parquet", compression="zstd")
    variants["Parquet-zstd-partitioned"] = part_dir

    rows = []
    for name, path in variants.items():
        size = dir_size_mb(path)
        is_csv = name == "CSV"
        is_part = name.endswith("partitioned")

        if is_csv:
            f_all = lambda: pl.read_csv(path)
            f_5 = lambda: pl.read_csv(path, columns=COLS5)
            f_q = lambda: pl.read_csv(path).filter(pl.col("REPDTE").cast(pl.Utf8) == ONE_QUARTER)
        elif is_part:
            f_all = lambda: pl.read_parquet(path / "*.parquet")
            f_5 = lambda: pl.read_parquet(path / "*.parquet", columns=COLS5)
            f_q = lambda: pl.read_parquet(path / f"REPDTE={ONE_QUARTER}.parquet")
        else:
            f_all = lambda: pl.read_parquet(path)
            f_5 = lambda: pl.read_parquet(path, columns=COLS5)
            f_q = lambda: (pl.scan_parquet(path)
                           .filter(pl.col("REPDTE") == ONE_QUARTER).collect())

        (m_all, s_all) = timeit(f_all)
        (m_5, s_5) = timeit(f_5)
        (m_q, s_q) = timeit(f_q)
        rows.append({
            "format": name, "size_MB": round(size, 1),
            "read_all_s": round(m_all, 3), "read_all_std": round(s_all, 3),
            "read_5cols_s": round(m_5, 3), "read_5cols_std": round(s_5, 3),
            "filter_1q_s": round(m_q, 3), "filter_1q_std": round(s_q, 3),
        })
        print(rows[-1])

    res = pd.DataFrame(rows)
    csv_size = res.loc[res.format == "CSV", "size_MB"].iloc[0]
    res["size_vs_CSV_%"] = (res["size_MB"] / csv_size * 100).round(1)
    res.to_csv(REPORTS / "storage_benchmark.csv", index=False)
    print("\n", res.to_string(index=False))
    print(f"\nSaved to {REPORTS / 'storage_benchmark.csv'}")


if __name__ == "__main__":
    main()