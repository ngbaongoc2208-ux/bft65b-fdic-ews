"""
Part 3 - Step 1: core transformations in Polars (eager / lazy / streaming) vs a pandas baseline.

Reads   : data_clean/panel.parquet  (made by 04_build_panel.py)
Writes  : reports/part3/*.csv, *.txt, *.png  and data_clean/bench/panel_x{k}.parquet
Run     : python src/05_polars_vs_pandas.py            (about 10-20 minutes; close browsers first)
Options : --max-scale 8  --pandas-max-scale 4

What it measures
  A. each core transformation in memory (pandas vs Polars), results checked to be identical
  B. the whole feature pipeline from the Parquet file, on 1x, 2x, 4x, 8x copies of the panel:
     time and PEAK MEMORY of pandas, Polars eager, Polars lazy, Polars lazy+streaming
     (every run is a fresh process, so memory is not polluted by earlier runs)
  C. the Polars query plans (explain), to show projection / predicate pushdown
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import psutil

SRC = Path("data_clean/panel.parquet")
BENCH = Path("data_clean/bench")
REP = Path("reports/part3")
NEEDED = ["CERT", "qi", "ASSET", "EQ", "NCLNLS", "LNLSGR", "CHBAL", "SC", "LNLSNET", "DEP", "ROA"]
FEATS = ["eq_assets", "npl_ratio", "liquid", "loans_dep", "log_asset",
         "eq_lag4", "eq_chg4", "roa_m4", "roa_s4", "max_asset", "eq_rank"]


# ------------------------------------------------------------------ pandas
def pd_ratios(df):
    return df.assign(
        eq_assets=df.EQ / df.ASSET * 100,
        npl_ratio=np.where(df.LNLSGR > 0, df.NCLNLS / df.LNLSGR * 100, np.nan),
        liquid=(df.CHBAL + df.SC) / df.ASSET * 100,
        loans_dep=np.where(df.DEP > 0, df.LNLSNET / df.DEP * 100, np.nan),
        log_asset=np.log(df.ASSET))


def pd_groupby(df):
    return (df.groupby("qi").agg(n=("CERT", "size"), mean_eq=("eq_assets", "mean"),
                                 med_roa=("ROA", "median"), tot_asset=("ASSET", "sum")).reset_index())


def pd_window(df):
    g = df.groupby("CERT")
    lag = g["eq_assets"].shift(4)
    return df.assign(
        eq_lag4=lag, eq_chg4=df["eq_assets"] - lag,
        roa_m4=g["ROA"].rolling(4).mean().reset_index(level=0, drop=True),
        roa_s4=g["ROA"].rolling(4).std().reset_index(level=0, drop=True))


def pd_join(df):
    m = df.groupby("CERT", as_index=False)["ASSET"].max().rename(columns={"ASSET": "max_asset"})
    return df.merge(m, on="CERT", how="left")


def pd_rank(df):
    return df.assign(eq_rank=df.groupby("qi")["eq_assets"].rank(pct=True))


def pd_full(path):
    df = pd.read_parquet(path, columns=NEEDED)
    return pd_rank(pd_join(pd_window(pd_ratios(df))))


# ------------------------------------------------------------------ polars (DataFrame or LazyFrame)
def pl_ratios(x):
    return x.with_columns(
        (pl.col("EQ") / pl.col("ASSET") * 100).alias("eq_assets"),
        pl.when(pl.col("LNLSGR") > 0).then(pl.col("NCLNLS") / pl.col("LNLSGR") * 100).alias("npl_ratio"),
        ((pl.col("CHBAL") + pl.col("SC")) / pl.col("ASSET") * 100).alias("liquid"),
        pl.when(pl.col("DEP") > 0).then(pl.col("LNLSNET") / pl.col("DEP") * 100).alias("loans_dep"),
        pl.col("ASSET").log().alias("log_asset"))


def pl_groupby(x):
    return (x.group_by("qi").agg(pl.len().alias("n"), pl.col("eq_assets").mean().alias("mean_eq"),
                                 pl.col("ROA").median().alias("med_roa"),
                                 pl.col("ASSET").sum().alias("tot_asset")).sort("qi"))


def pl_window(x):
    return x.with_columns(
        pl.col("eq_assets").shift(4).over("CERT").alias("eq_lag4"),
        pl.col("ROA").rolling_mean(window_size=4, min_samples=4).over("CERT").alias("roa_m4"),
        pl.col("ROA").rolling_std(window_size=4, min_samples=4).over("CERT").alias("roa_s4"),
    ).with_columns((pl.col("eq_assets") - pl.col("eq_lag4")).alias("eq_chg4"))


def pl_join(x):
    m = x.group_by("CERT").agg(pl.col("ASSET").max().alias("max_asset"))
    return x.join(m, on="CERT", how="left")


def pl_rank(x):
    return x.with_columns((pl.col("eq_assets").rank("average").over("qi")
                           / pl.col("eq_assets").count().over("qi")).alias("eq_rank"))


def pl_chain(x):
    return pl_rank(pl_join(pl_window(pl_ratios(x))))


def pl_lazy(path):
    return pl_chain(pl.scan_parquet(path).select(NEEDED))


def collect_streaming(lf):
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


VARIANTS = {
    "pandas": lambda p: pd_full(p),
    "polars_eager": lambda p: pl_chain(pl.read_parquet(p, columns=NEEDED)),
    "polars_lazy": lambda p: pl_lazy(p).collect(),
    "polars_streaming": lambda p: collect_streaming(pl_lazy(p)),
}


def checksum(res) -> dict:
    if isinstance(res, pd.DataFrame):
        return {c: float(res[c].sum()) for c in FEATS} | {"rows": int(len(res))}
    s = res.select([pl.col(c).sum().alias(c) for c in FEATS]).row(0, named=True)
    return {c: float(s[c]) for c in FEATS} | {"rows": int(res.height)}


# ------------------------------------------------------------------ memory + timing helpers
class PeakMem:
    def __init__(self):
        self.p = psutil.Process()
        self.base = self.p.memory_info().rss
        self.peak = self.base
        self._stop = False

    def _loop(self):
        while not self._stop:
            self.peak = max(self.peak, self.p.memory_info().rss)
            time.sleep(0.02)

    def __enter__(self):
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self.t.join()
        mi = self.p.memory_info()
        self.peak = max(self.peak, getattr(mi, "peak_wset", 0), mi.rss)

    @property
    def peak_mb(self):
        return (self.peak - self.base) / 1024**2


def timeit(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return st.mean(ts), (st.stdev(ts) if n > 1 else 0.0)


def worker(variant: str, scale: int, repeats: int) -> None:
    path = BENCH / f"panel_x{scale}.parquet"
    fn = VARIANTS[variant]
    with PeakMem() as pm:
        fn(path)                       # warm-up (not timed)
        ts, res = [], None
        for _ in range(repeats):
            res = None
            t0 = time.perf_counter()
            res = fn(path)
            ts.append(time.perf_counter() - t0)
        cs = checksum(res)
    print("RESULT:" + json.dumps({"variant": variant, "scale": scale, "times": ts,
                                  "peak_mb": pm.peak_mb, "check": cs}))


# ------------------------------------------------------------------ comparison helper
def as_np(x, cols):
    if isinstance(x, pd.DataFrame):
        return x[cols].to_numpy(dtype=float)
    return x.select(cols).cast(pl.Float64).to_numpy()


def same(a, b, cols, keys) -> bool:
    if isinstance(a, pd.DataFrame):
        a = a.sort_values(keys, kind="stable")
    else:
        a = a.sort(keys, maintain_order=True)
    if isinstance(b, pd.DataFrame):
        b = b.sort_values(keys, kind="stable")
    else:
        b = b.sort(keys, maintain_order=True)
    return bool(np.allclose(as_np(a, cols), as_np(b, cols), rtol=1e-6, atol=1e-6, equal_nan=True))


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=3, metavar=("VARIANT", "SCALE", "REPEATS"))
    ap.add_argument("--max-scale", type=int, default=8)
    ap.add_argument("--pandas-max-scale", type=int, default=4)
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), int(args.worker[2]))
        return

    REP.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    vm = psutil.virtual_memory()
    say(f"Machine: {psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads, "
        f"RAM {vm.total / 1024**3:.1f} GB (available now {vm.available / 1024**3:.1f} GB); "
        f"polars {pl.__version__}, pandas {pd.__version__}, numpy {np.__version__}")

    # ---- replicated files -------------------------------------------------
    scales = [s for s in (1, 2, 4, 8, 16) if s <= args.max_scale]
    base = pl.read_parquet(SRC, columns=NEEDED)
    for k in scales:
        p = BENCH / f"panel_x{k}.parquet"
        if not p.exists():
            pl.concat([base.with_columns(pl.col("CERT") + i * 1_000_000) for i in range(k)]).write_parquet(p)
    sizes = {k: (BENCH / f"panel_x{k}.parquet").stat().st_size / 1024**2 for k in scales}
    say(f"Rows in 1x panel: {base.height:,}; replicated files (MB on disk): "
        + ", ".join(f"x{k}: {v:.0f}" for k, v in sizes.items()))
    del base

    # ---- A. in-memory tasks at 1x ----------------------------------------
    say("\n=== A. Core transformations, in memory, 1x (mean +- std of 5 runs, after 1 warm-up) ===")
    P1 = BENCH / "panel_x1.parquet"
    dpd = pd.read_parquet(P1, columns=NEEDED)
    dpl = pl.read_parquet(P1, columns=NEEDED)
    dpd_r, dpl_r = pd_ratios(dpd), pl_ratios(dpl)
    tasks = [
        ("read 11 columns", lambda: pd.read_parquet(P1, columns=NEEDED), lambda: pl.read_parquet(P1, columns=NEEDED),
         None, None),
        ("ratios (5 columns)", lambda: pd_ratios(dpd), lambda: pl_ratios(dpl),
         ["eq_assets", "npl_ratio", "liquid", "loans_dep", "log_asset"], ["CERT", "qi"]),
        ("group by quarter (4 aggregates)", lambda: pd_groupby(dpd_r), lambda: pl_groupby(dpl_r),
         ["n", "mean_eq", "med_roa", "tot_asset"], ["qi"]),
        ("window: lag4, rolling mean/std per bank", lambda: pd_window(dpd_r), lambda: pl_window(dpl_r),
         ["eq_lag4", "eq_chg4", "roa_m4", "roa_s4"], ["CERT", "qi"]),
        ("join bank-level max asset", lambda: pd_join(dpd_r), lambda: pl_join(dpl_r),
         ["max_asset"], ["CERT", "qi"]),
        ("rank within quarter", lambda: pd_rank(dpd_r), lambda: pl_rank(dpl_r), ["eq_rank"], ["CERT", "qi"]),
    ]
    rows = []
    for name, fpd, fpl, cols, keys in tasks:
        mp, sp = timeit(fpd)
        ml, sl = timeit(fpl)
        ok = "-" if cols is None else str(same(fpd(), fpl(), cols, keys))
        rows.append({"task": name, "pandas_s": mp, "pandas_std": sp, "polars_s": ml, "polars_std": sl,
                     "speedup": mp / ml, "results_identical": ok})
        say(f"{name:<42} pandas {mp:7.3f}+-{sp:5.3f}s   polars {ml:7.3f}+-{sl:5.3f}s   x{mp / ml:5.1f}   identical: {ok}")
    pd.DataFrame(rows).to_csv(REP / "benchmark_tasks.csv", index=False)
    del dpd, dpl, dpd_r, dpl_r

    # ---- C. query plans ----------------------------------------------------
    say("\n=== C. Polars query plans ===")
    lf = pl_lazy(P1)
    (REP / "polars_plan_pipeline_optimized.txt").write_text(lf.explain(), encoding="utf-8")
    (REP / "polars_plan_pipeline_unoptimized.txt").write_text(lf.explain(optimized=False), encoding="utf-8")
    lf2 = (pl.scan_parquet(P1).filter(pl.col("qi") >= 2000 * 4).select(["CERT", "qi", "ASSET", "EQ"])
           .group_by("qi").agg(pl.col("EQ").sum()))
    (REP / "polars_plan_filter_optimized.txt").write_text(lf2.explain(), encoding="utf-8")
    (REP / "polars_plan_filter_unoptimized.txt").write_text(lf2.explain(optimized=False), encoding="utf-8")
    say("--- Query 2 (filter + select + group by), UNOPTIMISED plan ---")
    say(lf2.explain(optimized=False))
    say("--- Query 2, OPTIMISED plan ---")
    say(lf2.explain())
    say("--- Feature pipeline, OPTIMISED plan (first 45 lines) ---")
    say("\n".join(lf.explain().splitlines()[:45]))

    # ---- check file for the PySpark comparison -------------------------------
    chk = (lf.group_by("qi").agg(
        pl.len().alias("n"), pl.col("eq_assets").mean().alias("mean_eq_assets"),
        pl.col("npl_ratio").mean().alias("mean_npl_ratio"), pl.col("liquid").mean().alias("mean_liquid"),
        pl.col("loans_dep").mean().alias("mean_loans_dep"), pl.col("log_asset").mean().alias("mean_log_asset"),
        pl.col("eq_lag4").mean().alias("mean_eq_lag4"), pl.col("eq_chg4").mean().alias("mean_eq_chg4"),
        pl.col("roa_m4").mean().alias("mean_roa_m4"), pl.col("roa_s4").mean().alias("mean_roa_s4"),
        pl.col("max_asset").sum().alias("sum_max_asset"), pl.col("eq_rank").mean().alias("mean_eq_rank"))
        .sort("qi").collect())
    chk.write_csv(REP / "polars_check.csv")
    say(f"\nSaved {REP / 'polars_check.csv'} ({chk.height} quarters) for the PySpark comparison.")

    # ---- B. scaling ----------------------------------------------------------
    say("\n=== B. Whole pipeline from the Parquet file, scaling study (each run = fresh process) ===")
    res = []
    for k in scales:
        reps = {1: 5, 2: 3, 4: 3}.get(k, 2)
        for v in VARIANTS:
            if v == "pandas" and k > args.pandas_max_scale:
                continue
            t0 = time.time()
            r = subprocess.run([sys.executable, __file__, "--worker", v, str(k), str(reps)],
                               capture_output=True, text=True)
            line = [x for x in r.stdout.splitlines() if x.startswith("RESULT:")]
            if r.returncode != 0 or not line:
                say(f"x{k:<2} {v:<17} FAILED ({(r.stderr.strip().splitlines() or ['?'])[-1][:90]})")
                res.append({"scale": k, "variant": v, "failed": True})
                continue
            d = json.loads(line[0][7:])
            m, s = st.mean(d["times"]), (st.stdev(d["times"]) if len(d["times"]) > 1 else 0.0)
            res.append({"scale": k, "variant": v, "rows": d["check"]["rows"], "time_s": m, "time_std": s,
                        "peak_mb": d["peak_mb"], "check": d["check"], "failed": False})
            say(f"x{k:<2} {v:<17} rows {d['check']['rows']:>11,}  time {m:8.2f}+-{s:5.2f}s  "
                f"peak memory {d['peak_mb']:8.0f} MB   (wall {time.time() - t0:.0f}s)")
    ok_res = [r for r in res if not r["failed"]]
    pd.DataFrame([{k: v for k, v in r.items() if k != "check"} for r in res]).to_csv(
        REP / "benchmark_scaling.csv", index=False)

    ref = {r["variant"]: r["check"] for r in ok_res if r["scale"] == 1}
    say("\nResult check at 1x (sum of every feature, relative difference vs pandas):")
    for v, c in ref.items():
        if v == "pandas" or "pandas" not in ref:
            continue
        worst = max(abs(c[f] - ref["pandas"][f]) / (abs(ref["pandas"][f]) + 1e-12) for f in FEATS)
        say(f"  {v:<17} rows equal: {c['rows'] == ref['pandas']['rows']}   worst relative difference: {worst:.2e}")

    # ---- plot + extrapolation -------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    say("\nExtrapolation of peak memory (linear fit) to 70% of this machine's RAM:")
    for v in VARIANTS:
        pts = sorted([(r["rows"] / 1e6, r["time_s"], r["peak_mb"]) for r in ok_res if r["variant"] == v])
        if not pts:
            continue
        x = [p[0] for p in pts]
        ax[0].plot(x, [p[1] for p in pts], marker="o", label=v)
        ax[1].plot(x, [p[2] / 1024 for p in pts], marker="o", label=v)
        if len(pts) >= 2:
            slope, icpt = np.polyfit(x, [p[2] for p in pts], 1)
            cap = (0.7 * vm.total / 1024**2 - icpt) / slope
            say(f"  {v:<17} about {slope:,.0f} MB per million rows -> roughly {cap:,.0f} million rows "
                f"({cap / (pts[0][0] / scales[0]):,.0f}x the US panel) before RAM is exhausted")
    ax[0].set_xlabel("rows (millions)")
    ax[0].set_ylabel("seconds")
    ax[0].set_title("Run time")
    ax[1].set_xlabel("rows (millions)")
    ax[1].set_ylabel("peak memory (GB)")
    ax[1].set_title("Peak memory above baseline")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(REP / "scaling.png", dpi=200)
    (REP / "part3_polars_report.txt").write_text("\n".join(log), encoding="utf-8")
    say("\nDONE. Files in reports/part3/")


if __name__ == "__main__":
    main()