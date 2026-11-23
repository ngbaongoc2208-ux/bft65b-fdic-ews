"""
Part 5 - step 1: model selection, cost-based alarm threshold and frozen configuration.
USES THE DEVELOPMENT PERIOD ONLY: test rows are never even loaded (the file is read with a filter on split == 'dev').

Reads  : data_clean/features.parquet, data_raw/fdic/failures.parquet
Writes : reports/part5/*.csv, reports/part5/frozen_config.json, reports/part5/part5_dev_report.txt,
         data_clean/mllib_sample.parquet (small file for the Spark MLlib demo)
Run    : python src/08_model_selection.py        (about 15-30 minutes; pause OneDrive first)

Validation: rolling origin with an expanding training window and a gap of 4 quarters (labels look 4 quarters ahead):
    F1  train 1985Q1-1989Q4   gap 1990   validate 1991Q1-1993Q4   (tail of the savings-and-loan crisis)
    F2  train 1985Q1-1998Q4   gap 1999   validate 2000Q1-2005Q4   (calm years, few failures)
    F3  train 1985Q1-2004Q4   gap 2005   validate 2006Q1-2010Q4   (financial crisis)
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from fdic_features import FEATURES, make_preprocessor

warnings.filterwarnings("ignore")
FEAT_PATH = Path("data_clean/features.parquet")
FAIL_PATH = Path("data_raw/fdic/failures.parquet")
REP = Path("reports/part5")
REP.mkdir(parents=True, exist_ok=True)
SEED = 42
DEV_END = date(2010, 12, 31)
FOLDS = [("F1", date(1985, 3, 31), date(1989, 12, 31), date(1991, 3, 31), date(1993, 12, 31)),
         ("F2", date(1985, 3, 31), date(1998, 12, 31), date(2000, 3, 31), date(2005, 12, 31)),
         ("F3", date(1985, 3, 31), date(2004, 12, 31), date(2006, 3, 31), date(2010, 12, 31))]
CONFIGS = {
    "LR C=0.01": dict(kind="lr", C=0.01), "LR C=0.1": dict(kind="lr", C=0.1), "LR C=1": dict(kind="lr", C=1.0),
    "HGB depth3 w1": dict(kind="hgb", depth=3, lr=0.1, iters=200, w_pos=1.0),
    "HGB depth3 w10": dict(kind="hgb", depth=3, lr=0.1, iters=200, w_pos=10.0),
    "HGB depth5 w1": dict(kind="hgb", depth=5, lr=0.05, iters=300, w_pos=1.0),
    "HGB depth5 w10": dict(kind="hgb", depth=5, lr=0.05, iters=300, w_pos=10.0),
}
RHO_BASE = 0.01                        # cost of one alarm = 1% of the median cost of a failure (an ASSUMPTION)
RHO_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]

pd.set_option("display.width", 220)
pd.options.display.float_format = "{:,.3f}".format
LOG: list[str] = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def section(t):
    say("\n" + "=" * 78)
    say(t)
    say("=" * 78)


def parse_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    s = s.astype("string")
    d1 = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    d2 = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return d1.combine_first(d2).combine_first(pd.to_datetime(s, errors="coerce"))


def qidx(d: date) -> int:
    return d.year * 4 + (d.month - 1) // 3


# ============================================================================
section("1. LOAD THE DEVELOPMENT PERIOD ONLY")
# ============================================================================
dev = pl.scan_parquet(FEAT_PATH).filter(pl.col("split") == "dev").collect()
assert dev["split"].unique().to_list() == ["dev"] and dev["date"].max() <= DEV_END
say(f"Development rows: {dev.height:,}, positive rows {int(dev['y'].sum()):,}, banks {dev['CERT'].n_unique():,}")
for name, tr0, tr1, va0, va1 in FOLDS:
    assert qidx(va0) - qidx(tr1) >= 5, "gap of at least 4 quarters required"
say("All folds have a gap of at least 4 quarters between the last training row and the first validation row.")

# ============================================================================
section("2. COST OF A FAILURE, ESTIMATED FROM THE FAILURES TABLE")
# ============================================================================
f = pd.read_parquet(FAIL_PATH)
f["CERT"] = pd.to_numeric(f["CERT"], errors="coerce")
f["COST"] = pd.to_numeric(f["COST"], errors="coerce")
f["fail_date"] = parse_date(f["FAILDATE"])
f = f.dropna(subset=["CERT", "fail_date"]).sort_values("fail_date").drop_duplicates("CERT")
f["CERT"] = f["CERT"].astype("int64")
dev_fail = set(dev.filter(pl.col("y") == 1)["CERT"].unique().to_list())
cdev = f[f.CERT.isin(dev_fail)]
MEDIAN_COST = float(cdev["COST"].median())
say(f"Banks failing in the development period with a positive row: {len(dev_fail):,}; COST known for {cdev['COST'].notna().sum():,}")
say(f"COST of a failure (thousand USD): median {MEDIAN_COST:,.0f}, mean {cdev['COST'].mean():,.0f}, "
    f"90th percentile {cdev['COST'].quantile(0.9):,.0f}, max {cdev['COST'].max():,.0f}")
cost_map = {int(c): (float(v) if pd.notna(v) else MEDIAN_COST) for c, v in zip(f.CERT, f.COST)}
C_FA = RHO_BASE * MEDIAN_COST
say(f"Missed failure: its own COST (median {MEDIAN_COST:,.0f} if unknown). "
    f"Alarm: {RHO_BASE:.1%} of the median COST = {C_FA:,.1f} thousand USD per alarmed bank-quarter (assumption; sensitivity in section 6).")


# ============================================================================
def bank_detect(cert, y, s, thr):
    m = y == 1
    if m.sum() == 0:
        return np.nan
    mx = pd.DataFrame({"c": cert[m], "s": s[m]}).groupby("c")["s"].max()
    return float((mx >= thr).mean())


def metrics(cert, y, s):
    thr = np.quantile(s, 0.95)
    return {"positives": int(y.sum()), "pos_rate_%": y.mean() * 100,
            "pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
            "row_recall_top5%": y[s >= thr].sum() / y.sum(), "bank_detect_top5%": bank_detect(cert, y, s, thr)}


def fit_predict(cfg, Xtr, ytr, Xva):
    if cfg["kind"] == "lr":
        pipe = Pipeline(make_preprocessor().steps + [("clf", LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=300))])
        pipe.fit(Xtr, ytr)
        return pipe.predict_proba(Xva)[:, 1]
    m = HistGradientBoostingClassifier(max_depth=cfg["depth"], learning_rate=cfg["lr"], max_iter=cfg["iters"],
                                       l2_regularization=1.0, early_stopping=False, random_state=SEED)
    m.fit(Xtr, ytr, sample_weight=np.where(ytr == 1, cfg["w_pos"], 1.0))
    return m.predict_proba(Xva)[:, 1]


# ============================================================================
section("3. ROLLING-ORIGIN VALIDATION (expanding window, 4-quarter gap)")
# ============================================================================
rows, oof = [], {k: [] for k in CONFIGS}
t_start = time.time()
for fname, tr0, tr1, va0, va1 in FOLDS:
    tr = dev.filter((pl.col("date") >= tr0) & (pl.col("date") <= tr1))
    va = dev.filter((pl.col("date") >= va0) & (pl.col("date") <= va1))
    Xtr, ytr = tr.select(FEATURES).to_pandas(), tr["y"].to_numpy()
    Xva, yva, cva = va.select(FEATURES).to_pandas(), va["y"].to_numpy(), va["CERT"].to_numpy()
    say(f"\n{fname}: train {tr0}..{tr1} ({tr.height:,} rows, {int(ytr.sum()):,} positive) | "
        f"validate {va0}..{va1} ({va.height:,} rows, {int(yva.sum()):,} positive, {yva.mean() * 100:.2f}%)")
    for name, ind, sign in [("single indicator: texas_ratio", "texas_ratio", 1), ("single indicator: equity ratio", "eq_assets", -1)]:
        med = np.nanmedian(tr[ind].to_numpy())
        xv = va[ind].to_numpy()
        s = sign * np.where(np.isnan(xv), med, xv)
        rows.append({"fold": fname, "model": name, **metrics(cva, yva, s)})
    rows.append({"fold": fname, "model": "random score", "positives": int(yva.sum()), "pos_rate_%": yva.mean() * 100,
                 "pr_auc": yva.mean(), "roc_auc": 0.5, "row_recall_top5%": 0.05, "bank_detect_top5%": np.nan})
    for name, cfg in CONFIGS.items():
        t0 = time.time()
        s = fit_predict(cfg, Xtr, ytr, Xva)
        m = metrics(cva, yva, s)
        rows.append({"fold": fname, "model": name, **m})
        oof[name].append(pd.DataFrame({"cert": cva, "y": yva, "s": s, "fold": fname}))
        say(f"  {name:<16} PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}  "
            f"recall(top 5% rows) {m['row_recall_top5%']:.3f}  banks detected {m['bank_detect_top5%']:.3f}   ({time.time() - t0:.0f}s)")
cv = pd.DataFrame(rows)
cv.to_csv(REP / "validation_by_fold.csv", index=False)
summ = (cv.groupby("model")[["pr_auc", "roc_auc", "row_recall_top5%", "bank_detect_top5%"]].mean()
        .sort_values("pr_auc", ascending=False))
summ.to_csv(REP / "validation_summary.csv")
say(f"\nMean over the three folds ({time.time() - t_start:.0f} s in total):")
say(summ.to_string())

models_only = [m for m in summ.index if m in CONFIGS]
best_all = models_only[0]
best_lr = [m for m in models_only if CONFIGS[m]["kind"] == "lr"][0]
say(f"\nSelected model (highest mean PR-AUC): {best_all}.  Baseline (best logistic regression): {best_lr}.")

# ============================================================================
section("4. COST-BASED ALARM THRESHOLD (from out-of-fold predictions)")
# ============================================================================


def cost_curve(o: pd.DataFrame, c_fa: float):
    """Total cost for a grid of thresholds: c_fa x (alarmed rows) + COST of every failed bank that never raised an alarm."""
    s_sorted = np.sort(o["s"].to_numpy())
    pos = o[o.y == 1].groupby("cert")["s"].max()
    order = np.argsort(pos.to_numpy())
    bs, bc = pos.to_numpy()[order], np.array([cost_map.get(int(c), MEDIAN_COST) for c in pos.index])[order]
    cum = np.concatenate([[0.0], np.cumsum(bc)])
    ths = np.unique(np.concatenate([np.quantile(s_sorted, np.linspace(0.5, 0.9999, 500)), [s_sorted.max() * 1.0001]]))
    A = len(s_sorted) - np.searchsorted(s_sorted, ths, side="left")
    M = cum[np.searchsorted(bs, ths, side="left")]
    cost = c_fa * A + M
    det = 1 - np.searchsorted(bs, ths, side="left") / len(bs)
    return ths, A, M, cost, det, len(s_sorted), cum[-1]


frozen = {}
sens_rows = []
for label, name in [("final", best_all), ("baseline", best_lr)]:
    o = pd.concat(oof[name], ignore_index=True)
    for rho in RHO_GRID:
        ths, A, M, cost, det, n, total_miss = cost_curve(o, rho * MEDIAN_COST)
        k = int(np.argmin(cost))
        npos_rows = int(o.y.sum())
        sens_rows.append({"model": label + ": " + name, "rho": rho, "threshold": ths[k], "alarm_rate_%": A[k] / n * 100,
                          "banks_detected_%": det[k] * 100, "cost_saving_vs_no_alarm_%": (1 - cost[k] / total_miss) * 100,
                          "p_star_theory": rho / (1 + rho)})
        if abs(rho - RHO_BASE) < 1e-12:
            frozen[label] = {"model": name, "params": CONFIGS[name], "threshold": float(ths[k]),
                             "oof_alarm_rate_%": float(A[k] / n * 100), "oof_banks_detected_%": float(det[k] * 100)}
sens = pd.DataFrame(sens_rows)
sens.to_csv(REP / "threshold_sensitivity.csv", index=False)
say(sens.to_string(index=False))
say(f"\nChosen threshold for rho = {RHO_BASE}: final model {frozen['final']['threshold']:.4f}, baseline {frozen['baseline']['threshold']:.4f}")
say("p_star_theory = c_alarm / (c_alarm + c_miss) is the threshold for perfectly calibrated probabilities and one failure cost; "
    "the empirical threshold uses the out-of-fold scores and the actual COST of each failed bank.")

# ============================================================================
section("5. FREEZE THE CONFIGURATION (commit this file BEFORE running 09_final_test.py)")
# ============================================================================
cfg_out = {"created": datetime.now().isoformat(timespec="seconds"), "features": FEATURES, "seed": SEED,
           "folds": [[n, str(a), str(b), str(c), str(d)] for n, a, b, c, d in FOLDS],
           "rho_base": RHO_BASE, "median_cost_kUSD": MEDIAN_COST, "c_alarm_kUSD": C_FA,
           "baseline": frozen["baseline"], "final": frozen["final"]}
cfg_out["sha256"] = hashlib.sha256(json.dumps(cfg_out, sort_keys=True).encode()).hexdigest()
(REP / "frozen_config.json").write_text(json.dumps(cfg_out, indent=1), encoding="utf-8")
say(json.dumps({k: v for k, v in cfg_out.items() if k != "features"}, indent=1))

# ============================================================================
section("6. SMALL SAMPLE FOR THE SPARK MLlib DEMO (fold F3: train 1985-2004, validate 2006-2010)")
# ============================================================================
rng = np.random.default_rng(SEED)
tr = dev.filter(pl.col("date") <= FOLDS[2][2])
va = dev.filter((pl.col("date") >= FOLDS[2][3]) & (pl.col("date") <= FOLDS[2][4]))
ytr = tr["y"].to_numpy()
keep = (ytr == 1) | (rng.random(len(ytr)) < 0.25)
tr = tr.filter(pl.Series(keep))
cols = ["y"] + FEATURES
part = pa.table({"part": pa.array(["train"] * tr.height + ["valid"] * va.height),
                 "sample_weight": pa.array(np.concatenate([np.where(tr["y"].to_numpy() == 1, 1.0, 4.0), np.ones(va.height)]).astype("float32")),
                 **{c: pl.concat([tr.select(c), va.select(c)]).to_arrow().column(0).cast(pa.float32() if c != "y" else pa.int8())
                    for c in cols}})
out = Path("data_clean/mllib_sample.parquet")
pq.write_table(part, out, compression="snappy")
h = hashlib.md5()
with open(out, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
say(f"{out}: {part.num_rows:,} rows ({tr.height:,} train: all positives + 25% of negatives with weight 4; {va.height:,} validation)")
say(f"size {out.stat().st_size / 1e6:.1f} MB   md5 {h.hexdigest()}")

say("\nNothing in this script touched the test period. Next: commit reports/part5/frozen_config.json, then run 09_final_test.py ONCE.")
(REP / "part5_dev_report.txt").write_text("\n".join(LOG), encoding="utf-8")
say("DONE.")