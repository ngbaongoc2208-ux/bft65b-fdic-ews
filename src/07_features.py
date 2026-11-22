"""
Part 4 - Feature pipeline: leak-free feature table + preprocessing fitted on training data only.

Reads  : data_clean/panel.parquet (from 04_build_panel.py)
Writes : data_clean/features.parquet, reports/part4/*.csv, part4_report.txt
Run    : python src/07_features.py          (about 10-20 minutes; pause OneDrive first)

The development / test design is fixed HERE, before any modelling:
    burn-in : 1984 (lags need 4 quarters of history)       -> not used
    dev     : 1985Q1 - 2010Q4  (tuning, feature selection, thresholds are decided on this period only)
    gap     : 2011Q1 - 2011Q4  (4 quarters: the labels of the last dev rows look 4 quarters ahead)
    test    : 2012Q1 - 2025Q2  (used ONCE, in Part 5; nothing in this script fits or tunes on it)
"""
from __future__ import annotations

import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from fdic_features import FEATURE_GROUPS, FEATURE_SPEC, FEATURES, RAW_NEEDED, build_features, make_preprocessor

warnings.filterwarnings("ignore")
PANEL = Path("data_clean/panel.parquet")
OUT = Path("data_clean/features.parquet")
REP = Path("reports/part4")
REP.mkdir(parents=True, exist_ok=True)

DEV_START, DEV_END = date(1985, 3, 31), date(2010, 12, 31)
TEST_START, TEST_END = date(2012, 3, 31), date(2025, 6, 30)
TRAIN_END, VALID_START = date(2004, 12, 31), date(2006, 3, 31)      # validation inside dev (ablation only)
SEED = 42
NEG_FRAC = 1.0                                                        # share of negative training rows kept (1.0 = all)

pd.set_option("display.width", 200)
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


# ============================================================================
section("1. LOAD PANEL AND CHECK THE INPUT COLUMNS")
# ============================================================================
panel = pl.read_parquet(PANEL)
say(f"Panel: {panel.height:,} rows x {panel.width} columns")
forbidden = [c for c in RAW_NEEDED if c.startswith("meta_") or c in ("y", "label_complete")]
say("Feature inputs contain no label or meta_ column:", "PASS" if not forbidden else f"FAIL {forbidden}")

say("\nYear-to-date check: median of (item / gross loans x 100) by quarter of the year, 1990+, banks with loans > 0. "
    "If the item is year-to-date the values rise roughly 1 : 2 : 3 : 4 from March to December.")
chk = (panel.filter((pl.col("date") >= date(1990, 1, 1)) & (pl.col("LNLSGR") > 0))
       .with_columns(pl.col("date").dt.month().alias("m"))
       .group_by("m").agg((pl.col("NTLNLS") / pl.col("LNLSGR") * 100).median().alias("NTLNLS/LNLSGR %"),
                          (pl.col("ELNATR") / pl.col("LNLSGR") * 100).median().alias("ELNATR/LNLSGR %"),
                          pl.col("ROA").median().alias("ROA % (annualised)"))
       .sort("m").to_pandas())
say(chk.to_string(index=False))
r = chk.set_index("m")
say(f"Ratio December / March: net charge-offs {r.loc[12, 'NTLNLS/LNLSGR %'] / max(r.loc[3, 'NTLNLS/LNLSGR %'], 1e-9):.1f}, "
    f"provisions {r.loc[12, 'ELNATR/LNLSGR %'] / max(r.loc[3, 'ELNATR/LNLSGR %'], 1e-9):.1f}, "
    f"ROA {r.loc[12, 'ROA % (annualised)'] / r.loc[3, 'ROA % (annualised)']:.1f} (about 4 for year-to-date items, about 1 for annualised ratios)")

# ============================================================================
section("2. BUILD THE FEATURE TABLE (Polars lazy pipeline)")
# ============================================================================
t0 = time.time()
feats = build_features(panel).collect()
t_build = time.time() - t0
say(f"{len(FEATURES)} features built for {feats.height:,} rows in {t_build:.1f} s")

lab = panel.select(["CERT", "qi", "y", "label_complete"])
tab = feats.join(lab, on=["CERT", "qi"], how="left")
d = pl.col("date")
tab = tab.with_columns(
    pl.when(d < DEV_START).then(pl.lit("burnin"))
    .when(d <= DEV_END).then(pl.lit("dev"))
    .when(d < TEST_START).then(pl.lit("gap"))
    .when(d <= TEST_END).then(pl.lit("test"))
    .otherwise(pl.lit("unlabelled")).alias("split"))

sp = (tab.group_by("split").agg(pl.len().alias("rows"), pl.col("CERT").n_unique().alias("banks"),
                                pl.col("y").sum().alias("positive_rows"),
                                pl.col("CERT").filter(pl.col("y") == 1).n_unique().alias("failed_banks"),
                                pl.col("date").min().alias("first"), pl.col("date").max().alias("last"))
      .with_columns((pl.col("positive_rows") / pl.col("rows") * 100).alias("positive_%")).to_pandas()
      .set_index("split").reindex(["burnin", "dev", "gap", "test", "unlabelled"]))
sp.to_csv(REP / "split_summary.csv")
say(sp.to_string())
n_test_fail = int(sp.loc["test", "failed_banks"])
n_dev_fail = int(sp.loc["dev", "failed_banks"])
say(f"\nFailed banks with at least one positive row: dev {n_dev_fail:,}, test {n_test_fail:,}  "
    f"(exam requires at least 50 in test): {'PASS' if n_test_fail >= 50 else 'FAIL'}")
assert (tab.filter(pl.col("split") == "unlabelled")["label_complete"].sum() == 0), "unlabelled rows must have incomplete labels"

# feature dictionary + dev summary
dev = tab.filter(pl.col("split") == "dev")
rows = []
for name, grp, defn, why in FEATURE_SPEC:
    x = dev[name].drop_nulls().to_numpy()
    rows.append({"feature": name, "group": grp, "definition": defn, "why": why,
                 "missing_%_dev": dev[name].null_count() / dev.height * 100,
                 "p1": np.percentile(x, 1), "median": np.median(x), "p99": np.percentile(x, 99)})
fd = pd.DataFrame(rows)
fd.to_csv(REP / "feature_dictionary.csv", index=False)
say("\nFeatures with more than 1% missing values in the development period:")
say(fd[fd["missing_%_dev"] > 1][["feature", "group", "missing_%_dev"]].to_string(index=False))

# ============================================================================
section("3. SAVE THE FEATURE TABLE (and verify the file)")
# ============================================================================
cols = ["CERT", "date", "qi", "split", "y"] + FEATURES
tab.select(cols).write_parquet(OUT, compression="zstd")
back = pl.read_parquet(OUT)
ok_arrow = True
try:
    pq.ParquetFile(OUT)
except Exception as e:                                                # noqa: BLE001
    ok_arrow = False
say(f"Saved {OUT}: {back.height:,} rows x {back.width} columns, {OUT.stat().st_size / 1024**2:.1f} MB; "
    f"re-read rows equal: {back.height == tab.height}; pyarrow can open it: {ok_arrow}")

# ============================================================================
section("4. LEAKAGE TESTS")
# ============================================================================
say("Test A - truncation invariance. Features of rows dated <= T computed from the full history must equal "
    "the features computed from data truncated at T (a feature that looks ahead would change).")
for cutoff in [date(1995, 12, 31), date(2008, 12, 31), date(2015, 12, 31)]:
    full = build_features(panel).collect().filter(pl.col("date") <= cutoff).sort(["CERT", "qi"])
    trunc = build_features(panel.filter(pl.col("date") <= cutoff)).collect().sort(["CERT", "qi"])
    A, B = full.select(FEATURES).to_numpy(), trunc.select(FEATURES).to_numpy()
    same_keys = full.height == trunc.height and bool((full["CERT"] == trunc["CERT"]).all()
                                                     and (full["qi"] == trunc["qi"]).all())
    null_mis = int((np.isnan(A) != np.isnan(B)).sum())
    both = ~np.isnan(A) & ~np.isnan(B)
    maxdiff = float(np.abs(A[both] - B[both]).max()) if both.any() else 0.0
    say(f"  T = {cutoff}: rows {full.height:,}; keys equal {same_keys}; missing-pattern mismatches {null_mis}; "
        f"max abs difference {maxdiff:.2e} -> {'PASS' if same_keys and null_mis == 0 and maxdiff < 1e-9 else 'FAIL'}")

say("\nTest B - negative control: the same test applied to a deliberately LEAKY feature (next quarter's ROA).")
cut = date(2008, 12, 31)
base = panel.sort(["CERT", "qi"])
leak_full = (base.with_columns(pl.col("ROA").shift(-1).over("CERT").alias("lead"))
             .filter(pl.col("date") <= cut).select(["CERT", "qi", "lead"]))
leak_tr = (base.filter(pl.col("date") <= cut).with_columns(pl.col("ROA").shift(-1).over("CERT").alias("lead"))
           .select(["CERT", "qi", "lead"]))
mism = int((leak_full["lead"].fill_null(-9e9) != leak_tr["lead"].fill_null(-9e9)).sum())
say(f"  lead_roa: {mism:,} rows differ between full and truncated data -> the test detects look-ahead: "
    f"{'PASS' if mism > 0 else 'FAIL'}")

# ============================================================================
section("5. PREPROCESSING FITTED ON TRAINING DATA ONLY")
# ============================================================================
train = tab.filter((pl.col("split") == "dev") & (pl.col("date") <= TRAIN_END))
valid = tab.filter((pl.col("split") == "dev") & (pl.col("date") >= VALID_START))
assert train["date"].max() <= TRAIN_END and valid["date"].min() > TRAIN_END
say(f"Validation scheme inside the development period (used for ablation here, and as one fold in Part 5):")
say(f"  train {train['date'].min()} to {train['date'].max()}: {train.height:,} rows, {int(train['y'].sum()):,} positive")
say(f"  gap   2005Q1-2005Q4 (4 quarters)")
say(f"  valid {valid['date'].min()} to {valid['date'].max()}: {valid.height:,} rows, {int(valid['y'].sum()):,} positive")

Xtr = train.select(FEATURES).to_pandas()
pre = make_preprocessor().fit(Xtr)
say(f"\nPipeline fitted on {len(Xtr):,} training rows (no validation or test row seen):")
say(f"  missing-value flags added for: {pre.named_steps['flags'].cols_}")
say(f"  columns dropped as >0.95 correlated (dropped -> kept): {pre.named_steps['corr'].dropped_}")
say(f"  columns after preprocessing: {len(pre.named_steps['corr'].keep_)}")
lo_tr, hi_tr = pre.named_steps["winsor"].lo_, pre.named_steps["winsor"].hi_

Xdev = dev.select(FEATURES).to_pandas()
pre_dev = make_preprocessor().fit(Xdev)
rel = ((pre_dev.named_steps["winsor"].hi_ - hi_tr).abs() / (hi_tr.abs() + 1e-9)).sort_values(ascending=False)
say("\nWhy fitting on the right window matters: upper clipping bounds fitted on 1985-2004 versus on all of the "
    "development period differ by (relative), top 5:")
say(rel.head(5).to_string())
del Xdev, pre_dev

# ============================================================================
section("6. SINGLE-FEATURE SANITY CHECK (development period)")
# ============================================================================
Xd = dev.select(FEATURES).to_numpy()
yd = dev["y"].to_numpy()
rows = []
for j, name in enumerate(FEATURES):
    m = ~np.isnan(Xd[:, j])
    a = roc_auc_score(yd[m], Xd[m, j]) if 0 < yd[m].sum() < m.sum() else np.nan
    rows.append({"feature": name, "auc": a, "auc_strength": max(a, 1 - a), "direction": "higher=riskier" if a >= .5 else "lower=riskier"})
ua = pd.DataFrame(rows).sort_values("auc_strength", ascending=False)
ua.to_csv(REP / "single_feature_auc.csv", index=False)
say(ua.head(10).to_string(index=False))
n_hi = int((ua.auc_strength >= 0.90).sum())
say(f"Largest single-feature AUC: {ua.auc_strength.max():.3f}; features with AUC >= 0.90: {n_hi}. "
    f"Heuristic alarm: {'REVIEW - see section 9 for the diagnosis' if n_hi else 'none'}")
del Xd, yd


# ============================================================================
def fit_eval(tr, va, feats, shuffle=False, seed=SEED, **prep):
    rng = np.random.default_rng(seed)
    ytr = tr["y"].to_numpy()
    keep = (ytr == 1) | (rng.random(len(ytr)) < NEG_FRAC)
    tr = tr.filter(pl.Series(keep))
    y = tr["y"].to_numpy()
    if shuffle:
        y = rng.permutation(y)
    pipe = Pipeline(make_preprocessor(**prep).steps + [("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=200))])
    pipe.fit(tr.select(feats).to_pandas(), y)
    yv = va["y"].to_numpy()
    p = pipe.predict_proba(va.select(feats).to_pandas())[:, 1]
    top = p >= np.quantile(p, 0.95)
    return {"pr_auc": average_precision_score(yv, p), "roc_auc": roc_auc_score(yv, p),
            "recall_top5pct": yv[top].sum() / yv.sum(), "n_features": len(pipe.named_steps["corr"].keep_)}


section("7. ABLATION BY FEATURE GROUP (train 1985-2004, validate 2006-2010, logistic regression)")
say(f"Baseline rate of positive rows in validation: {valid['y'].mean() * 100:.2f}%  (PR-AUC of a random score = this rate)")
res = [{"setting": "all features", **fit_eval(train, valid, FEATURES)}]
say(res[0])
for g, fs in FEATURE_GROUPS.items():
    rest = [f for f in FEATURES if f not in fs]
    r1 = {"setting": f"drop {g}", **fit_eval(train, valid, rest)}
    r2 = {"setting": f"only {g}", **fit_eval(train, valid, fs)}
    res += [r1, r2]
    say(r1)
    say(r2)
ab = pd.DataFrame(res)
base_pr = ab.loc[0, "pr_auc"]
ab["delta_pr_auc_vs_all"] = ab["pr_auc"] - base_pr
ab.to_csv(REP / "ablation_groups.csv", index=False)
say("\n" + ab.to_string(index=False))

section("8. PREPROCESSING CHOICES AND PLACEBO TEST (same split)")
pp = [{"setting": "winsorise 1% / 99%, corr filter 0.95 (chosen)", **res[0]}]
for name, kw in [("no winsorising, corr filter 0.95", dict(winsor_q=None)),
                 ("winsorise 0.1% / 99.9%, corr filter 0.95", dict(winsor_q=0.001)),
                 ("winsorise 1% / 99%, no corr filter", dict(corr_threshold=None))]:
    pp.append({"setting": name, **fit_eval(train, valid, FEATURES, **kw)})
pp = pd.DataFrame(pp)
pp.to_csv(REP / "preprocessing_choices.csv", index=False)
say(pp.to_string(index=False))

say("\nPlacebo (informational): train on SHUFFLED labels, evaluate on the real validation labels, 5 shuffles.")
plc = [fit_eval(train, valid, FEATURES, shuffle=True, seed=s_)["roc_auc"] for s_ in range(5)]
say("  validation ROC-AUC per shuffle: " + ", ".join(f"{a:.3f}" for a in plc) + f"   mean {np.mean(plc):.3f}, sd {np.std(plc):.3f}")
say("  A model fitted on noise has random weights. When many features are individually strong predictors,")
say("  random weights can give an AUC far from 0.5 in either direction, so this placebo is NOT used as evidence")
say("  about leakage (truncation tests A and B and the diagnosis in section 9 are).")

section("9. DIAGNOSIS: WHY ARE SOME SINGLE-FEATURE AUCs HIGH? (development period; labels used for diagnosis only)")
dd = dev.join(panel.select(["CERT", "qi", "meta_days_to_fail"]), on=["CERT", "qi"], how="left")
signs = {"texas_ratio": 1, "eq_assets": -1, "roa": -1, "npl_ratio": 1, "rk_eq_assets": -1}


def dauc(df, f):
    x, yy = df[f].to_numpy(), df["y"].to_numpy()
    m = ~np.isnan(x)
    if m.sum() == 0 or yy[m].sum() == 0 or yy[m].sum() == m.sum():
        return np.nan
    return roc_auc_score(yy[m], signs[f] * x[m])


say("(a) AUC in three sub-periods (values below 0.5 would mean the wrong direction):")
rows = []
for name, a, b in [("1985-1993", date(1985, 3, 31), date(1993, 12, 31)),
                   ("1994-2007", date(1994, 3, 31), date(2007, 12, 31)),
                   ("2008-2010", date(2008, 3, 31), date(2010, 12, 31))]:
    sub = dd.filter((pl.col("date") >= a) & (pl.col("date") <= b))
    rows.append({"period": name, "rows": sub.height, "positive_rows": int(sub["y"].sum()),
                 **{f: dauc(sub, f) for f in signs}})
dp = pd.DataFrame(rows)
dp.to_csv(REP / "diagnostics_by_period.csv", index=False)
say(dp.to_string(index=False))

say("\n(b) Are positive rows already visibly critical at the report date? (share of rows)")
pos, neg = dd.filter(pl.col("y") == 1), dd.filter(pl.col("y") == 0)
for label, cond in [("eq_assets < 0 (book-insolvent)", pl.col("eq_assets") < 0),
                    ("eq_assets < 2%", pl.col("eq_assets") < 2),
                    ("texas_ratio > 100", pl.col("texas_ratio") > 100)]:
    sp_, sn_ = pos.filter(cond).height / max(pos.height, 1) * 100, neg.filter(cond).height / max(neg.height, 1) * 100
    say(f"  {label:<34} positives {sp_:5.1f}%   negatives {sn_:5.1f}%")

say("\n(c) AUC among rows that are NOT yet critical (eq_assets >= 2%):")
ok = dd.filter(pl.col("eq_assets") >= 2)
say(f"  rows {ok.height:,}; positive rows kept {int(ok['y'].sum()):,} of {int(dd['y'].sum()):,}")
say("  " + ", ".join(f"{f}: {dauc(ok, f):.3f}" for f in signs))

say("\n(d) AUC by time to failure (positive rows in the bucket versus all negative rows):")
rows = []
for lo, hi in [(0, 92), (92, 183), (183, 274), (274, 366)]:
    sub = dd.filter((pl.col("y") == 0) | ((pl.col("meta_days_to_fail") > lo) & (pl.col("meta_days_to_fail") <= hi)))
    rows.append({"days_to_failure": f"{lo + 1}-{hi}", "positive_rows": int(sub["y"].sum()),
                 **{f: dauc(sub, f) for f in signs}})
dh = pd.DataFrame(rows)
dh.to_csv(REP / "diagnostics_horizon.csv", index=False)
say(dh.to_string(index=False))
say("Reading guide: a leak would give near-perfect AUC at every horizon and in every period; "
    "genuine signal fades as the horizon grows and differs by period.")

say(f"\nFeature pipeline build time: {t_build:.1f} s for {tab.height:,} rows.")
say("Nothing in this script fitted, tuned or evaluated on the test period; test rows were only written to the table.")
(REP / "part4_report.txt").write_text("\n".join(LOG), encoding="utf-8")
say("DONE.")