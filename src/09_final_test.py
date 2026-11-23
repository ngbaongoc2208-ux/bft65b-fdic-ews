"""
Part 5 - step 2: evaluate on the TEST period, ONCE.

Run only after reports/part5/frozen_config.json (made by 08_model_selection.py) has been committed to Git.
Nothing is tuned here: the model, its hyper-parameters, the features and the alarm threshold come from the frozen file.
The script refuses to run again once reports/part5/test_results.json exists (use --force only after a crash).

Reads  : data_clean/features.parquet, data_raw/fdic/failures.parquet, reports/part5/frozen_config.json
Writes : reports/part5/test_*.csv, test_results.json, part5_test_report.txt
Run    : python src/09_final_test.py          (about 5-10 minutes)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from fdic_features import FEATURES, make_preprocessor

warnings.filterwarnings("ignore")
FEAT_PATH = Path("data_clean/features.parquet")
FAIL_PATH = Path("data_raw/fdic/failures.parquet")
REP = Path("reports/part5")
RESULT = REP / "test_results.json"
SEED = 42
TEST_START, TEST_END = date(2012, 3, 31), date(2025, 6, 30)
N_BOOT = 200

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true", help="only after a crash: the test period may be evaluated once")
args = ap.parse_args()
if RESULT.exists() and not args.force:
    print(f"{RESULT} exists: the test period has already been evaluated. The test set is used ONCE; stopping.")
    sys.exit(1)

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


def fit_predict(cfg, Xtr, ytr, Xte):
    if cfg["kind"] == "lr":
        pipe = Pipeline(make_preprocessor().steps + [("clf", LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=300))])
        pipe.fit(Xtr, ytr)
        return pipe.predict_proba(Xte)[:, 1]
    m = HistGradientBoostingClassifier(max_depth=cfg["depth"], learning_rate=cfg["lr"], max_iter=cfg["iters"],
                                       l2_regularization=1.0, early_stopping=False, random_state=SEED)
    m.fit(Xtr, ytr, sample_weight=np.where(ytr == 1, cfg["w_pos"], 1.0))
    return m.predict_proba(Xte)[:, 1]


# ============================================================================
section("1. FROZEN CONFIGURATION")
# ============================================================================
cfg = json.loads((REP / "frozen_config.json").read_text(encoding="utf-8"))
chk = dict(cfg)
sha = chk.pop("sha256")
assert hashlib.sha256(json.dumps(chk, sort_keys=True).encode()).hexdigest() == sha, "frozen_config.json was modified"
assert cfg["features"] == FEATURES, "feature list differs from the frozen one"
MEDIAN_COST, C_FA, RHO = cfg["median_cost_kUSD"], cfg["c_alarm_kUSD"], cfg["rho_base"]
say(f"Frozen on {cfg['created']} (sha256 {sha[:16]}...)")
for k in ("baseline", "final"):
    say(f"  {k}: {cfg[k]['model']}, threshold {cfg[k]['threshold']:.4f}")
say(f"  cost of one alarm {C_FA:,.1f} thousand USD (rho = {RHO}); median cost of a failure {MEDIAN_COST:,.0f} thousand USD")

dev = pl.scan_parquet(FEAT_PATH).filter(pl.col("split") == "dev").collect()
test = pl.scan_parquet(FEAT_PATH).filter(pl.col("split") == "test").collect()
assert dev["date"].max() < TEST_START and test["date"].min() >= TEST_START and test["date"].max() <= TEST_END
say(f"Fit on development rows: {dev.height:,} (1985Q1-2010Q4). Test rows: {test.height:,}, positive {int(test['y'].sum()):,}, "
    f"banks {test['CERT'].n_unique():,}")

f = pd.read_parquet(FAIL_PATH)
f["CERT"] = pd.to_numeric(f["CERT"], errors="coerce")
f["COST"] = pd.to_numeric(f["COST"], errors="coerce")
f["fail_date"] = parse_date(f["FAILDATE"])
f = f.dropna(subset=["CERT", "fail_date"]).sort_values("fail_date").drop_duplicates("CERT")
f["CERT"] = f["CERT"].astype("int64")
cost_map = {int(c): (float(v) if pd.notna(v) else MEDIAN_COST) for c, v in zip(f.CERT, f.COST)}
fyear = {int(c): int(d.year) for c, d in zip(f.CERT, f.fail_date)}

# ============================================================================
section("2. FIT ON THE DEVELOPMENT PERIOD AND SCORE THE TEST PERIOD")
# ============================================================================
Xdev, ydev = dev.select(FEATURES).to_pandas(), dev["y"].to_numpy()
Xte = test.select(FEATURES).to_pandas()
y, cert = test["y"].to_numpy(), test["CERT"].to_numpy()
scores, cache = {}, {}
for label in ("baseline", "final"):
    name = cfg[label]["model"]
    if name not in cache:
        t0 = time.time()
        cache[name] = fit_predict(cfg[label]["params"], Xdev, ydev, Xte)
        say(f"  fitted {name} on {len(Xdev):,} rows in {time.time() - t0:.0f} s")
    scores[label] = cache[name]
eq, tx = test["eq_assets"].to_numpy(), test["texas_ratio"].to_numpy()
rules = {"rule: equity < 2% of assets": (np.nan_to_num(eq, nan=1e9) < 2),
         "rule: Texas ratio > 100": (np.nan_to_num(tx, nan=-1e9) > 100)}


def evaluate(alarm, y, cert, s=None, mask=None):
    """Row and bank-level results for a boolean alarm vector (optionally on a subset of rows)."""
    if mask is not None:
        y, cert, alarm = y[mask], cert[mask], alarm[mask]
        s = None if s is None else s[mask]
    d = pd.DataFrame({"c": cert, "y": y, "a": alarm})
    fb = d[d.y == 1].groupby("c")["a"].max()
    missed = fb[~fb]
    cost_missed = float(sum(cost_map.get(int(c), MEDIAN_COST) for c in missed.index))
    cost_all = float(sum(cost_map.get(int(c), MEDIAN_COST) for c in fb.index))
    n_alarm = int(alarm.sum())
    total = C_FA * n_alarm + cost_missed
    out = {"rows": len(y), "failed_banks": len(fb), "alarm_rows": n_alarm, "alarm_rate_%": n_alarm / len(y) * 100,
           "precision_rows_%": (y[alarm].mean() * 100) if n_alarm else np.nan,
           "row_recall_%": y[alarm].sum() / max(y.sum(), 1) * 100,
           "banks_detected": int(fb.sum()), "banks_detected_%": fb.mean() * 100 if len(fb) else np.nan,
           "cost_kUSD": total, "cost_no_alarm_kUSD": cost_all,
           "saving_vs_no_alarm_%": (1 - total / cost_all) * 100 if cost_all else np.nan}
    if s is not None and y.sum() > 0:
        out["pr_auc"], out["roc_auc"] = average_precision_score(y, s), roc_auc_score(y, s)
    return out


# ============================================================================
section("3. RESULTS ON THE TEST PERIOD (single evaluation)")
# ============================================================================
rows = []
say(f"Base rate of positive rows: {y.mean() * 100:.3f}%  (random-score PR-AUC); failing banks: {len(set(cert[y == 1])):,}")
for label in ("baseline", "final"):
    s = scores[label]
    thr = cfg[label]["threshold"]
    r = evaluate(s >= thr, y, cert, s)
    for k in (1, 5):
        tk = np.quantile(s, 1 - k / 100)
        rk = evaluate(s >= tk, y, cert)
        r[f"row_recall_top{k}%"], r[f"banks_detected_top{k}%"] = rk["row_recall_%"], rk["banks_detected_%"]
    rows.append({"method": f"{label}: {cfg[label]['model']} @ frozen threshold", **r})
for name, a in rules.items():
    rows.append({"method": name, **evaluate(a, y, cert)})
rows.append({"method": "no alarms", **evaluate(np.zeros(len(y), bool), y, cert)})
res = pd.DataFrame(rows)
res.to_csv(REP / "test_metrics.csv", index=False)
show = ["method", "pr_auc", "roc_auc", "alarm_rate_%", "precision_rows_%", "row_recall_%", "banks_detected", "banks_detected_%", "saving_vs_no_alarm_%"]
say(res[show].to_string(index=False))
say("\nRecall when the top 1% / 5% of rows by score raise an alarm (row recall / banks detected):")
say(res[res.method.str.startswith(("baseline", "final"))][["method", "row_recall_top1%", "banks_detected_top1%", "row_recall_top5%", "banks_detected_top5%"]].to_string(index=False))

# ---- banks not yet critical
section("4. BANKS THAT ARE NOT YET CRITICAL (equity >= 2% of assets at the report date)")
mask = ~np.isnan(eq) & (eq >= 2)
rows = []
for label in ("baseline", "final"):
    rows.append({"method": label + ": " + cfg[label]["model"], **evaluate(scores[label] >= cfg[label]["threshold"], y, cert, scores[label], mask)})
sub = pd.DataFrame(rows)
sub.to_csv(REP / "test_not_critical.csv", index=False)
say(sub[["method", "rows", "failed_banks", "pr_auc", "roc_auc", "alarm_rate_%", "banks_detected", "banks_detected_%"]].to_string(index=False))

# ---- by failure year
section("5. FAILING BANKS BY YEAR OF FAILURE (detected = at least one alarm in the 4 quarters before failure)")
d = pd.DataFrame({"c": cert, "y": y, "fin": scores["final"] >= cfg["final"]["threshold"],
                  "base": scores["baseline"] >= cfg["baseline"]["threshold"], "r1": rules["rule: equity < 2% of assets"],
                  "r2": rules["rule: Texas ratio > 100"]})
fb = d[d.y == 1].groupby("c")[["fin", "base", "r1", "r2"]].max()
fb["year"] = [fyear.get(int(c), -1) for c in fb.index]
by = fb.groupby("year").agg(failed=("fin", "size"), final=("fin", "sum"), baseline=("base", "sum"),
                            rule_equity=("r1", "sum"), rule_texas=("r2", "sum")).reset_index()
tot = by[["failed", "final", "baseline", "rule_equity", "rule_texas"]].sum()
by = pd.concat([by, pd.DataFrame([{"year": "all", **tot.to_dict()}])], ignore_index=True)
by.to_csv(REP / "test_by_year.csv", index=False)
say(by.to_string(index=False))

# ---- calibration
section("6. CALIBRATION OF THE FINAL SCORES")
s = scores["final"]
bins = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0001]
cut = pd.cut(s, bins, right=False)
cal = pd.DataFrame({"bin": cut, "s": s, "y": y}).groupby("bin", observed=True).agg(rows=("y", "size"), mean_score=("s", "mean"), observed_rate=("y", "mean")).reset_index()
cal.to_csv(REP / "test_calibration.csv", index=False)
say(cal.to_string(index=False))
say("(Scores of a class-weighted model are not probabilities; only the ranking and the frozen threshold are used.)")

# ---- cluster bootstrap by bank
section(f"7. UNCERTAINTY: BOOTSTRAP OVER BANKS ({N_BOOT} resamples)")
order = np.argsort(cert, kind="stable")
cs = cert[order]
starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
lens = np.diff(np.r_[starts, len(cs)])
ids = np.arange(len(starts))
yo = y[order]
so = {k: v[order] for k, v in scores.items()}
rng = np.random.default_rng(SEED)
boot = []
for b in range(N_BOOT):
    pick = rng.choice(ids, size=len(ids), replace=True)
    idx = np.concatenate([np.arange(starts[i], starts[i] + lens[i]) for i in pick])
    yy = yo[idx]
    if yy.sum() == 0:
        continue
    rec = {}
    for label in ("baseline", "final"):
        rec[label + "_pr_auc"] = average_precision_score(yy, so[label][idx])
        rec[label + "_roc_auc"] = roc_auc_score(yy, so[label][idx])
    rec["diff_pr_auc"] = rec["final_pr_auc"] - rec["baseline_pr_auc"]
    boot.append(rec)
bt = pd.DataFrame(boot)
ci = bt.quantile([0.025, 0.5, 0.975]).T
ci.columns = ["2.5%", "median", "97.5%"]
ci.to_csv(REP / "test_bootstrap.csv")
say(ci.to_string())

# ---- cost sensitivity at the frozen threshold
section("8. COST SENSITIVITY (frozen threshold; alarm cost = rho x median failure cost; lambda = share of a failure's cost an alarm avoids)")
rows = []
a_fin = scores["final"] >= cfg["final"]["threshold"]
base_eval = evaluate(a_fin, y, cert)
n_alarm = base_eval["alarm_rows"]
cost_missed = base_eval["cost_kUSD"] - C_FA * n_alarm
cost_detected = base_eval["cost_no_alarm_kUSD"] - cost_missed
for lam in [1.0, 0.5, 0.25]:
    for rho in [0.001, 0.003, 0.01, 0.03, 0.1]:
        tot = rho * MEDIAN_COST * n_alarm + cost_missed + (1 - lam) * cost_detected
        rows.append({"lambda": lam, "rho": rho, "model_cost_kUSD": tot, "no_alarm_cost_kUSD": base_eval["cost_no_alarm_kUSD"],
                     "saving_%": (1 - tot / base_eval["cost_no_alarm_kUSD"]) * 100})
cs_ = pd.DataFrame(rows)
cs_.to_csv(REP / "test_cost_sensitivity.csv", index=False)
say(cs_.to_string(index=False))
say("lambda = 1 is the optimistic case in which an alarm avoids the whole cost of the failure; lambda < 1 is more realistic.")

out = {"evaluated": time.strftime("%Y-%m-%dT%H:%M:%S"), "config_sha256": sha,
       "final": {k: (None if pd.isna(v) else float(v)) for k, v in res.iloc[1].drop("method").items()},
       "baseline": {k: (None if pd.isna(v) else float(v)) for k, v in res.iloc[0].drop("method").items()}}
RESULT.write_text(json.dumps(out, indent=1), encoding="utf-8")
(REP / "part5_test_report.txt").write_text("\n".join(LOG), encoding="utf-8")
say("\nDONE. The test period has now been used. Do not change the model and re-run.")