"""
Part 2 - Step 2: build the CLEAN bank-quarter panel with exits and the label.

Label   : y = 1 if the bank fails in (REPDTE, REPDTE + 4 quarters]  (mergers are NOT failures)
Reads   : data_raw/fdic/financials/*.parquet, failures.parquet, institutions.parquet
Writes  : data_clean/panel.parquet  (zstd)  +  reports/quality/panel_build_*.csv/.txt
Run     : python src/04_build_panel.py        (~2-4 minutes)

Point-in-time rules used here
 - every cleaning rule looks at ONE row only (no statistics computed on the whole history)
 - rows dated on/after a bank's failure date are removed
 - columns starting with 'meta_' and the columns y / label_complete use FUTURE information:
   they must NEVER be used as model features.
"""
from __future__ import annotations

import warnings
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

FIN_DIR = Path("data_raw/fdic/financials")
FAIL_PATH = Path("data_raw/fdic/failures.parquet")
OUT_DIR = Path("data_clean")
REP = Path("reports/quality")
OUT_DIR.mkdir(exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

# Which rows of the Failures table count as failure events. None = all rows
# (the table contains failures AND assistance transactions). Example: ["FAILURE"].
EVENT_RESTYPES = None

NUM = ["ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",
       "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",
       "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",
       "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",
       "ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ"]
FIN_FIELDS = ["CERT", "REPDTE", "STNAME"] + NUM

pd.set_option("display.width", 220)
pd.options.display.float_format = "{:,.2f}".format
LOG: list[str] = []


def say(*a) -> None:
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def section(t: str) -> None:
    say("\n" + "=" * 78)
    say(t)
    say("=" * 78)


def parse_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    s = s.astype("string")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d1 = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
        d2 = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        d3 = pd.to_datetime(s, errors="coerce")
    return d1.combine_first(d2).combine_first(d3)


attrition: list[dict] = []


def step(name: str, df: pl.DataFrame) -> None:
    attrition.append({"step": name, "rows": df.height, "banks": df["CERT"].n_unique()})
    say(f"[{name}] rows {df.height:,}  banks {df['CERT'].n_unique():,}")


# ============================================================================
section("1. LOAD financials with one explicit schema")
# ============================================================================
parts = []
for f in sorted(FIN_DIR.glob("*.parquet")):
    d = pl.read_parquet(f)
    for c in FIN_FIELDS:
        if c not in d.columns:
            d = d.with_columns(pl.lit(None).alias(c))
    d = d.with_columns([pl.col(c).cast(pl.Float64, strict=False).fill_nan(None) for c in NUM])
    d = d.with_columns(pl.col("CERT").cast(pl.Int64, strict=False),
                       pl.col("REPDTE").cast(pl.Utf8), pl.col("STNAME").cast(pl.Utf8))
    parts.append(d.select(FIN_FIELDS))
panel = pl.concat(parts, how="vertical")
del parts
panel = panel.with_columns(
    pl.coalesce(pl.col("REPDTE").str.to_date("%Y%m%d", strict=False),
                pl.col("REPDTE").str.to_date("%Y-%m-%d", strict=False)).alias("date"))
panel = panel.with_columns(
    (pl.col("date").dt.year().cast(pl.Int32) * 4 + (pl.col("date").dt.month().cast(pl.Int32) - 1) // 3).alias("qi"))
LAST_DATE = panel["date"].max()
LABEL_CUTOFF = date(LAST_DATE.year - 1, LAST_DATE.month, LAST_DATE.day)
say(f"Last quarter in data: {LAST_DATE}. Labels are complete only for REPDTE <= {LABEL_CUTOFF}.")
step("raw panel", panel)

# ============================================================================
section("2. ROW-LEVEL CLEANING (each rule uses only the row itself)")
# ============================================================================
n0 = panel.height
panel = panel.filter(pl.col("CERT").is_not_null() & pl.col("date").is_not_null())
panel = panel.unique(subset=["CERT", "date"], keep="first", maintain_order=True)
say(f"Rows removed: missing CERT/date or duplicate key = {n0 - panel.height:,}")
step("after key checks", panel)
n0 = panel.height
panel = panel.filter(pl.col("ASSET").is_not_null() & (pl.col("ASSET") > 0))
say(f"Rows removed: ASSET missing or <= 0 = {n0 - panel.height:,}")
step("after ASSET > 0", panel)

fix_rules = [
    ("EQ > ASSET -> EQ set to missing", ["EQ"], pl.col("EQ") > pl.col("ASSET")),
    ("DEP < 0 -> DEP set to missing", ["DEP"], pl.col("DEP") < 0),
    ("LNLSGR > ASSET -> LNLSGR, LNLSNET set to missing", ["LNLSGR", "LNLSNET"], pl.col("LNLSGR") > pl.col("ASSET")),
    ("NCLNLS < 0 -> NCLNLS set to missing", ["NCLNLS"], pl.col("NCLNLS") < 0),
]
fix_rows = []
for name, cols, cond in fix_rules:
    cond = cond.fill_null(False)
    k = panel.select(cond.sum()).item()
    panel = panel.with_columns([pl.when(cond).then(None).otherwise(pl.col(c)).alias(c) for c in cols])
    fix_rows.append({"rule": name, "rows_affected": k})
    say(f"{name}: {k:,} rows")
say("Kept on purpose: negative equity (insolvency signal), DEP > ASSET (flagged only), extreme ROA/ROE/RBC "
    "(winsorised later, on training data only).")
panel = panel.sort(["CERT", "date"])

# ============================================================================
section("3. FAILURE EVENTS (from the Failures table)")
# ============================================================================
fails = pd.read_parquet(FAIL_PATH)
if "RESTYPE" not in fails.columns:
    fails["RESTYPE"] = None
fails["CERT"] = pd.to_numeric(fails["CERT"], errors="coerce")
fails["fail_date"] = parse_date(fails["FAILDATE"])
say(f"Failures table rows: {len(fails):,}")
say(f"  without CERT: {fails.CERT.isna().sum():,}   without valid FAILDATE: {fails.fail_date.isna().sum():,}")
say("RESTYPE values (failure vs assistance):")
say(fails["RESTYPE"].value_counts(dropna=False).to_string())
ev = fails.dropna(subset=["CERT", "fail_date"]).copy()
ev["CERT"] = ev.CERT.astype("int64")
if EVENT_RESTYPES:
    ev = ev[ev.RESTYPE.isin(EVENT_RESTYPES)]
say(f"Event rows used: {len(ev):,}; distinct banks: {ev.CERT.nunique():,}; "
    f"banks with more than one event row: {int((ev.CERT.value_counts() > 1).sum()):,} (earliest event is used)")
first_ev = ev.sort_values("fail_date").drop_duplicates("CERT")[["CERT", "fail_date", "RESTYPE"]]
first_ev = first_ev.rename(columns={"fail_date": "meta_fail_date", "RESTYPE": "restype"})
ev_pl = pl.from_pandas(first_ev).with_columns(pl.col("meta_fail_date").cast(pl.Date),
                                              pl.col("restype").cast(pl.Utf8))
panel = panel.join(ev_pl, on="CERT", how="left")

# ============================================================================
section("4. EXITS: remove observations on/after the failure date")
# ============================================================================
post = (pl.col("meta_fail_date").is_not_null() & (pl.col("date") >= pl.col("meta_fail_date"))).fill_null(False)
pdf = panel.filter(post)
say(f"Rows dated on/after the failure date: {pdf.height:,} (banks: {pdf['CERT'].n_unique():,})")
if pdf.height:
    say(pdf.group_by("restype").agg(pl.len().alias("rows"), pl.col("CERT").n_unique().alias("banks"))
        .sort("rows", descending=True).to_pandas().to_string(index=False))
panel = panel.filter(~post)
step("after removing post-failure rows", panel)

last = panel.group_by("CERT").agg(pl.col("date").max().alias("last_date"))
panel = panel.join(last, on="CERT", how="left")
panel = panel.with_columns(
    pl.when(pl.col("meta_fail_date").is_not_null()).then(pl.lit("failure"))
    .when(pl.col("last_date") < pl.lit(LAST_DATE)).then(pl.lit("other_exit"))
    .otherwise(pl.lit("active")).alias("meta_exit_type"))
say("Banks by exit type (failure = in Failures table; other_exit = merger/closure/etc., NOT a failure):")
say(panel.group_by("meta_exit_type").agg(pl.col("CERT").n_unique().alias("banks"), pl.len().alias("rows"))
    .sort("banks", descending=True).to_pandas().to_string(index=False))

# ============================================================================
section("5. LABEL y and helper columns")
# ============================================================================
panel = panel.with_columns(
    pl.col("date").dt.offset_by("1y").alias("_h_end"),
    (pl.col("qi") - pl.col("qi").shift(1)).over("CERT").alias("qgap_prev"))
panel = panel.with_columns(
    (pl.col("meta_fail_date").is_not_null() & (pl.col("meta_fail_date") > pl.col("date"))
     & (pl.col("meta_fail_date") <= pl.col("_h_end"))).cast(pl.Int8).alias("y"),
    (pl.col("date") <= pl.lit(LABEL_CUTOFF)).alias("label_complete"))
panel = panel.with_columns(
    pl.when(pl.col("y") == 1).then((pl.col("meta_fail_date") - pl.col("date")).dt.total_days())
    .otherwise(None).alias("meta_days_to_fail"))
panel = panel.select(["CERT", "date", "REPDTE", "qi", "STNAME"] + NUM +
                     ["qgap_prev", "meta_exit_type", "meta_fail_date", "meta_days_to_fail",
                      "label_complete", "y"])

lab = panel.filter(pl.col("label_complete"))
say(f"Rows with a complete label (REPDTE <= {LABEL_CUTOFF}): {lab.height:,} of {panel.height:,}")
say(f"Positive rows (y=1): {int(lab['y'].sum()):,}  ({lab['y'].mean() * 100:.3f}% of labelled rows)")
say(f"Rows with a gap before them (qgap_prev > 1): {panel.filter(pl.col('qgap_prev') > 1).height:,}")

fb = (panel.filter(pl.col("meta_exit_type") == "failure")
      .group_by("CERT").agg(pl.col("meta_fail_date").first().alias("fd"), pl.col("y").sum().alias("n_pos"),
                            (pl.col("y").sum() > 0).alias("has_pos"))
      .with_columns(pl.col("fd").dt.year().alias("fail_year")).to_pandas())
lab_pos = (lab.filter(pl.col("y") == 1).group_by("CERT").agg(pl.col("meta_fail_date").first().alias("fd"))
           .with_columns(pl.col("fd").dt.year().alias("fail_year")).to_pandas())
usable = pd.DataFrame({
    "failed_banks_in_panel": fb.groupby("fail_year").size(),
    "with_labelled_positive_rows": lab_pos.groupby("fail_year").size()}).fillna(0).astype(int)
usable.index.name = "fail_year"
usable.to_csv(REP / "failures_usable_by_year.csv")
say("\nFailed banks per year: in the panel / with at least one labelled positive row")
say(usable.to_string())
say(f"Failed banks with NO positive row (stopped reporting >4 quarters early or too recent): "
    f"{int((~fb.has_pos).sum()):,}")

# ============================================================================
section("6. SANITY CHECKS (must all be PASS)")
# ============================================================================
checks = {
    "no duplicate (CERT, date)": panel.height == panel.select(["CERT", "date"]).unique().height,
    "no row on/after failure date": panel.filter(pl.col("meta_fail_date").is_not_null() &
                                                 (pl.col("date") >= pl.col("meta_fail_date"))).height == 0,
    "non-failed banks never have y=1": panel.filter(pl.col("meta_exit_type") != "failure")["y"].sum() == 0,
    "y=1 rows fail within 1..366 days": (lambda s: s.is_empty() or (s.min() >= 1 and s.max() <= 366))(
        panel.filter(pl.col("y") == 1)["meta_days_to_fail"].drop_nulls()),
    "ASSET > 0 everywhere": panel.filter(pl.col("ASSET") <= 0).height == 0,
}
for k, v in checks.items():
    say(f"{'PASS' if v else 'FAIL'}  {k}")

# ============================================================================
section("7. SAVE")
# ============================================================================
panel.write_parquet(OUT_DIR / "panel.parquet", compression="zstd")
say(f"Saved {OUT_DIR / 'panel.parquet'}: {panel.height:,} rows x {panel.width} columns, "
    f"{(OUT_DIR / 'panel.parquet').stat().st_size / 1024**2:.1f} MB")
pd.DataFrame(attrition).to_csv(REP / "panel_build_attrition.csv", index=False)
pd.DataFrame(fix_rows).to_csv(REP / "panel_build_fixes.csv", index=False)
say("Columns starting with 'meta_', plus y and label_complete, are NOT features (they use future information).")
(REP / "panel_build_report.txt").write_text("\n".join(LOG), encoding="utf-8")
say("DONE.")