"""
Part 2 - Automated data-quality audit of the FDIC panel (step 1: AUDIT only).

Reads  : data_raw/fdic/financials/*.parquet, failures.parquet, institutions.parquet
Writes : reports/quality/*.csv, *.png and quality_report.txt
Run    : python src/03_quality_report.py      (from the repo root, ~2-4 minutes)
"""
from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

FIN_DIR = Path("data_raw/fdic/financials")
FAIL_PATH = Path("data_raw/fdic/failures.parquet")
INST_PATH = Path("data_raw/fdic/institutions.parquet")
OUT = Path("reports/quality")
OUT.mkdir(parents=True, exist_ok=True)

NUM = ["ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",
       "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",
       "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",
       "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",
       "ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ"]
FIN_FIELDS = ["CERT", "REPDTE", "STNAME"] + NUM

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
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
    """Parse dates written as M/D/YYYY or YYYY-MM-DD (or already datetime)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    s = s.astype("string")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d1 = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
        d2 = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        d3 = pd.to_datetime(s, errors="coerce")
    return d1.combine_first(d2).combine_first(d3)


# ============================================================================
section("1. SCHEMA AUDIT: are column types identical in all 170 files?")
# ============================================================================
files = sorted(FIN_DIR.glob("*.parquet"))
say(f"Quarterly files found: {len(files)}")
types: dict[str, Counter] = defaultdict(Counter)
missing_cols: Counter = Counter()
for f in files:
    sch = pq.read_schema(f)
    names = set(sch.names)
    for n, t in zip(sch.names, sch.types):
        types[n][str(t)] += 1
    for c in FIN_FIELDS:
        if c not in names:
            missing_cols[c] += 1
rows = [{"column": c,
         "types_found": "; ".join(f"{t} x{k}" for t, k in types[c].items()),
         "n_types": len(types[c]),
         "files_without_column": missing_cols[c]} for c in FIN_FIELDS]
schema_df = pd.DataFrame(rows)
schema_df.to_csv(OUT / "schema_audit.csv", index=False)
bad = schema_df[(schema_df.n_types > 1) | (schema_df.files_without_column > 0)]
say(f"Columns with mixed types or absent from some files: {len(bad)} of {len(FIN_FIELDS)}")
if len(bad):
    say(bad.to_string(index=False))

# ============================================================================
section("2. LOAD AND TYPE THE PANEL (one explicit schema)")
# ============================================================================
parts, coerced = [], Counter()
for f in files:
    d = pl.read_parquet(f)
    for c in FIN_FIELDS:
        if c not in d.columns:
            d = d.with_columns(pl.lit(None).alias(c))
    for c in NUM:
        before = d[c].null_count()
        d = d.with_columns(pl.col(c).cast(pl.Float64, strict=False).fill_nan(None))
        coerced[c] += d[c].null_count() - before
    d = d.with_columns(pl.col("CERT").cast(pl.Int64, strict=False),
                       pl.col("REPDTE").cast(pl.Utf8),
                       pl.col("STNAME").cast(pl.Utf8))
    parts.append(d.select(FIN_FIELDS))
panel = pl.concat(parts, how="vertical")
del parts
say(f"Panel: {panel.height:,} rows x {panel.width} columns")
say(f"Values that could not be converted to numbers (set to missing): {sum(coerced.values()):,}")

panel = panel.with_columns(
    pl.coalesce(pl.col("REPDTE").str.to_date("%Y%m%d", strict=False),
                pl.col("REPDTE").str.to_date("%Y-%m-%d", strict=False)).alias("date"))
panel = panel.with_columns(pl.col("date").dt.year().cast(pl.Int32).alias("year"),
                           pl.col("date").dt.month().cast(pl.Int32).alias("month"))
panel = panel.with_columns((pl.col("year") * 4 + (pl.col("month") - 1) // 3).alias("qi"))

# ============================================================================
section("3. KEY CHECKS")
# ============================================================================
n = panel.height
n_dup = n - panel.select(["CERT", "REPDTE"]).unique().height
not_qend = panel.filter(~pl.col("month").is_in([3, 6, 9, 12]) |
                        (pl.col("date") != pl.col("date").dt.month_end())).height
say(f"Rows                          : {n:,}")
say(f"Missing CERT                  : {panel['CERT'].null_count():,}")
say(f"Unparseable REPDTE            : {panel['date'].null_count():,}")
say(f"REPDTE not a quarter-end      : {not_qend:,}")
say(f"Duplicate (CERT, REPDTE)      : {n_dup:,}")
say(f"Distinct quarters / banks     : {panel['date'].n_unique()} / {panel['CERT'].n_unique():,}")
say(f"First / last quarter          : {panel['date'].min()} / {panel['date'].max()}")

# ============================================================================
section("4. MISSING VALUES by variable and by year (schema drift)")
# ============================================================================
overall = pd.Series({c: panel[c].null_count() / n for c in NUM}).sort_values(ascending=False)
overall.mul(100).rename("pct_missing").to_csv(OUT / "missing_by_variable.csv")
say("Share of missing values per variable (%), top 12:")
say(overall.head(12).mul(100).round(2).to_string())

by_year = (panel.group_by("year")
           .agg([pl.col(c).is_null().mean().alias(c) for c in NUM])
           .sort("year").to_pandas().set_index("year"))
by_year.mul(100).round(2).to_csv(OUT / "missing_by_year.csv")
late = []
for c in NUM:
    ok = by_year.index[by_year[c] < 0.5]
    first = int(ok.min()) if len(ok) else None
    if first is None or first > by_year.index.min():
        late.append((c, first))
say("\nVariables that are mostly missing in the early years (first year with <50% missing):")
say(", ".join(f"{c}: {y}" for c, y in late) if late else "none")

fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(by_year.T.values * 100, aspect="auto", cmap="Reds", vmin=0, vmax=100)
ax.set_yticks(range(len(NUM)))
ax.set_yticklabels(NUM, fontsize=7)
xt = list(range(0, len(by_year), 5))
ax.set_xticks(xt)
ax.set_xticklabels([by_year.index[i] for i in xt], fontsize=8)
ax.set_title("Share of missing values (%) by variable and year")
fig.colorbar(im, label="% missing")
fig.tight_layout()
fig.savefig(OUT / "missing_heatmap.png", dpi=200)
plt.close(fig)

# ============================================================================
section("5. IMPLAUSIBLE VALUES (rules: count, share, proposed handling)")
# ============================================================================
a, e = pl.col("ASSET"), pl.col("EQ")
rules = [
    ("ASSET missing or <= 0", a.is_null() | (a <= 0), "EXCLUDE from modelling: ratios undefined"),
    ("EQ < 0 (negative equity)", e < 0, "KEEP: insolvency signal, not an error"),
    ("EQ > ASSET (ASSET > 0)", (e > a) & (a > 0), "FLAG; set EQ-based ratios to missing"),
    ("DEP < 0", pl.col("DEP") < 0, "FLAG; set to missing"),
    ("DEP > ASSET", pl.col("DEP") > a, "FLAG; keep, check size"),
    ("DEPDOM > DEP", pl.col("DEPDOM") > pl.col("DEP"), "FLAG; keep"),
    ("LNLSGR > ASSET", pl.col("LNLSGR") > a, "FLAG; set loan ratios to missing"),
    ("LNLSNET > LNLSGR", pl.col("LNLSNET") > pl.col("LNLSGR"), "FLAG; keep"),
    ("NCLNLS < 0", pl.col("NCLNLS") < 0, "FLAG; set to missing"),
    ("|ROA| > 50 (%)", pl.col("ROA").abs() > 50, "KEEP value; winsorise in training pipeline"),
    ("|RBC1AAJ| > 100 (%)", pl.col("RBC1AAJ").abs() > 100, "KEEP value; winsorise in training pipeline"),
]
cnt = panel.select([ex.fill_null(False).sum().alias(f"r{i}") for i, (_, ex, _) in enumerate(rules)]).row(0)
rule_df = pd.DataFrame({"rule": [r[0] for r in rules], "rows": cnt,
                        "share_%": [c / n * 100 for c in cnt],
                        "proposed_handling": [r[2] for r in rules]})
rule_df.to_csv(OUT / "implausible_rules.csv", index=False)
say(rule_df.to_string(index=False))

# ============================================================================
section("6. DISTRIBUTIONS, OUTLIERS AND UNITS")
# ============================================================================
chk = panel.with_columns(pl.when(a > 0).then(e / a * 100).otherwise(None).alias("EQ_ASSET_pct"))
vars_ = ["ASSET", "EQ", "DEP", "LNLSGR", "NCLNLS", "NETINC", "ROA", "ROE", "NIMY",
         "RBC1AAJ", "RBCRWAJ", "EQ_ASSET_pct"]
qs = [0, 0.1, 1, 50, 99, 99.9, 100]
rows = []
for c in vars_:
    x = chk[c].drop_nulls().to_numpy()
    rows.append(dict(variable=c, **{f"p{q}": np.percentile(x, q) for q in qs}))
pct_df = pd.DataFrame(rows)
pct_df.to_csv(OUT / "percentiles.csv", index=False)
say(pct_df.to_string(index=False))

ye = (panel.filter(pl.col("month") == 12).group_by("year")
      .agg(pl.len().alias("n_banks"),
           pl.col("ASSET").median().alias("median_ASSET_kUSD"),
           (pl.col("ASSET").sum() / 1e9).alias("total_ASSET_trillion_USD"))
      .sort("year").to_pandas())
ye.to_csv(OUT / "units_check_by_year.csv", index=False)
say("\nUnit check (ASSET is in thousand USD; totals in trillion USD), selected year-ends:")
say(ye[ye.year.isin([1985, 1990, 2000, 2008, 2010, 2020, 2023, 2025])].to_string(index=False))

# ============================================================================
section("7. CONTINUITY: gaps in each bank's quarterly series")
# ============================================================================
bank = (panel.filter(pl.col("qi").is_not_null()).group_by("CERT")
        .agg(pl.len().alias("n_obs"), pl.col("qi").min().alias("first_q"),
             pl.col("qi").max().alias("last_q"))
        .with_columns((pl.col("last_q") - pl.col("first_q") + 1 - pl.col("n_obs")).alias("n_gap_q")))
bk = bank.to_pandas()
say(f"Banks: {len(bk):,}")
say(f"Banks with at least one missing quarter between first and last report: {(bk.n_gap_q > 0).sum():,}")
say(f"Total missing bank-quarters inside series: {int(bk.n_gap_q.sum()):,}")
say(f"Banks with a single observation: {(bk.n_obs == 1).sum():,}")
say("Observations per bank (quarters): " + bk.n_obs.describe().round(1).to_dict().__str__())
bk[bk.n_gap_q > 0].to_csv(OUT / "banks_with_gaps.csv", index=False)

cq = panel.group_by("date").agg(pl.len().alias("n")).sort("date").to_pandas()
cq.to_csv(OUT / "banks_per_quarter.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(pd.to_datetime(cq["date"]), cq["n"])
ax.set_title("Number of reporting banks per quarter")
ax.set_ylabel("banks")
fig.tight_layout()
fig.savefig(OUT / "banks_per_quarter.png", dpi=200)
plt.close(fig)

# ============================================================================
section("8. FAILURES AND INSTITUTIONS: raw formats and cross-table checks")
# ============================================================================
fails = pd.read_parquet(FAIL_PATH)
inst = pd.read_parquet(INST_PATH)
say("Failures: dtypes and first rows (to see the raw date format)")
say(fails[["CERT", "FAILDATE", "FAILYR", "COST"]].dtypes.to_string())
say(fails[["CERT", "FAILDATE", "FAILYR", "COST"]].head(4).to_string())
say("\nInstitutions: first rows")
say(inst[["CERT", "ACTIVE", "ESTYMD", "ENDEFYMD", "CHANGEC1"]].head(4).to_string())

fails["CERT"] = pd.to_numeric(fails["CERT"], errors="coerce")
inst["CERT"] = pd.to_numeric(inst["CERT"], errors="coerce")
fails["fail_date"] = parse_date(fails["FAILDATE"])
fin_c, fail_c, inst_c = set(bk.CERT), set(fails.CERT.dropna().astype(int)), set(inst.CERT.dropna().astype(int))
say(f"\nFailures rows {len(fails):,}; distinct CERT {len(fail_c):,}; "
    f"unparseable FAILDATE {fails.fail_date.isna().sum():,}")
say(f"CERT overlap  Financials & Failures: {len(fin_c & fail_c):,}   "
    f"Failures not in Financials: {len(fail_c - fin_c):,}")
say(f"CERT overlap  Financials & Institutions: {len(fin_c & inst_c):,}   "
    f"Financials not in Institutions: {len(fin_c - inst_c):,}   "
    f"Institutions not in Financials: {len(inst_c - fin_c):,}")

fails["fail_year"] = fails.fail_date.dt.year
fails["in_fin"] = fails.CERT.isin(fin_c)
tab = (fails.groupby("fail_year").agg(n_failures=("CERT", "size"), n_in_financials=("in_fin", "sum"))
       .reset_index())
tab.to_csv(OUT / "failures_by_year.csv", index=False)
say("\nFailures by year (only those with Financials data can become positive labels):")
say(tab[tab.fail_year >= 1984].astype(int).to_string(index=False))
say(f"Failures before 1984 (no Financials possible): {int(tab[tab.fail_year < 1984].n_failures.sum()):,}")
cost = pd.to_numeric(fails["COST"], errors="coerce")
say(f"\nCOST (thousand USD): non-missing {cost.notna().sum():,} of {len(fails):,}; "
    f"median {cost.median():,.0f}; mean {cost.mean():,.0f}; total {cost.sum():,.0f}")

ff = (fails.dropna(subset=["CERT", "fail_date"]).sort_values("fail_date")
      .drop_duplicates("CERT")[["CERT", "fail_date"]].copy())
ff["CERT"] = ff.CERT.astype("int64")
ff["fail_q"] = ff.fail_date.dt.year * 4 + (ff.fail_date.dt.month - 1) // 3
m = ff.merge(bk[["CERT", "last_q"]], on="CERT", how="inner")
m["lag_q"] = m.fail_q - m.last_q
say("\nFailed banks: (quarter of failure) - (quarter of last Call Report).")
say("  1 = last report is the quarter-end just before failure (normal); <=0 = report at/after failure; >=2 = stopped reporting early")
say(m.lag_q.clip(-2, 6).value_counts().sort_index().rename("n_banks").to_string())
pf = pl.from_pandas(ff[["CERT", "fail_date"]]).with_columns(pl.col("fail_date").cast(pl.Date))
post = panel.select(["CERT", "date"]).join(pf, on="CERT", how="inner").filter(pl.col("date") > pl.col("fail_date"))
say(f"Rows dated AFTER the failure date: {post.height:,} (banks: {post['CERT'].n_unique():,})")

# ============================================================================
section("9. EXITS: who stops reporting, and why (failure vs other)")
# ============================================================================
max_q = int(bk.last_q.max())
bk["exited"] = bk.last_q < max_q
bk["in_failures"] = bk.CERT.isin(fail_c)
say(f"Last quarter in panel (index {max_q}). Banks whose last report is earlier = 'exited'.")
say(pd.crosstab(bk.exited, bk.in_failures, margins=True).rename(
    index={True: "exited", False: "still reporting"}, columns={True: "in Failures", False: "not in Failures"}).to_string())

i1 = inst.drop_duplicates("CERT")[["CERT", "ACTIVE", "CHANGEC1", "ENDEFYMD"]].copy()
i1["end_date"] = parse_date(i1["ENDEFYMD"])
i1["end_q"] = i1.end_date.dt.year * 4 + (i1.end_date.dt.month - 1) // 3
bk = bk.merge(i1, on="CERT", how="left")
bk["group"] = np.where(~bk.exited, "still reporting", np.where(bk.in_failures, "exit: failure", "exit: other"))
for g in ["exit: failure", "exit: other", "still reporting"]:
    sub = bk[bk.group == g]
    say(f"\n--- {g}: {len(sub):,} banks ---")
    say("ACTIVE flag in Institutions: " + sub.ACTIVE.value_counts(dropna=False).to_dict().__str__())
    if g != "still reporting":
        say("Top CHANGEC1 codes:")
        say(sub.CHANGEC1.value_counts(dropna=False).head(12).rename("n").to_string())
        lag = (sub.end_q - sub.last_q).dropna().clip(-4, 8)
        say("ENDEFYMD quarter minus last-report quarter (clipped to -4..8):")
        say(lag.value_counts().sort_index().rename("n").to_string())
bk.drop(columns=["end_date"]).to_csv(OUT / "bank_summary.csv", index=False)

(OUT / "quality_report.txt").write_text("\n".join(LOG), encoding="utf-8")
say(f"\nDONE. Files saved in {OUT.resolve()}")