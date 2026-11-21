"""
Feature definitions and the training-only preprocessing pipeline (Part 4).

build_features(panel)  : point-in-time features, Polars (works on a DataFrame or a LazyFrame)
make_preprocessor()    : scikit-learn Pipeline; EVERY step learns its parameters in fit(), so it must be
                         fitted on training rows only and then applied unchanged to validation / test rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# name, CAMELS-style group, definition, why it matters
FEATURE_SPEC = [
    ("eq_assets", "Capital", "EQ / ASSET x 100", "Loss-absorbing buffer; thin capital precedes failure"),
    ("rbc1aaj", "Capital", "Tier-1 leverage ratio (RBC1AAJ)", "Regulatory capital adequacy used by supervisors"),
    ("rbcrwaj", "Capital", "Total risk-based capital ratio (RBCRWAJ)", "Capital relative to risk-weighted assets"),
    ("texas_ratio", "Capital", "(NCLNLS + ORE) / (EQ + LNATRES) x 100", "Bad assets against loss-absorbing resources; classic failure indicator"),
    ("neg_eq", "Capital", "1 if EQ < 0", "Book insolvency"),
    ("npl_ratio", "Asset quality", "NCLNLS / LNLSGR x 100", "Share of loans that do not perform"),
    ("p9_ratio", "Asset quality", "P9ASSET / ASSET x 100", "Assets 90+ days past due"),
    ("na_ratio", "Asset quality", "NAASSET / ASSET x 100", "Non-accrual assets"),
    ("ore_assets", "Asset quality", "ORE / ASSET x 100", "Foreclosed real estate, a trace of past credit losses"),
    ("reserves_loans", "Asset quality", "LNATRES / LNLSGR x 100", "Reserve cushion relative to loans"),
    ("reserves_npl", "Asset quality", "LNATRES / NCLNLS x 100", "Coverage of non-performing loans by reserves"),
    ("nco_rate", "Asset quality", "NTLNLS / LNLSGR x 100 x 12/month", "Annualised net charge-offs (realised losses)"),
    ("prov_rate", "Asset quality", "ELNATR / LNLSGR x 100 x 12/month", "Annualised loan-loss provisions (expected losses)"),
    ("roa", "Earnings", "ROA (annualised %, FDIC)", "Profits rebuild capital, losses erode it"),
    ("roe", "Earnings", "ROE (annualised %, FDIC)", "Return to shareholders"),
    ("nimy", "Earnings", "Net interest margin (NIMY)", "Core earning power of the loan and deposit business"),
    ("efficiency", "Earnings", "NONIX / (INTINC - EINTEXP + NONII) x 100", "Cost burden relative to revenue"),
    ("int_exp_share", "Earnings", "EINTEXP / INTINC x 100", "Share of interest income paid out as funding cost"),
    ("liquid", "Liquidity & funding", "(CHBAL + SC) / ASSET x 100", "Liquid assets that can meet withdrawals"),
    ("loans_dep", "Liquidity & funding", "LNLSNET / DEP x 100", "Loans financed by deposits"),
    ("loans_assets", "Liquidity & funding", "LNLSGR / ASSET x 100", "Share of assets tied up in loans"),
    ("bro_dep", "Liquidity & funding", "BRO / DEP x 100", "Reliance on brokered, volatile deposits"),
    ("dep_assets", "Liquidity & funding", "DEP / ASSET x 100", "Deposit funding share"),
    ("re_loans", "Composition & size", "LNRE / LNLSGR x 100", "Real-estate concentration"),
    ("ci_loans", "Composition & size", "LNCI / LNLSGR x 100", "Commercial and industrial loan share"),
    ("cons_loans", "Composition & size", "LNCON / LNLSGR x 100", "Consumer loan share"),
    ("sec_assets", "Composition & size", "SC / ASSET x 100", "Securities share, proxy for interest-rate sensitivity"),
    ("log_asset", "Composition & size", "ln(ASSET)", "Size; small banks fail more often"),
    ("asset_g4", "Growth", "Year-on-year growth of ASSET (%)", "Rapid growth often reflects risk taking"),
    ("loan_g4", "Growth", "Year-on-year growth of LNLSGR (%)", "Fast loan growth lowers underwriting quality"),
    ("dep_g4", "Growth", "Year-on-year growth of DEP (%)", "Fast deposit growth signals aggressive funding"),
    ("d4_eq_assets", "Dynamics", "eq_assets minus its value 4 quarters ago", "Speed of capital deterioration"),
    ("d4_npl_ratio", "Dynamics", "npl_ratio minus its value 4 quarters ago", "Speed of asset-quality deterioration"),
    ("d4_roa", "Dynamics", "roa minus its value 4 quarters ago", "Earnings trend"),
    ("d4_liquid", "Dynamics", "liquid minus its value 4 quarters ago", "Liquidity trend"),
    ("d4_texas_ratio", "Dynamics", "texas_ratio minus its value 4 quarters ago", "Trend of bad assets against resources"),
    ("roa_m4", "Dynamics", "Mean of roa over the last 4 quarters (past only)", "Persistent profitability"),
    ("roa_s4", "Dynamics", "Std of roa over the last 4 quarters (past only)", "Earnings volatility"),
    ("lag1_eq_assets", "Dynamics", "eq_assets one quarter ago", "Short memory of capital"),
    ("lag1_npl_ratio", "Dynamics", "npl_ratio one quarter ago", "Short memory of asset quality"),
    ("lag1_roa", "Dynamics", "roa one quarter ago", "Short memory of earnings"),
    ("rk_eq_assets", "Relative to peers", "Percentile rank of eq_assets in the same quarter", "Removes time effects; standing among peers"),
    ("rk_npl_ratio", "Relative to peers", "Percentile rank of npl_ratio in the same quarter", "Relative asset quality"),
    ("rk_roa", "Relative to peers", "Percentile rank of roa in the same quarter", "Relative profitability"),
    ("rk_texas_ratio", "Relative to peers", "Percentile rank of texas_ratio in the same quarter", "Relative bad-asset burden"),
    ("rk_liquid", "Relative to peers", "Percentile rank of liquid in the same quarter", "Relative liquidity"),
    ("rk_loans_dep", "Relative to peers", "Percentile rank of loans_dep in the same quarter", "Relative funding structure"),
]
FEATURES = [s[0] for s in FEATURE_SPEC]
FEATURE_GROUPS: dict[str, list[str]] = {}
for _n, _g, _d, _w in FEATURE_SPEC:
    FEATURE_GROUPS.setdefault(_g, []).append(_n)

RAW_NEEDED = ["CERT", "date", "qi", "ASSET", "DEP", "EQ", "BRO", "CHBAL", "SC", "LNLSGR", "LNLSNET", "LNRE",
              "LNCI", "LNCON", "LNATRES", "NCLNLS", "P9ASSET", "NAASSET", "ORE", "INTINC", "EINTEXP",
              "NONII", "NONIX", "ELNATR", "NTLNLS", "ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ"]


def _ratio(num: pl.Expr, den: pl.Expr, scale: float = 100.0) -> pl.Expr:
    """num/den*scale, missing when the denominator is missing or not positive (no infinite values)."""
    return pl.when(den > 0).then(num / den * scale)


def _lag(col: str, k: int) -> pl.Expr:
    """Value k QUARTERS ago (not k rows ago): missing if the bank has no report exactly k quarters earlier."""
    return pl.when((pl.col("qi") - pl.col("qi").shift(k).over("CERT")) == k).then(pl.col(col).shift(k).over("CERT"))


def _rank_pct(col: str) -> pl.Expr:
    """Percentile rank among all banks reporting in the SAME quarter (contemporaneous, no future information)."""
    return pl.col(col).rank("average").over("qi") / pl.col(col).count().over("qi")


def build_features(panel):
    """Point-in-time features. Only columns of RAW_NEEDED are read: 'y' and every 'meta_' column are never used.
    Rolling/lag operations look backwards only; the panel must hold one row per (CERT, quarter)."""
    lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel
    lf = lf.select(RAW_NEEDED).sort(["CERT", "qi"])
    c = pl.col
    ann = 12 / c("date").dt.month()                         # income items are year-to-date: annualise
    s1 = lf.with_columns(
        _ratio(c("EQ"), c("ASSET")).alias("eq_assets"),
        c("RBC1AAJ").alias("rbc1aaj"), c("RBCRWAJ").alias("rbcrwaj"),
        _ratio(c("NCLNLS") + c("ORE"), c("EQ") + c("LNATRES")).alias("texas_ratio"),
        pl.when(c("EQ").is_not_null()).then((c("EQ") < 0).cast(pl.Float64)).alias("neg_eq"),
        _ratio(c("NCLNLS"), c("LNLSGR")).alias("npl_ratio"),
        _ratio(c("P9ASSET"), c("ASSET")).alias("p9_ratio"),
        _ratio(c("NAASSET"), c("ASSET")).alias("na_ratio"),
        _ratio(c("ORE"), c("ASSET")).alias("ore_assets"),
        _ratio(c("LNATRES"), c("LNLSGR")).alias("reserves_loans"),
        _ratio(c("LNATRES"), c("NCLNLS")).alias("reserves_npl"),
        (_ratio(c("NTLNLS"), c("LNLSGR")) * ann).alias("nco_rate"),
        (_ratio(c("ELNATR"), c("LNLSGR")) * ann).alias("prov_rate"),
        c("ROA").alias("roa"), c("ROE").alias("roe"), c("NIMY").alias("nimy"),
        _ratio(c("NONIX"), c("INTINC") - c("EINTEXP") + c("NONII")).alias("efficiency"),
        _ratio(c("EINTEXP"), c("INTINC")).alias("int_exp_share"),
        _ratio(c("CHBAL") + c("SC"), c("ASSET")).alias("liquid"),
        _ratio(c("LNLSNET"), c("DEP")).alias("loans_dep"),
        _ratio(c("LNLSGR"), c("ASSET")).alias("loans_assets"),
        _ratio(c("BRO"), c("DEP")).alias("bro_dep"),
        _ratio(c("DEP"), c("ASSET")).alias("dep_assets"),
        _ratio(c("LNRE"), c("LNLSGR")).alias("re_loans"),
        _ratio(c("LNCI"), c("LNLSGR")).alias("ci_loans"),
        _ratio(c("LNCON"), c("LNLSGR")).alias("cons_loans"),
        _ratio(c("SC"), c("ASSET")).alias("sec_assets"),
        c("ASSET").log().alias("log_asset"),
        pl.when(_lag("ASSET", 4) > 0).then((c("ASSET") / _lag("ASSET", 4) - 1) * 100).alias("asset_g4"),
        pl.when(_lag("LNLSGR", 4) > 0).then((c("LNLSGR") / _lag("LNLSGR", 4) - 1) * 100).alias("loan_g4"),
        pl.when(_lag("DEP", 4) > 0).then((c("DEP") / _lag("DEP", 4) - 1) * 100).alias("dep_g4"),
    )
    span3 = (c("qi") - c("qi").shift(3).over("CERT")) == 3   # four consecutive quarters present
    s2 = s1.with_columns(
        (c("eq_assets") - _lag("eq_assets", 4)).alias("d4_eq_assets"),
        (c("npl_ratio") - _lag("npl_ratio", 4)).alias("d4_npl_ratio"),
        (c("roa") - _lag("roa", 4)).alias("d4_roa"),
        (c("liquid") - _lag("liquid", 4)).alias("d4_liquid"),
        (c("texas_ratio") - _lag("texas_ratio", 4)).alias("d4_texas_ratio"),
        pl.when(span3).then(c("roa").rolling_mean(window_size=4, min_samples=4).over("CERT")).alias("roa_m4"),
        pl.when(span3).then(c("roa").rolling_std(window_size=4, min_samples=4).over("CERT")).alias("roa_s4"),
        _lag("eq_assets", 1).alias("lag1_eq_assets"),
        _lag("npl_ratio", 1).alias("lag1_npl_ratio"),
        _lag("roa", 1).alias("lag1_roa"),
        _rank_pct("eq_assets").alias("rk_eq_assets"),
        _rank_pct("npl_ratio").alias("rk_npl_ratio"),
        _rank_pct("roa").alias("rk_roa"),
        _rank_pct("texas_ratio").alias("rk_texas_ratio"),
        _rank_pct("liquid").alias("rk_liquid"),
        _rank_pct("loans_dep").alias("rk_loans_dep"),
    )
    return s2.select(["CERT", "date", "qi"] + FEATURES)


# --------------------------------------------------------------------------- preprocessing (fit on TRAIN only)
class AddMissingFlags(BaseEstimator, TransformerMixin):
    """Adds a 0/1 'miss_<col>' column for every column that is missing in at least `min_rate` of the TRAINING rows."""
    def __init__(self, min_rate: float = 0.01):
        self.min_rate = min_rate

    def fit(self, X, y=None):
        self.cols_ = [c for c in X.columns if X[c].isna().mean() >= self.min_rate]
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols_:
            X["miss_" + c] = X[c].isna().astype("float64")
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips each column at its training quantiles (None = no clipping)."""
    def __init__(self, q: float | None = 0.001):
        self.q = q

    def fit(self, X, y=None):
        if self.q is None:
            self.lo_, self.hi_ = None, None
        else:
            self.lo_ = X.quantile(self.q)
            self.hi_ = X.quantile(1 - self.q)
        return self

    def transform(self, X):
        return X if self.lo_ is None else X.clip(lower=self.lo_, upper=self.hi_, axis=1)


class MedianImputer(BaseEstimator, TransformerMixin):
    """Replaces missing values by the TRAINING median of the column."""
    def fit(self, X, y=None):
        self.medians_ = X.median().fillna(0.0)
        return self

    def transform(self, X):
        return X.fillna(self.medians_)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drops a column if its |Pearson correlation| with an earlier kept column exceeds `threshold` (TRAINING sample)."""
    def __init__(self, threshold: float | None = 0.95, sample: int = 200_000, seed: int = 42):
        self.threshold, self.sample, self.seed = threshold, sample, seed

    def fit(self, X, y=None):
        self.dropped_ = {}
        self.keep_ = list(X.columns)
        if self.threshold is None:
            return self
        Xs = X.sample(min(len(X), self.sample), random_state=self.seed)
        corr = Xs.corr().abs().fillna(0.0)
        keep: list[str] = []
        for col in X.columns:
            partner = next((k for k in keep if corr.loc[col, k] > self.threshold), None)
            if partner is None:
                keep.append(col)
            else:
                self.dropped_[col] = partner
        self.keep_ = keep
        return self

    def transform(self, X):
        return X[self.keep_]


class Standardizer(BaseEstimator, TransformerMixin):
    """(x - training mean) / training std."""
    def fit(self, X, y=None):
        self.mean_ = X.mean()
        self.std_ = X.std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


def make_preprocessor(winsor_q: float | None = 0.001, corr_threshold: float | None = 0.95,
                      min_missing_rate: float = 0.01) -> Pipeline:
    """Order: missing flags -> winsorise -> median-impute -> drop highly correlated -> standardise."""
    return Pipeline([("flags", AddMissingFlags(min_missing_rate)), ("winsor", Winsorizer(winsor_q)),
                     ("impute", MedianImputer()), ("corr", CorrelationFilter(corr_threshold)),
                     ("scale", Standardizer())])