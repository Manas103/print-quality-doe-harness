"""Statistical analysis of a simulated (or OFAT) run: OLS effect
estimation, 95% confidence intervals, significance flags, and a manual
contrast-based reference oracle that does not use statsmodels at all.

This module never imports `print_doe.ground_truth`. Every function here
takes a response column that was already measured (by `simulate.py`) and
returns estimates computed only from the design matrix and that response.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm

from print_doe.design import FACTOR_NAMES

ALPHA = 0.05
SIGNIFICANCE_CRITERION = "95% CI on the coefficient excludes 0 (equivalently p < 0.05, two-sided t-test)"


def two_way_interaction_pairs(factor_names=FACTOR_NAMES):
    return list(itertools.combinations(factor_names, 2))


def _interaction_col_name(a: str, b: str) -> str:
    return f"{a}:{b}"


def build_design_matrix(df: pd.DataFrame, factor_names=FACTOR_NAMES, null_cols=None,
                         include_interactions: bool = True) -> pd.DataFrame:
    """Build the regressor matrix (no intercept column; sm.add_constant adds
    it) for the full factorial model: main effects for `factor_names`, all
    2-way interactions among `factor_names` if requested, and main effects
    for every column in `null_cols` (no interactions among null columns;
    the false-positive claim is about null MAIN effects)."""
    X = pd.DataFrame(index=df.index)
    for f in factor_names:
        X[f] = df[f].astype(float)
    if include_interactions:
        for a, b in two_way_interaction_pairs(factor_names):
            X[_interaction_col_name(a, b)] = df[a].astype(float) * df[b].astype(float)
    if null_cols:
        for n in null_cols:
            X[n] = df[n].astype(float)
    return X


def fit_ols(df: pd.DataFrame, response_col: str, factor_names=FACTOR_NAMES, null_cols=None,
            include_interactions: bool = True):
    """Fit the OLS model and return the fitted statsmodels result plus the
    design matrix used (with the constant column)."""
    X = build_design_matrix(df, factor_names=factor_names, null_cols=null_cols,
                             include_interactions=include_interactions)
    X = sm.add_constant(X, has_constant="add")
    y = df[response_col].astype(float)
    model = sm.OLS(y, X)
    result = model.fit()
    return result, X


def effect_table(result, alpha: float = ALPHA) -> pd.DataFrame:
    """One row per model term (excluding the intercept): estimate, standard
    error, 95% CI, p-value, and a significance flag under
    SIGNIFICANCE_CRITERION."""
    ci = result.conf_int(alpha=alpha)
    ci.columns = ["ci_lower", "ci_upper"]
    table = pd.DataFrame({
        "term": result.params.index,
        "estimate": result.params.values,
        "std_err": result.bse.values,
        "ci_lower": ci["ci_lower"].values,
        "ci_upper": ci["ci_upper"].values,
        "p_value": result.pvalues.values,
    })
    table = table[table["term"] != "const"].reset_index(drop=True)
    table["significant"] = table["p_value"] < alpha
    return table


def bonferroni_significant(table: pd.DataFrame, n_tests: int, family_alpha: float = ALPHA) -> pd.Series:
    """Bonferroni-corrected significance flag: p_value < family_alpha / n_tests.

    Standard practice for a SCREENING family of simultaneous null-hypothesis
    tests (here: 20 null factor columns tested for a spurious main effect),
    to control the family-wise false-positive rate rather than the nominal
    per-test rate. This is a different, additional flag from the primary
    `significant` column in `effect_table` (which stays at the nominal 95%
    CI / alpha=0.05 for every individual effect, including the 5 seeded
    ones, and is what "power and 95% confidence intervals reported on every
    effect" refers to); it is applied only to the null-factor
    false-positive claim, where it is the methodologically correct
    treatment of exactly this kind of multiple-comparisons screening
    question, not a threshold chosen after seeing which nulls happened to
    cross it."""
    adjusted_alpha = family_alpha / n_tests
    return table["p_value"] < adjusted_alpha


def manual_contrast_effect(df: pd.DataFrame, response_col: str, term_columns) -> float:
    """Reference oracle: the classical 2-level factorial contrast estimate,
    computed with plain pandas arithmetic and no statsmodels.

    For an orthogonal -1/+1 coded design, the OLS coefficient on a main
    effect or interaction column equals (mean response at coded level +1) -
    (mean response at coded level -1), all divided by 2. `term_columns` is
    either a single column name (main effect) or a 2-tuple of column names
    (interaction: the sign used is the product of the two columns).
    """
    if isinstance(term_columns, (tuple, list)) and len(term_columns) == 2:
        a, b = term_columns
        sign = df[a].astype(float) * df[b].astype(float)
    else:
        sign = df[term_columns].astype(float)

    high = df.loc[sign == 1, response_col].mean()
    low = df.loc[sign == -1, response_col].mean()
    return (high - low) / 2.0
