"""
tests/test_pipeline.py — Tests for key functions in the pipeline.

Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import re
import sys
import os


# =============================================
# Helper functions (copied from scripts so tests are self-contained)
# =============================================

def parse_age(age_str):
    """Convert '18 Years' or '6 Months' to a float in years."""
    if pd.isna(age_str) or str(age_str).strip().upper() == "N/A":
        return np.nan
    match = re.match(r"(\d+)\s*(year|month|week|day)", str(age_str).strip().lower())
    if not match:
        return np.nan
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "year": return float(value)
    elif unit == "month": return value / 12.0
    elif unit == "week": return value / 52.0
    elif unit == "day": return value / 365.0
    return np.nan


def count_inclusion_exclusion(text):
    """Count inclusion and exclusion criteria from eligibility text."""
    if pd.isna(text) or not str(text).strip() or str(text).strip() == "nan":
        return 0, 0
    text = str(text)
    text_lower = text.lower()
    incl_pos = text_lower.find("inclusion")
    excl_pos = text_lower.find("exclusion")
    if incl_pos >= 0 and excl_pos >= 0:
        inclusion_text = text[incl_pos:excl_pos]
        exclusion_text = text[excl_pos:]
    elif incl_pos >= 0:
        inclusion_text = text[incl_pos:]
        exclusion_text = ""
    elif excl_pos >= 0:
        inclusion_text = text[:excl_pos]
        exclusion_text = text[excl_pos:]
    else:
        inclusion_text = text
        exclusion_text = ""

    def count_items(section):
        tilde_items = re.findall(r"~\*", section)
        if len(tilde_items) > 0:
            return len(tilde_items)
        items = re.findall(r"(?:^|\n)\s*(?:[-*•]|\d+[.):]).*", section)
        if len(items) > 0:
            return len(items)
        lines = [l for l in section.split("\n") if l.strip() and len(l.strip()) > 10]
        return max(len(lines) - 1, 0)

    try:
        return count_items(inclusion_text), count_items(exclusion_text)
    except:
        return 0, 0


# =============================================
# Tests for parse_age
# =============================================
class TestParseAge:
    def test_years(self):
        assert parse_age("18 Years") == 18.0

    def test_months(self):
        assert abs(parse_age("6 Months") - 0.5) < 0.01

    def test_days(self):
        assert abs(parse_age("30 Days") - 30/365.0) < 0.01

    def test_weeks(self):
        assert abs(parse_age("4 Weeks") - 4/52.0) < 0.01

    def test_na_string(self):
        assert np.isnan(parse_age("N/A"))

    def test_none(self):
        assert np.isnan(parse_age(None))

    def test_empty(self):
        assert np.isnan(parse_age(""))

    def test_garbage(self):
        assert np.isnan(parse_age("not a number"))


# =============================================
# Tests for count_inclusion_exclusion
# =============================================
class TestCountCriteria:
    def test_aact_format(self):
        """Test AACT's ~* bullet format."""
        text = "Inclusion Criteria:~* Age 18+~* Has diagnosis~Exclusion Criteria:~* Pregnant~* Prior treatment"
        incl, excl = count_inclusion_exclusion(text)
        assert incl == 2
        assert excl == 2

    def test_empty_text(self):
        incl, excl = count_inclusion_exclusion("")
        assert incl == 0
        assert excl == 0

    def test_none(self):
        incl, excl = count_inclusion_exclusion(None)
        assert incl == 0
        assert excl == 0

    def test_nan(self):
        incl, excl = count_inclusion_exclusion(float("nan"))
        assert incl == 0
        assert excl == 0

    def test_inclusion_only(self):
        text = "Inclusion Criteria:~* Age 18+~* Has diagnosis~* Willing to consent"
        incl, excl = count_inclusion_exclusion(text)
        assert incl == 3
        assert excl == 0

    def test_standard_bullets(self):
        text = "Inclusion Criteria:\n- Age 18+\n- Has diagnosis\nExclusion Criteria:\n- Pregnant"
        incl, excl = count_inclusion_exclusion(text)
        assert incl >= 1
        assert excl >= 1


# =============================================
# Tests for data file existence and structure
# =============================================
class TestDataFiles:
    def test_raw_trials_exists(self):
        assert os.path.exists("raw_trials.csv"), "raw_trials.csv not found. Run load_data.py first."

    def test_raw_trials_has_required_columns(self):
        df = pd.read_csv("raw_trials.csv", nrows=5)
        required = ["nct_id", "enrollment_actual", "phase", "eligibility_criteria", "num_sites"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_cleaned_trials_exists(self):
        assert os.path.exists("cleaned_trials.csv"), "cleaned_trials.csv not found. Run clean_data.py first."

    def test_cleaned_no_duplicates(self):
        df = pd.read_csv("cleaned_trials.csv")
        assert df["nct_id"].duplicated().sum() == 0, "Duplicates found in cleaned data"

    def test_cleaned_no_zero_enrollment(self):
        df = pd.read_csv("cleaned_trials.csv")
        assert (df["enrollment_actual"] > 0).all(), "Zero enrollment found in cleaned data"

    def test_model_ready_exists(self):
        assert os.path.exists("model_ready.csv"), "model_ready.csv not found. Run feature_engineering.py first."

    def test_model_ready_has_target(self):
        df = pd.read_csv("model_ready.csv", nrows=5)
        assert "enrollment_success" in df.columns, "Target variable missing"

    def test_model_ready_target_is_binary(self):
        df = pd.read_csv("model_ready.csv")
        assert set(df["enrollment_success"].unique()).issubset({0, 1}), "Target should be 0 or 1"

    def test_model_ready_no_nulls_in_features(self):
        df = pd.read_csv("model_ready.csv")
        feature_cols = [c for c in df.columns if c not in
                       ["nct_id", "brief_title", "phase", "enrollment_actual",
                        "lead_sponsor_class", "conditions", "enrollment_success"]]
        nulls = df[feature_cols].isnull().sum().sum()
        assert nulls == 0, f"Found {nulls} null values in features"
