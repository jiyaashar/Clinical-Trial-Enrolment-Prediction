"""
feature_engineering.py — Turn cleaned trial data into model-ready features.

Takes cleaned_trials.csv and:
  1. Parses eligibility criteria text to count inclusion/exclusion rules
  2. Extracts protocol complexity metrics (arms, duration, sites)
  3. Parses age limits from strings into numbers
  4. Encodes categorical variables (phase, sponsor type)
  5. Creates target variable: enrollment_success (1 = success, 0 = failure)
  6. Saves model_ready.csv

Usage:
    python feature_engineering.py
"""

import pandas as pd
import numpy as np
import re

df = pd.read_csv("cleaned_trials.csv")
print(f"Loaded {len(df)} trials from cleaned_trials.csv\n")



# STEP 1: Parse eligibility criteria text
# The eligibility_criteria column is a big block of text like:
#   "Inclusion Criteria:
#    - Must be 18 years or older
#    - Diagnosed with X
#    Exclusion Criteria:
#    - Pregnant
#    - Prior treatment with Y"
#
# We count how many inclusion and exclusion rules there are.
# More rules = more restrictive trial = harder to enroll patients.

print("Step 1: Parsing eligibility criteria text...")

def count_inclusion_exclusion(text):
    """Count inclusion and exclusion criteria from eligibility text."""
    if pd.isna(text) or not str(text).strip() or str(text).strip() == "nan":
        return 0, 0

    text = str(text)
    text_lower = text.lower()

    # Find where "inclusion" and "exclusion" sections start
    incl_pos = text_lower.find("inclusion")
    excl_pos = text_lower.find("exclusion")

    # Split text into inclusion and exclusion sections
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
        """Count bullet points, numbered items, or lines that look like criteria."""
        # AACT uses ~* as bullet separators (e.g. "~* Must be 18 years")
        tilde_items = re.findall(r"~\*", section)
        if len(tilde_items) > 0:
            return len(tilde_items)
        # Also check for regular bullet points / numbered items
        items = re.findall(r"(?:^|\n)\s*(?:[-*•]|\d+[.):]).*", section)
        if len(items) > 0:
            return len(items)
        # Fallback: count non-empty lines longer than 10 chars
        lines = [l for l in section.split("\n") if l.strip() and len(l.strip()) > 10]
        return max(len(lines) - 1, 0)

    try:
        return count_items(inclusion_text), count_items(exclusion_text)
    except:
        return 0, 0

# Apply to every row
criteria = df["eligibility_criteria"].apply(count_inclusion_exclusion)
df["num_inclusion_criteria"] = criteria.apply(lambda x: x[0])
df["num_exclusion_criteria"] = criteria.apply(lambda x: x[1])
df["total_criteria"] = df["num_inclusion_criteria"] + df["num_exclusion_criteria"]

# Also get the raw text length as a proxy for protocol complexity
df["criteria_text_length"] = df["eligibility_criteria"].str.len()

print(f"  Inclusion criteria: median={df['num_inclusion_criteria'].median():.0f}, "
      f"mean={df['num_inclusion_criteria'].mean():.1f}")
print(f"  Exclusion criteria: median={df['num_exclusion_criteria'].median():.0f}, "
      f"mean={df['num_exclusion_criteria'].mean():.1f}")
print(f"  Total criteria:     median={df['total_criteria'].median():.0f}, "
      f"mean={df['total_criteria'].mean():.1f}")



# STEP 2: Extract protocol complexity metrics

# These are already numeric columns from the cleaned data.
# We just need to compute a couple derived features.

print("\nStep 2: Extracting protocol complexity metrics...")

# Study duration (already computed during cleaning, but let's make sure)
df["start_date"] = pd.to_datetime(df["start_date"], format="mixed", errors="coerce")
df["completion_date"] = pd.to_datetime(df["completion_date"], format="mixed", errors="coerce")
df["duration_days"] = (df["completion_date"] - df["start_date"]).dt.days

# Enrollment per site — how many patients each site recruited on average
# If num_sites is 0, we just use total enrollment
df["enrollment_per_site"] = np.where(
    df["num_sites"] > 0,
    df["enrollment_actual"] / df["num_sites"],
    df["enrollment_actual"]
)

# Whether the trial has multiple treatment arms (e.g. drug vs placebo)
df["has_multiple_arms"] = (df["num_arms"] > 1).astype(int)

print(f"  Duration: median={df['duration_days'].median():.0f} days")
print(f"  Enrollment per site: median={df['enrollment_per_site'].median():.1f}")
print(f"  Multi-arm trials: {df['has_multiple_arms'].sum()} ({df['has_multiple_arms'].mean():.1%})")



# STEP 3: Parse age limits

# Age fields look like "18 Years", "6 Months", "N/A"
# We convert them to numbers (in years) for modeling.

print("\nStep 3: Parsing age limits...")

def parse_age(age_str):
    """Convert '18 Years' or '6 Months' to a float in years."""
    if pd.isna(age_str) or str(age_str).strip().upper() == "N/A":
        return np.nan
    match = re.match(r"(\d+)\s*(year|month|week|day)", str(age_str).strip().lower())
    if not match:
        return np.nan
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "year":
        return float(value)
    elif unit == "month":
        return value / 12.0
    elif unit == "week":
        return value / 52.0
    elif unit == "day":
        return value / 365.0
    return np.nan

df["min_age_years"] = df["minimum_age"].apply(parse_age)
df["max_age_years"] = df["maximum_age"].apply(parse_age)
df["age_range_years"] = df["max_age_years"] - df["min_age_years"]

# Fill missing ages with median (many trials don't set an upper age limit)
median_min = df["min_age_years"].median()
median_max = df["max_age_years"].median()
df["min_age_years"] = df["min_age_years"].fillna(median_min)
df["max_age_years"] = df["max_age_years"].fillna(median_max)
df["age_range_years"] = df["age_range_years"].fillna(median_max - median_min)

print(f"  Min age: median={df['min_age_years'].median():.0f} years")
print(f"  Max age: median={df['max_age_years'].median():.0f} years")
print(f"  Age range: median={df['age_range_years'].median():.0f} years")



# STEP 4: Encode categorical variables
# Models need numbers, not strings. We convert categories to binary columns.

print("\nStep 4: Encoding categorical variables...")

# Phase — binary indicators
df["is_phase2"] = df["phase"].str.contains("PHASE2", na=False).astype(int)
df["is_phase3"] = df["phase"].str.contains("PHASE3", na=False).astype(int)

# Sponsor type — binary: industry vs non-industry
df["is_industry_sponsored"] = (df["lead_sponsor_class"] == "INDUSTRY").astype(int)

# Sex restriction — binary: restricted to one sex vs open to all
df["sex_restricted"] = (df["sex"] != "ALL").astype(int)

# Therapeutic area — extract the most common conditions as binary features
# This lets the model know if a trial is about cancer, diabetes, etc.
top_conditions = ["cancer", "diabetes", "hiv", "asthma", "pain",
                  "hypertension", "depression", "schizophrenia"]

for condition in top_conditions:
    col_name = f"is_{condition}"
    df[col_name] = df["conditions"].str.lower().str.contains(condition, na=False).astype(int)

print(f"  Phase indicators: is_phase2, is_phase3")
print(f"  Sponsor: is_industry_sponsored ({df['is_industry_sponsored'].mean():.1%} are industry)")
print(f"  Sex restricted: {df['sex_restricted'].mean():.1%}")
print(f"  Condition flags: {', '.join('is_' + c for c in top_conditions)}")



# STEP 5: Create target variable
# Our goal: predict whether a trial "successfully" enrolled enough patients.
# Problem: ClinicalTrials.gov doesn't give us a "target enrollment" number
# separate from actual enrollment for most completed trials.
# Solution: We compare each trial's enrollment to the median enrollment
# for trials in the same phase + sponsor group. If a trial enrolled at
# least 80% of its group's median, we call it a success.
#
# This is a proxy — not perfect, but it gives us a reasonable binary target.

print("\nStep 5: Creating target variable...")

THRESHOLD = 0.80

# Group median: what's the "expected" enrollment for this type of trial?
df["group_median"] = df.groupby(["phase", "is_industry_sponsored"])["enrollment_actual"].transform("median")

# Enrollment ratio: actual / expected
df["enrollment_ratio"] = df["enrollment_actual"] / df["group_median"]

# Binary target: 1 = success (ratio >= 0.80), 0 = failure
df["enrollment_success"] = (df["enrollment_ratio"] >= THRESHOLD).astype(int)

success_rate = df["enrollment_success"].mean()
n_success = df["enrollment_success"].sum()
n_failure = len(df) - n_success

print(f"  Threshold: {THRESHOLD} (trial must reach 80% of group median)")
print(f"  Success rate: {success_rate:.1%}")
print(f"  Successes: {n_success}")
print(f"  Failures: {n_failure}")
print(f"  Class balance: {'balanced' if 0.35 < success_rate < 0.65 else 'IMBALANCED — may need to address this'}")


# STEP 6: Select final features and save
print("\nStep 6: Selecting features and saving...")

# These are the columns the model will use as input (X)
feature_cols = [
    # Protocol complexity
    "duration_days",
    "num_sites",
    "num_countries",
    "num_arms",
    "num_interventions",
    "enrollment_per_site",
    "has_multiple_arms",
    # Eligibility criteria
    "num_inclusion_criteria",
    "num_exclusion_criteria",
    "total_criteria",
    "criteria_text_length",
    # Demographics
    "min_age_years",
    "max_age_years",
    "age_range_years",
    "sex_restricted",
    # Phase
    "is_phase2",
    "is_phase3",
    # Sponsor
    "is_industry_sponsored",
    # Therapeutic area
    "is_cancer",
    "is_diabetes",
    "is_hiv",
    "is_asthma",
    "is_pain",
    "is_hypertension",
    "is_depression",
    "is_schizophrenia",
]

# Target variable
target_col = "enrollment_success"

# Columns to keep for reference (not used in model, but useful to have)
id_cols = ["nct_id", "brief_title", "phase", "enrollment_actual",
           "lead_sponsor_class", "conditions"]

# Build final dataframe
keep_cols = id_cols + feature_cols + [target_col]
df_final = df[keep_cols].copy()

# Make sure no NaN in features (fill any stragglers with 0)
for col in feature_cols:
    if df_final[col].isna().any():
        df_final[col] = df_final[col].fillna(0)

df_final.to_csv("model_ready.csv", index=False)

print(f"\n  Total features: {len(feature_cols)}")
print(f"  Total trials: {len(df_final)}")
print(f"  Saved to: model_ready.csv")

# Print a summary of all features
print(f"\n--- Feature Summary ---")
print(f"{'Feature':<30} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10}")
print("-" * 70)
for col in feature_cols:
    print(f"{col:<30} {df_final[col].mean():>10.1f} {df_final[col].median():>10.1f} "
          f"{df_final[col].min():>10.1f} {df_final[col].max():>10.1f}")

print(f"\n{'Target: enrollment_success':<30} {df_final[target_col].mean():>10.1%}")
