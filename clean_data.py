"""
clean_data.py — Clean raw trial data and prepare it for modeling.

Takes raw_trials.csv and:
  1. Removes duplicate trials
  2. Handles missing values
  3. Standardizes and validates dates
  4. Removes outliers
  5. Documents data quality issues
  6. Saves cleaned_trials.csv

Usage:
    python clean_data.py
"""

import pandas as pd
import numpy as np

df = pd.read_csv("raw_trials.csv")
print(f"Loaded {len(df)} trials from raw_trials.csv\n")



# DATA QUALITY REPORT (before cleaning)
print("=" * 60)
print("DATA QUALITY REPORT (before cleaning)")
print("=" * 60)

# Check missing values in every column
print("\nMissing values per column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
quality = pd.DataFrame({"missing": missing, "pct": missing_pct})
quality = quality[quality["missing"] > 0].sort_values("missing", ascending=False)
if len(quality) > 0:
    print(quality.to_string())
else:
    print("  No missing values!")

# Check duplicates
dupes = df["nct_id"].duplicated().sum()
print(f"\n Duplicate trial IDs: {dupes} ")

# Check enrollment issues
zero_enrollment = (df["enrollment_actual"] == 0).sum()
negative_enrollment = (df["enrollment_actual"] < 0).sum()
print(f"\n Enrollment issues :")
print(f"  Zero enrollment: {zero_enrollment}")
print(f"  Negative enrollment: {negative_enrollment}")

# Check date issues
print(f"\n Date issues - ")
print(f"  Missing start_date: {df['start_date'].isna().sum()}")
print(f"  Missing completion_date: {df['completion_date'].isna().sum()}")


# STEP 1: Remove duplicate trials
print("\n" + "=" * 60)
print("STEP 1: Remove duplicates")
print("=" * 60)

before = len(df)
df = df.drop_duplicates(subset=["nct_id"], keep="first")  # keep first occurrence
print(f"  {before} → {len(df)} ({before - len(df)} duplicates removed)")


# STEP 2: Handle missing enrollment values

print("\n" + "=" * 60)
print("STEP 2: Handle missing enrollment")
print("=" * 60)

before = len(df)

# Drop rows where enrollment is missing or zero — we can't use these
df = df[df["enrollment_actual"].notna()]
df = df[df["enrollment_actual"] > 0]

print(f"  {before} → {len(df)} ({before - len(df)} removed: missing or zero enrollment)")


# STEP 3: Standardize and validate dates
print("\n" + "=" * 60)
print("STEP 3: Standardize and validate dates")
print("=" * 60)

# Convert date strings to actual datetime objects
# format="mixed" handles different date formats (e.g. "2020-01-15" vs "January 2020")
# errors="coerce" turns unparseable dates into NaT (missing) instead of crashing
df["start_date"] = pd.to_datetime(df["start_date"], format="mixed", errors="coerce")
df["completion_date"] = pd.to_datetime(df["completion_date"], format="mixed", errors="coerce")

# Count how many dates couldn't be parsed
bad_start = df["start_date"].isna().sum()
bad_end = df["completion_date"].isna().sum()
print(f"  Unparseable start dates: {bad_start}")
print(f"  Unparseable completion dates: {bad_end}")

# Drop rows where either date is missing (we need both to calculate duration)
before = len(df)
df = df.dropna(subset=["start_date", "completion_date"])
print(f"  Dropped {before - len(df)} rows with missing dates")

# Drop rows where completion date is before start date (data error)
before = len(df)
bad_range = df["completion_date"] < df["start_date"]
print(f"  Trials where completion < start: {bad_range.sum()}")
df = df[~bad_range]
print(f"  Dropped {before - len(df)} rows with invalid date ranges")

# Calculate study duration in days
df["duration_days"] = (df["completion_date"] - df["start_date"]).dt.days

# Drop trials with unreasonably short or long durations
before = len(df)
df = df[(df["duration_days"] >= 7) & (df["duration_days"] <= 7300)]  # 1 week to 20 years
print(f"  Dropped {before - len(df)} rows with unreasonable duration (<7 days or >20 years)")

print(f"  Duration range: {df['duration_days'].min()} to {df['duration_days'].max()} days")
print(f"  Median duration: {df['duration_days'].median():.0f} days")


# STEP 4: Remove outliers
print("\n" + "=" * 60)
print("STEP 4: Remove enrollment outliers")
print("=" * 60)

# Use the 1st and 99th percentiles to define outlier bounds
lower = df["enrollment_actual"].quantile(0.01)
upper = df["enrollment_actual"].quantile(0.99)
print(f"  1st percentile: {lower:.0f}")
print(f"  99th percentile: {upper:.0f}")

before = len(df)
df = df[(df["enrollment_actual"] >= lower) & (df["enrollment_actual"] <= upper)]
print(f"  {before} → {len(df)} ({before - len(df)} outliers removed)")
print(f"  Enrollment range now: {df['enrollment_actual'].min():.0f} to {df['enrollment_actual'].max():.0f}")



# STEP 5: Standardize categorical columns
print("\n" + "=" * 60)
print("STEP 5: Standardize categorical fields")
print("=" * 60)

# Make sure phase values are consistent
df["phase"] = df["phase"].fillna("UNKNOWN").str.upper().str.strip()
print(f"  Phase values: {df['phase'].value_counts().to_dict()}")

# Standardize sponsor class
df["lead_sponsor_class"] = df["lead_sponsor_class"].fillna("OTHER").str.upper().str.strip()
print(f"  Sponsor types: {df['lead_sponsor_class'].value_counts().to_dict()}")

# Standardize sex
df["sex"] = df["sex"].fillna("All").str.upper().str.strip()
print(f"  Sex values: {df['sex'].value_counts().to_dict()}")


# STEP 6: Handle remaining missing values
print("\n" + "=" * 60)
print("STEP 6: Handle remaining missing values")
print("=" * 60)

# Fill missing text fields with empty strings
df["eligibility_criteria"] = df["eligibility_criteria"].fillna("")
df["conditions"] = df["conditions"].fillna("")
df["intervention_types"] = df["intervention_types"].fillna("")
df["why_stopped"] = df["why_stopped"].fillna("")
df["brief_title"] = df["brief_title"].fillna("")

# Fill missing numeric fields with 0 (reasonable default for counts)
df["num_sites"] = df["num_sites"].fillna(0).astype(int)
df["num_countries"] = df["num_countries"].fillna(0).astype(int)
df["num_arms"] = df["num_arms"].fillna(1).astype(int)
df["num_interventions"] = df["num_interventions"].fillna(1).astype(int)

# Fill missing age fields
df["minimum_age"] = df["minimum_age"].fillna("N/A")
df["maximum_age"] = df["maximum_age"].fillna("N/A")

# Check what's left
remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]
if len(remaining_missing) > 0:
    print("  Remaining missing values:")
    print(remaining_missing.to_string())
else:
    print("  No missing values remaining!")


# FINAL QUALITY REPORT (after cleaning)
print("\n" + "=" * 60)
print("FINAL QUALITY REPORT")
print("=" * 60)
print(f"  Trials before cleaning: 10000")
print(f"  Trials after cleaning:  {len(df)}")
print(f"  Rows removed:           {10000 - len(df)}")
print(f"\n  Enrollment: median={df['enrollment_actual'].median():.0f}, "
      f"mean={df['enrollment_actual'].mean():.0f}, "
      f"min={df['enrollment_actual'].min():.0f}, "
      f"max={df['enrollment_actual'].max():.0f}")
print(f"  Duration:   median={df['duration_days'].median():.0f} days, "
      f"mean={df['duration_days'].mean():.0f} days")
print(f"  Sites:      median={df['num_sites'].median():.0f}, "
      f"mean={df['num_sites'].mean():.1f}")
print(f"\n  Phase distribution:")
print(f"    {df['phase'].value_counts().to_string()}")


# saving to csv
df.to_csv("cleaned_trials.csv", index=False)
print(f"\nSaved {len(df)} cleaned trials to cleaned_trials.csv")
