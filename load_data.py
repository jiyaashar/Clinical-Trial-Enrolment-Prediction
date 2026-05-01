"""
load_data.py — Load and filter clinical trial data from AACT database files.

Data source: https://aact.ctti-clinicaltrials.org/downloads
We downloaded the "Flat Text Files" zip and unzipped into data/ folder.

This script reads the relevant .txt files, filters to the trials we care about,
merges them into one table, and saves raw_trials.csv for the next steps.

Usage:
    python load_data.py
"""

import pandas as pd
import os

DATA_DIR = "data"
OUTPUT = "raw_trials.csv"


# Helper to load pipe-delimited AACT files
def load(filename):
    """Each AACT file is pipe-delimited (| instead of comma). This reads one."""
    path = os.path.join(DATA_DIR, filename)
    print(f"  Loading {filename}...", end=" ")
    df = pd.read_csv(path, sep="|", low_memory=False)
    print(f"{len(df)} rows")
    return df


#1. Loading the main studies table & filter

studies = load("studies.txt")
print(f"\n  Total studies in database: {len(studies)}")

# Keep only completed trials (not recruiting, terminated, etc.)
studies = studies[studies["overall_status"] == "COMPLETED"]
print(f"  After filtering to completed: {len(studies)}")

# Keep only interventional trials (drug/device trials, not observational)
studies = studies[studies["study_type"] == "INTERVENTIONAL"]
print(f"  After filtering to interventional: {len(studies)}")

# Keep only Phase 2, Phase 3, or Phase 2/3 trials
studies = studies[studies["phase"].isin(["PHASE2", "PHASE3", "PHASE2/PHASE3"])]
print(f"  After filtering to Phase 2/3: {len(studies)}")

# Keep only trials that reported actual (not estimated) enrollment > 0
studies = studies[studies["enrollment_type"] == "ACTUAL"]
studies = studies[studies["enrollment"].notna() & (studies["enrollment"] > 0)]
print(f"  After filtering to actual enrollment: {len(studies)}")

# If there are too many, sample down to 10k for manageable size
if len(studies) > 10000:
    studies = studies.sample(n=10000, random_state=42)
    print(f"  Sampled down to: {len(studies)}")

# Set of trial IDs we're keeping — used to filter other tables
nct_ids = set(studies["nct_id"])


#Load supporting tables

# Each table has nct_id so we can join them to our studies, We only keep rows matching our filtered trial IDs.

# Eligibility criteria — the inclusion/exclusion rules for each trial
elig = load("eligibilities.txt")
elig = elig[elig["nct_id"].isin(nct_ids)]  # keep only our trials
elig = elig[["nct_id", "criteria", "gender", "minimum_age", "maximum_age"]]
elig = elig.drop_duplicates("nct_id")  # one row per trial

# Facilities — each row is one site; we count them per trial
fac = load("facilities.txt")
fac = fac[fac["nct_id"].isin(nct_ids)]
site_counts = fac.groupby("nct_id").size().reset_index(name="num_sites")  # count sites
country_counts = fac.groupby("nct_id")["country"].nunique().reset_index(name="num_countries")

# Sponsors — who funded the trial (pharma company, university, NIH, etc.)
sponsors = load("sponsors.txt")
sponsors = sponsors[(sponsors["nct_id"].isin(nct_ids)) & (sponsors["lead_or_collaborator"] == "lead")]
sponsors = sponsors.drop_duplicates("nct_id")[["nct_id", "name", "agency_class"]]

# Design groups — treatment arms (e.g. drug vs placebo)
groups = load("design_groups.txt")
groups = groups[groups["nct_id"].isin(nct_ids)]
arm_counts = groups.groupby("nct_id").size().reset_index(name="num_arms")

# Interventions — what drug/device/procedure was tested
interv = load("interventions.txt")
interv = interv[interv["nct_id"].isin(nct_ids)]
interv_counts = interv.groupby("nct_id").size().reset_index(name="num_interventions")
interv_types = interv.groupby("nct_id")["intervention_type"].apply(
    lambda x: ", ".join(x.dropna().unique())  # e.g. "DRUG, BIOLOGICAL"
).reset_index(name="intervention_types")

# Conditions — what disease the trial is about
cond = load("conditions.txt")
cond = cond[cond["nct_id"].isin(nct_ids)]
cond_text = cond.groupby("nct_id")["name"].apply(
    lambda x: ", ".join(x.dropna().unique()[:5])  # keep first 5 conditions
).reset_index(name="conditions")



#Merge everything into one dataframe

print("\nStep 3: Merging all tables together...")

# Starting with the columns we need from studies
df = studies[["nct_id", "brief_title", "overall_status", "phase",
              "enrollment", "start_date", "completion_date",
              "study_type", "why_stopped", "number_of_arms"]].copy()

df = df.rename(columns={
    "enrollment": "enrollment_actual",
    "number_of_arms": "num_arms_orig",
})

# Join each supporting table by nct_id (left join keeps all studies)
df = df.merge(elig, on="nct_id", how="left")
df = df.merge(site_counts, on="nct_id", how="left")
df = df.merge(country_counts, on="nct_id", how="left")
df = df.merge(sponsors, on="nct_id", how="left")
df = df.merge(arm_counts, on="nct_id", how="left")
df = df.merge(interv_counts, on="nct_id", how="left")
df = df.merge(interv_types, on="nct_id", how="left")
df = df.merge(cond_text, on="nct_id", how="left")

# Rename columns to cleaner names
df = df.rename(columns={
    "criteria": "eligibility_criteria",
    "gender": "sex",
    "name": "lead_sponsor_name",
    "agency_class": "lead_sponsor_class",
})

# Use counted arms if original column was missing
df["num_arms"] = df["num_arms_orig"].fillna(df["num_arms"]).fillna(1).astype(int)
df = df.drop(columns=["num_arms_orig"])



#Fill missing values and save

print("Cleaning up and saving...\n")

df["num_sites"] = df["num_sites"].fillna(0).astype(int)
df["num_countries"] = df["num_countries"].fillna(0).astype(int)
df["num_interventions"] = df["num_interventions"].fillna(1).astype(int)
df["why_stopped"] = df["why_stopped"].fillna("")
df["eligibility_criteria"] = df["eligibility_criteria"].fillna("")
df["conditions"] = df["conditions"].fillna("")
df["intervention_types"] = df["intervention_types"].fillna("")
df["lead_sponsor_class"] = df["lead_sponsor_class"].fillna("Other")
df["sex"] = df["sex"].fillna("All")
df["minimum_age"] = df["minimum_age"].fillna("N/A")
df["maximum_age"] = df["maximum_age"].fillna("N/A")

# Save final CSV
df.to_csv(OUTPUT, index=False)

print(f"Saved {len(df)} trials to {OUTPUT}")
print(f"Columns: {list(df.columns)}")
print(f"\nPreview:")
print(df[["nct_id", "phase", "enrollment_actual", "num_sites", "conditions"]].head(10))
