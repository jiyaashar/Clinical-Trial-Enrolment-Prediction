"""
explore_data.py — Initial data exploration, summary stats, and visualizations.

This script takes raw_trials.csv and:
  1. Prints summary statistics about the dataset
  2. Creates visualizations showing enrollment patterns
  3. Runs K-Means clustering to find natural groupings of trials

Usage:
    python explore_data.py

Output:
    Prints stats to terminal + saves plots as .png files in the current folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re

# =============================================
# SETUP
# =============================================
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 150, "font.size": 12})

df = pd.read_csv("raw_trials.csv")
print(f"Loaded {len(df)} trials from raw_trials.csv\n")


# =============================================
# STEP 1: Basic summary statistics
# =============================================
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total trials: {len(df)}")
print(f"Date range: {df['start_date'].dropna().min()} to {df['start_date'].dropna().max()}")
print(f"Columns: {len(df.columns)}")

print(f"\n--- Enrollment ---")
print(f"  Mean:   {df['enrollment_actual'].mean():.0f}")
print(f"  Median: {df['enrollment_actual'].median():.0f}")
print(f"  Min:    {df['enrollment_actual'].min():.0f}")
print(f"  Max:    {df['enrollment_actual'].max():.0f}")

print(f"\n--- Phase distribution ---")
print(df["phase"].value_counts().to_string())

print(f"\n--- Sponsor type ---")
print(df["lead_sponsor_class"].value_counts().to_string())

print(f"\n--- Sex eligibility ---")
print(df["sex"].value_counts().to_string())

print(f"\n--- Top 10 conditions ---")
# Each trial can have multiple conditions; split and count
all_conditions = df["conditions"].dropna().str.split(", ").explode()
print(all_conditions.value_counts().head(10).to_string())

print(f"\n--- Sites per trial ---")
print(f"  Mean:   {df['num_sites'].mean():.1f}")
print(f"  Median: {df['num_sites'].median():.0f}")
print(f"  Trials with 0 sites listed: {(df['num_sites'] == 0).sum()}")

print(f"\n--- Missing values ---")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print(missing.to_string())
else:
    print("  None!")


# =============================================
# STEP 2: Parse some fields for plotting
# =============================================

# Parse age strings like "18 Years" into numbers
def parse_age(age_str):
    if pd.isna(age_str) or age_str == "N/A":
        return np.nan
    match = re.match(r"(\d+)\s*(year|month)", str(age_str).lower())
    if not match:
        return np.nan
    val = int(match.group(1))
    if match.group(2) == "month":
        return val / 12.0
    return float(val)

df["min_age_years"] = df["minimum_age"].apply(parse_age)
df["max_age_years"] = df["maximum_age"].apply(parse_age)

# Count eligibility criteria (split by bullet points / newlines)
def count_criteria(text):
    if pd.isna(text) or not str(text).strip():
        return 0
    # count lines that look like criteria (start with dash, bullet, or number)
    items = re.findall(r"(?:^|\n)\s*(?:[-*•]|\d+[.):]).*", str(text))
    if len(items) == 0:
        # fallback: count non-empty lines
        lines = [l for l in str(text).split("\n") if l.strip() and len(l.strip()) > 10]
        return len(lines)
    return len(items)

df["num_criteria"] = df["eligibility_criteria"].apply(count_criteria)

# Study duration in days
df["start_date"] = pd.to_datetime(df["start_date"], format="mixed", errors="coerce")
df["completion_date"] = pd.to_datetime(df["completion_date"], format="mixed", errors="coerce")
df["duration_days"] = (df["completion_date"] - df["start_date"]).dt.days


# =============================================
# PLOT 1: Distribution of enrollment
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["enrollment_actual"].clip(upper=2000), bins=50,
             edgecolor="black", alpha=0.7, color="#2196F3")
axes[0].set_xlabel("Actual Enrollment")
axes[0].set_ylabel("Number of Trials")
axes[0].set_title("Distribution of Trial Enrollment (capped at 2000)")
axes[0].axvline(df["enrollment_actual"].median(), color="red", linestyle="--",
                label=f'Median: {df["enrollment_actual"].median():.0f}')
axes[0].legend()

# Log scale — enrollment is heavily right-skewed so log makes patterns clearer
axes[1].hist(np.log10(df["enrollment_actual"].clip(lower=1)), bins=50,
             edgecolor="black", alpha=0.7, color="#4CAF50")
axes[1].set_xlabel("Log10(Enrollment)")
axes[1].set_ylabel("Number of Trials")
axes[1].set_title("Enrollment Distribution (Log Scale)")

plt.tight_layout()
plt.savefig("plot1_enrollment_distribution.png", bbox_inches="tight")
plt.close()
print("\nSaved: plot1_enrollment_distribution.png")


# =============================================
# PLOT 2: Enrollment by phase
# =============================================
fig, ax = plt.subplots(figsize=(10, 6))
phase_order = ["PHASE2", "PHASE2/PHASE3", "PHASE3"]
df_phase = df[df["phase"].isin(phase_order)]

sns.boxplot(data=df_phase, x="phase", y="enrollment_actual", order=phase_order,
            hue="phase", palette="Set2", legend=False, ax=ax)
ax.set_ylim(0, df_phase["enrollment_actual"].quantile(0.95))  # cap y-axis at 95th pctile
ax.set_xlabel("Trial Phase")
ax.set_ylabel("Actual Enrollment")
ax.set_title("Enrollment by Trial Phase")

# Add sample counts
for i, phase in enumerate(phase_order):
    n = (df_phase["phase"] == phase).sum()
    ax.text(i, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("plot2_enrollment_by_phase.png", bbox_inches="tight")
plt.close()
print("Saved: plot2_enrollment_by_phase.png")


# =============================================
# PLOT 3: Enrollment by sponsor type
# =============================================
fig, ax = plt.subplots(figsize=(10, 6))
sponsor_stats = df.groupby("lead_sponsor_class")["enrollment_actual"].agg(["median", "count"])
sponsor_stats = sponsor_stats[sponsor_stats["count"] >= 20]  # only groups with enough data
sponsor_stats = sponsor_stats.sort_values("median", ascending=True)

ax.barh(sponsor_stats.index, sponsor_stats["median"],
        color="#2196F3", edgecolor="black", alpha=0.8)
for i, (idx, row) in enumerate(sponsor_stats.iterrows()):
    ax.text(row["median"] + 5, i, f'n={int(row["count"])}', va="center", fontsize=10)

ax.set_xlabel("Median Enrollment")
ax.set_ylabel("Sponsor Type")
ax.set_title("Median Enrollment by Sponsor Type")
plt.tight_layout()
plt.savefig("plot3_enrollment_by_sponsor.png", bbox_inches="tight")
plt.close()
print("Saved: plot3_enrollment_by_sponsor.png")


# =============================================
# PLOT 4: Sites vs enrollment (scatter)
# =============================================
fig, ax = plt.subplots(figsize=(10, 6))
df_sites = df[df["num_sites"] > 0]  # only trials with site data

ax.scatter(df_sites["num_sites"], df_sites["enrollment_actual"],
           alpha=0.3, s=10, color="#2196F3")
ax.set_xlabel("Number of Sites")
ax.set_ylabel("Actual Enrollment")
ax.set_title("Number of Sites vs. Enrollment")
ax.set_xscale("log")
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("plot4_sites_vs_enrollment.png", bbox_inches="tight")
plt.close()
print("Saved: plot4_sites_vs_enrollment.png")


# =============================================
# PLOT 5: Top conditions bar chart
# =============================================
fig, ax = plt.subplots(figsize=(10, 6))
top_cond = all_conditions.value_counts().head(15)
top_cond = top_cond.sort_values(ascending=True)  # horizontal bars read bottom-up

ax.barh(top_cond.index, top_cond.values, color="#FF9800", edgecolor="black", alpha=0.8)
ax.set_xlabel("Number of Trials")
ax.set_title("Top 15 Conditions Studied")
plt.tight_layout()
plt.savefig("plot5_top_conditions.png", bbox_inches="tight")
plt.close()
print("Saved: plot5_top_conditions.png")


# =============================================
# PLOT 6: Correlation heatmap
# =============================================
numeric_cols = ["enrollment_actual", "num_sites", "num_countries", "num_arms",
                "num_interventions", "num_criteria", "duration_days",
                "min_age_years", "max_age_years"]
available = [c for c in numeric_cols if c in df.columns]
corr = df[available].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plot6_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved: plot6_correlation_heatmap.png")


# =============================================
# PLOT 7: K-Means clustering
# =============================================
# We cluster trials by their numeric characteristics to find natural groupings.
# This uses concepts from the clustering unit (K-Means, distance/similarity).

print("\nRunning K-Means clustering...")

# Pick features for clustering
cluster_features = ["enrollment_actual", "num_sites", "num_arms", "duration_days", "num_criteria"]
df_cluster = df[cluster_features].dropna()  # drop rows with missing values

# Standardize — K-Means uses distance, so features need to be on same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# Run K-Means with k=3 (small, medium, large trials)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster["cluster"] = kmeans.fit_predict(X_scaled)

# Print cluster summaries
print(f"\nCluster sizes:")
print(df_cluster["cluster"].value_counts().sort_index().to_string())
print(f"\nCluster means:")
print(df_cluster.groupby("cluster")[cluster_features].mean().round(1).to_string())

# Scatter plot: enrollment vs num_sites, colored by cluster
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["#F44336", "#2196F3", "#4CAF50"]

for c in sorted(df_cluster["cluster"].unique()):
    subset = df_cluster[df_cluster["cluster"] == c]
    axes[0].scatter(subset["num_sites"], subset["enrollment_actual"],
                    alpha=0.4, s=10, color=colors[c], label=f"Cluster {c}")

axes[0].set_xlabel("Number of Sites")
axes[0].set_ylabel("Actual Enrollment")
axes[0].set_title("K-Means Clusters: Sites vs Enrollment")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].legend()

# Scatter plot: duration vs enrollment, colored by cluster
for c in sorted(df_cluster["cluster"].unique()):
    subset = df_cluster[df_cluster["cluster"] == c]
    axes[1].scatter(subset["duration_days"], subset["enrollment_actual"],
                    alpha=0.4, s=10, color=colors[c], label=f"Cluster {c}")

axes[1].set_xlabel("Study Duration (days)")
axes[1].set_ylabel("Actual Enrollment")
axes[1].set_title("K-Means Clusters: Duration vs Enrollment")
axes[1].set_yscale("log")
axes[1].legend()

plt.tight_layout()
plt.savefig("plot7_kmeans_clusters.png", bbox_inches="tight")
plt.close()
print("Saved: plot7_kmeans_clusters.png")


# =============================================
# PLOT 8: Elbow method (finding optimal k)
# =============================================
# The elbow method helps us pick the best number of clusters.
# We run K-Means for k=2 through k=8 and plot the inertia (within-cluster variance).
# The "elbow" in the curve suggests the best k.

inertias = []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_range, inertias, "o-", color="#2196F3", linewidth=2, markersize=8)
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia (within-cluster variance)")
ax.set_title("Elbow Method for Optimal k")
ax.set_xticks(list(k_range))
plt.tight_layout()
plt.savefig("plot8_elbow_method.png", bbox_inches="tight")
plt.close()
print("Saved: plot8_elbow_method.png")


# =============================================
# DONE
# =============================================
print("\n" + "=" * 60)
print("All plots saved! Check your folder for plot1 through plot8.")
print("=" * 60)
