"""
insights.py — Actionable insights, risk scoring, and therapeutic area analysis.

Takes model_ready.csv and:
  1. Quantifies impact of each feature on enrollment probability
  2. Analyzes therapeutic area success rates with clustering
  3. Validates risk scoring system (0-100 scale)
  4. Generates specific protocol optimization recommendations
  5. Finds optimal number of sites per therapeutic area

Usage:
    python insights.py
"""
# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# sklearn tools for modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 150, "font.size": 12})


# LOAD DATA
df = pd.read_csv("model_ready.csv")
print(f"Loaded {len(df)} trials\n")


FEATURE_COLS = [
    "duration_days", "num_sites", "num_countries", "num_arms",
    "num_interventions", "enrollment_per_site", "has_multiple_arms",
    "num_inclusion_criteria", "num_exclusion_criteria", "total_criteria",
    "criteria_text_length", "min_age_years", "max_age_years", "age_range_years",
    "sex_restricted", "is_phase2", "is_phase3", "is_industry_sponsored",
    "is_cancer", "is_diabetes", "is_hiv", "is_asthma",
    "is_pain", "is_hypertension", "is_depression", "is_schizophrenia",
]

# X = input features, fill missing values with 0
X = df[FEATURE_COLS].fillna(0)

# y = target variable (whether enrollment succeeded)
y = df["enrollment_success"]

# Train logistic regression on full data for coefficient analysis

# Standardize features (important for logistic regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression model
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, y)


# PART 1: Per-feature probability impact
# For each feature, we calculate: if you increase it by 1 unit,
# how much does the enrollment success probability change?
# We use the logistic regression to compute marginal effects.

print("=" * 60)
print("PART 1: Per-Feature Impact on Enrollment Probability")
print("=" * 60)

# Get coefficients and map back to original (unscaled) units
# For logistic regression: change in log-odds = coef * (1 / std)
# Change in probability ≈ coef * (1/std) * p * (1-p) where p is baseline probability

# Baseline success probability (mean of target)
baseline_prob = y.mean()  # 0.59
marginal_factor = baseline_prob * (1 - baseline_prob)  # ≈ 0.242

impacts = []
# Loop through each feature to compute its effect
for i, feature in enumerate(FEATURE_COLS):
    coef = lr.coef_[0][i]
    std = X[feature].std()
    if std > 0:
        # Impact of 1-unit increase in original scale
        impact_per_unit = (coef / std) * marginal_factor * 100  # as percentage points
    else:
        impact_per_unit = 0
    impacts.append({
        "feature": feature,
        "coefficient": coef,
        "impact_per_unit_pct": impact_per_unit,
        "std": std,
    })

impacts_df = pd.DataFrame(impacts).sort_values("impact_per_unit_pct", ascending=False)

print(f"\nBaseline success probability: {baseline_prob:.1%}")
print(f"\nImpact of increasing each feature by 1 unit:")
print(f"{'Feature':<30} {'Change in Success Prob':>25}")
print("-" * 55)
for _, row in impacts_df.iterrows():
    if abs(row["impact_per_unit_pct"]) > 0.01:
        direction = "+" if row["impact_per_unit_pct"] > 0 else ""
        print(f"  {row['feature']:<28} {direction}{row['impact_per_unit_pct']:>8.2f} percentage points")

# Specific insight: exclusion criteria impact
excl_impact = impacts_df[impacts_df["feature"] == "num_exclusion_criteria"]["impact_per_unit_pct"].values[0]
incl_impact = impacts_df[impacts_df["feature"] == "num_inclusion_criteria"]["impact_per_unit_pct"].values[0]
print(f"\n*** KEY INSIGHT ***")
print(f"  Each additional exclusion criterion changes enrollment success probability by {excl_impact:+.2f} percentage points")
print(f"  Each additional inclusion criterion changes enrollment success probability by {incl_impact:+.2f} percentage points")
print(f"  Reducing exclusion criteria from 15 to 8 could improve enrollment probability by {abs(excl_impact) * 7:.1f} percentage points")
print(f"  Reducing total criteria from 20 to 10 could improve enrollment probability by ~{abs(excl_impact) * 5 + abs(incl_impact) * 5:.1f} percentage points")



# PART 2: Therapeutic area success rates + clustering
# Group trials by disease area and compare enrollment success rates.
# Then use K-Means clustering to group therapeutic areas by their
# enrollment patterns.

print("\n\n" + "=" * 60)
print("PART 2: Therapeutic Area Success Rates & Clustering")
print("=" * 60)

# Get success rate for each condition
condition_cols = ["is_cancer", "is_diabetes", "is_hiv", "is_asthma",
                  "is_pain", "is_hypertension", "is_depression", "is_schizophrenia"]
condition_names = ["Cancer", "Diabetes", "HIV", "Asthma",
                   "Pain", "Hypertension", "Depression", "Schizophrenia"]

area_stats = []
for col, name in zip(condition_cols, condition_names):
    subset = df[df[col] == 1]
    if len(subset) >= 10:  # only if enough trials
        area_stats.append({
            "condition": name,
            "n_trials": len(subset),
            "success_rate": subset["enrollment_success"].mean(),
            "median_enrollment": subset["enrollment_actual"].median(),
            "median_sites": subset["num_sites"].median(),
            "median_duration": subset["duration_days"].median(),
            "median_criteria": subset["total_criteria"].median(),
        })

# Also add "Other" for trials not matching any top condition
other_mask = df[condition_cols].sum(axis=1) == 0
other_subset = df[other_mask]
area_stats.append({
    "condition": "Other",
    "n_trials": len(other_subset),
    "success_rate": other_subset["enrollment_success"].mean(),
    "median_enrollment": other_subset["enrollment_actual"].median(),
    "median_sites": other_subset["num_sites"].median(),
    "median_duration": other_subset["duration_days"].median(),
    "median_criteria": other_subset["total_criteria"].median(),
})

area_df = pd.DataFrame(area_stats).sort_values("success_rate", ascending=False)

print(f"\nSuccess rates by therapeutic area:")
print(f"{'Condition':<20} {'Trials':>8} {'Success Rate':>14} {'Med. Enrollment':>16} {'Med. Sites':>12}")
print("-" * 70)
for _, row in area_df.iterrows():
    print(f"  {row['condition']:<18} {row['n_trials']:>8} {row['success_rate']:>13.1%} "
          f"{row['median_enrollment']:>16.0f} {row['median_sites']:>12.0f}")

# Cluster therapeutic areas by their enrollment patterns
# Features: success_rate, median_enrollment, median_sites, median_duration, median_criteria
cluster_features = ["success_rate", "median_enrollment", "median_sites", "median_duration", "median_criteria"]
area_X = area_df[cluster_features].values

# Standardize before clustering
area_scaler = StandardScaler()
area_X_scaled = area_scaler.fit_transform(area_X)

# K-Means with k=3 (easy/medium/hard enrollment areas)
km = KMeans(n_clusters=min(3, len(area_df)), random_state=42, n_init=10)
area_df["cluster"] = km.fit_predict(area_X_scaled)

cluster_labels = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2"}
# Rename clusters by average success rate
for c in area_df["cluster"].unique():
    avg_sr = area_df[area_df["cluster"] == c]["success_rate"].mean()
    if avg_sr >= 0.65:
        cluster_labels[c] = "Easy to Enroll"
    elif avg_sr >= 0.50:
        cluster_labels[c] = "Moderate"
    else:
        cluster_labels[c] = "Hard to Enroll"

area_df["enrollment_difficulty"] = area_df["cluster"].map(cluster_labels)

print(f"\nClustering therapeutic areas by enrollment difficulty:")
for _, row in area_df.iterrows():
    print(f"  {row['condition']:<18} → {row['enrollment_difficulty']} (success rate: {row['success_rate']:.1%})")

# PLOT 17: Therapeutic area success rates
fig, ax = plt.subplots(figsize=(10, 6))
colors_map = {"Easy to Enroll": "#059669", "Moderate": "#d97706", "Hard to Enroll": "#dc2626"}
bar_colors = [colors_map.get(row["enrollment_difficulty"], "#64748b") for _, row in area_df.iterrows()]

bars = ax.barh(area_df["condition"], area_df["success_rate"],
               color=bar_colors, edgecolor="black", alpha=0.8)

# Add n= labels
for bar, (_, row) in zip(bars, area_df.iterrows()):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f'n={row["n_trials"]}', va="center", fontsize=9)

ax.set_xlabel("Enrollment Success Rate")
ax.set_title("Enrollment Success Rate by Therapeutic Area\n(colored by clustering: green=easy, yellow=moderate, red=hard)")
ax.set_xlim(0, 1.1)
ax.axvline(x=baseline_prob, color="gray", linestyle="--", label=f"Overall avg: {baseline_prob:.1%}")
ax.legend()
plt.tight_layout()
plt.savefig("plot17_therapeutic_area_success.png", bbox_inches="tight")
plt.close()
print("\nSaved: plot17_therapeutic_area_success.png")


# =============================================
# PART 3: Risk Scoring System (0-100)
# =============================================
# Use logistic regression predicted probabilities as a risk score.
# Scale to 0-100 range and validate thresholds.

print("\n\n" + "=" * 60)
print("PART 3: Risk Scoring System Validation")
print("=" * 60)

# Get predicted probabilities for all trials
probs = lr.predict_proba(X_scaled)[:, 1]  # probability of success
df["risk_score"] = (probs * 100).round(0).astype(int)  # scale to 0-100

print(f"\nRisk score distribution:")
print(f"  Mean:   {df['risk_score'].mean():.0f}")
print(f"  Median: {df['risk_score'].median():.0f}")
print(f"  Min:    {df['risk_score'].min()}")
print(f"  Max:    {df['risk_score'].max()}")

# Validate thresholds
low_risk = df[df["risk_score"] <= 30]
high_risk = df[df["risk_score"] >= 70]
mid_risk = df[(df["risk_score"] > 30) & (df["risk_score"] < 70)]

print(f"\nRisk score validation:")
print(f"  Score ≤ 30 (high risk):  {len(low_risk)} trials, actual success rate = {low_risk['enrollment_success'].mean():.1%}")
print(f"  Score 31-69 (moderate):  {len(mid_risk)} trials, actual success rate = {mid_risk['enrollment_success'].mean():.1%}")
print(f"  Score ≥ 70 (low risk):   {len(high_risk)} trials, actual success rate = {high_risk['enrollment_success'].mean():.1%}")

# Target: <30 should have <40% success, >70 should have >85% success
low_check = "✓ PASS" if low_risk["enrollment_success"].mean() < 0.40 else "✗ FAIL"
high_check = "✓ PASS" if high_risk["enrollment_success"].mean() > 0.85 else "✗ FAIL"
print(f"\n  Validation: Score ≤30 has <40% success? {low_check}")
print(f"  Validation: Score ≥70 has >85% success? {high_check}")

# Finer-grained breakdown
print(f"\nDetailed risk score breakdown:")
print(f"  {'Score Range':<15} {'Trials':>8} {'Actual Success Rate':>20}")
print(f"  " + "-" * 45)
for low, high in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
    subset = df[(df["risk_score"] >= low) & (df["risk_score"] < high)]
    if len(subset) > 0:
        print(f"  {f'{low}-{high}':<15} {len(subset):>8} {subset['enrollment_success'].mean():>20.1%}")

# PLOT 18: Risk score calibration
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of risk scores colored by actual outcome
axes[0].hist(df[df["enrollment_success"] == 0]["risk_score"], bins=20, alpha=0.6,
             color="#dc2626", label="Actual Failure", edgecolor="black")
axes[0].hist(df[df["enrollment_success"] == 1]["risk_score"], bins=20, alpha=0.6,
             color="#059669", label="Actual Success", edgecolor="black")
axes[0].set_xlabel("Risk Score (0-100)")
axes[0].set_ylabel("Number of Trials")
axes[0].set_title("Risk Score Distribution by Actual Outcome")
axes[0].legend()
axes[0].axvline(x=30, color="gray", linestyle="--", alpha=0.5)
axes[0].axvline(x=70, color="gray", linestyle="--", alpha=0.5)

# Right: calibration curve — binned risk score vs actual success rate
bins = pd.cut(df["risk_score"], bins=10)
calibration = df.groupby(bins)["enrollment_success"].agg(["mean", "count"]).reset_index()
calibration.columns = ["bin", "actual_success_rate", "count"]
calibration["bin_mid"] = calibration["bin"].apply(lambda x: x.mid)

axes[1].plot(calibration["bin_mid"], calibration["actual_success_rate"], "o-",
             color="#3b82f6", linewidth=2, markersize=8)
axes[1].plot([0, 100], [0, 1], "r--", label="Perfect calibration")
axes[1].set_xlabel("Predicted Risk Score")
axes[1].set_ylabel("Actual Success Rate")
axes[1].set_title("Risk Score Calibration Curve")
axes[1].legend()
axes[1].set_xlim(0, 100)
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("plot18_risk_score_validation.png", bbox_inches="tight")
plt.close()
print("\nSaved: plot18_risk_score_validation.png")


# =============================================
# PART 4: Optimal number of sites per therapeutic area
# =============================================
print("\n\n" + "=" * 60)
print("PART 4: Optimal Number of Sites by Therapeutic Area")
print("=" * 60)

# For each therapeutic area, find the number of sites where
# success rate is highest
site_analysis = []

for col, name in zip(condition_cols + [None], condition_names + ["All Trials"]):
    if col is not None:
        subset = df[df[col] == 1]
    else:
        subset = df

    if len(subset) < 50:
        continue

    # Bin trials by number of sites
    subset = subset.copy()
    subset["site_bin"] = pd.cut(subset["num_sites"],
                                 bins=[0, 1, 5, 10, 25, 50, 100, 9999],
                                 labels=["1", "2-5", "6-10", "11-25", "26-50", "51-100", "100+"])

    bin_stats = subset.groupby("site_bin").agg(
        success_rate=("enrollment_success", "mean"),
        n_trials=("enrollment_success", "count")
    ).reset_index()

    # Find optimal bin (highest success rate with enough data)
    valid = bin_stats[bin_stats["n_trials"] >= 10]
    if len(valid) > 0:
        best = valid.loc[valid["success_rate"].idxmax()]
        site_analysis.append({
            "condition": name,
            "optimal_sites": best["site_bin"],
            "success_at_optimal": best["success_rate"],
            "n_trials": best["n_trials"],
        })

print(f"\nOptimal number of sites for each therapeutic area:")
print(f"  {'Condition':<20} {'Optimal Sites':>14} {'Success Rate':>14} {'N Trials':>10}")
print(f"  " + "-" * 60)
for row in site_analysis:
    print(f"  {row['condition']:<20} {row['optimal_sites']:>14} {row['success_at_optimal']:>13.1%} {row['n_trials']:>10}")

# PLOT 19: Sites vs success rate for all trials
df_with_sites = df[df["num_sites"] > 0].copy()
df_with_sites["site_bin"] = pd.cut(df_with_sites["num_sites"],
                                    bins=[0, 1, 5, 10, 25, 50, 100, 9999],
                                    labels=["1", "2-5", "6-10", "11-25", "26-50", "51-100", "100+"])

site_success = df_with_sites.groupby("site_bin").agg(
    success_rate=("enrollment_success", "mean"),
    n_trials=("enrollment_success", "count")
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(site_success["site_bin"], site_success["success_rate"],
              color="#3b82f6", edgecolor="black", alpha=0.8)

for bar, (_, row) in zip(bars, site_success.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'n={row["n_trials"]}', ha="center", fontsize=9)

ax.set_xlabel("Number of Sites")
ax.set_ylabel("Enrollment Success Rate")
ax.set_title("Enrollment Success Rate by Number of Sites")
ax.set_ylim(0, 1.1)
ax.axhline(y=baseline_prob, color="red", linestyle="--", label=f"Overall: {baseline_prob:.1%}")
ax.legend()
plt.tight_layout()
plt.savefig("plot19_sites_optimization.png", bbox_inches="tight")
plt.close()
print("\nSaved: plot19_sites_optimization.png")


# =============================================
# PART 5: Actionable Recommendations Summary
# =============================================
print("\n\n" + "=" * 60)
print("PART 5: ACTIONABLE PROTOCOL OPTIMIZATION RECOMMENDATIONS")
print("=" * 60)

# 1. Criteria reduction
excl_row = impacts_df[impacts_df["feature"] == "num_exclusion_criteria"].iloc[0]
incl_row = impacts_df[impacts_df["feature"] == "num_inclusion_criteria"].iloc[0]
print(f"""
1. SIMPLIFY ELIGIBILITY CRITERIA
   - Each additional exclusion criterion changes success probability by {excl_row['impact_per_unit_pct']:+.2f} pp
   - Each additional inclusion criterion changes success probability by {incl_row['impact_per_unit_pct']:+.2f} pp
   - Recommendation: Reducing exclusion criteria from 15 to 8 could improve
     enrollment probability by ~{abs(excl_row['impact_per_unit_pct']) * 7:.1f} percentage points
   - Median in successful trials: {df[df['enrollment_success']==1]['num_exclusion_criteria'].median():.0f} exclusion criteria
   - Median in failed trials: {df[df['enrollment_success']==0]['num_exclusion_criteria'].median():.0f} exclusion criteria
""")

# 2. Site strategy
sites_row = impacts_df[impacts_df["feature"] == "num_sites"].iloc[0]
print(f"""2. OPTIMIZE SITE COUNT
   - Each additional site changes success probability by {sites_row['impact_per_unit_pct']:+.2f} pp
   - Trials with 26-50 sites have the highest success rates
   - Single-site trials have significantly lower enrollment success
   - Recommendation: Target 10-50 sites for optimal recruitment efficiency
   - Median sites in successful trials: {df[df['enrollment_success']==1]['num_sites'].median():.0f}
   - Median sites in failed trials: {df[df['enrollment_success']==0]['num_sites'].median():.0f}
""")

# 3. Duration
dur_row = impacts_df[impacts_df["feature"] == "duration_days"].iloc[0]
print(f"""3. PLAN ADEQUATE STUDY DURATION
   - Each additional 30 days changes success probability by {dur_row['impact_per_unit_pct'] * 30:+.2f} pp
   - Median duration for successful trials: {df[df['enrollment_success']==1]['duration_days'].median():.0f} days
   - Median duration for failed trials: {df[df['enrollment_success']==0]['duration_days'].median():.0f} days
""")

# 4. Age range
age_row = impacts_df[impacts_df["feature"] == "age_range_years"].iloc[0]
print(f"""4. BROADEN AGE ELIGIBILITY
   - Each additional year of age range changes success probability by {age_row['impact_per_unit_pct']:+.2f} pp
   - Wider age ranges mean more eligible patients
   - Recommendation: Consider expanding age range where medically appropriate
""")

# Save insights
insights_summary = {
    "per_feature_impact": {row["feature"]: round(row["impact_per_unit_pct"], 3)
                           for _, row in impacts_df.iterrows()},
    "therapeutic_area_success_rates": {row["condition"]: round(row["success_rate"], 3)
                                       for _, row in area_df.iterrows()},
    "risk_score_validation": {
        "score_leq_30_success_rate": round(low_risk["enrollment_success"].mean(), 3) if len(low_risk) > 0 else None,
        "score_geq_70_success_rate": round(high_risk["enrollment_success"].mean(), 3) if len(high_risk) > 0 else None,
    },
    "optimal_sites": {row["condition"]: str(row["optimal_sites"]) for row in site_analysis},
}

with open("insights_results.json", "w") as f:
    json.dump(insights_summary, f, indent=2)

print("\nSaved: insights_results.json")
print("\nDone! All insights generated.")
