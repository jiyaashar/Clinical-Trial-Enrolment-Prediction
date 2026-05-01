"""
train_models.py — Train and evaluate classification models.

Takes model_ready.csv and:
  1. Splits data into train/test sets
  2. Trains 5 classification models (KNN, Decision Tree, Random Forest, Logistic Regression, Naive Bayes)
  3. Evaluates each with accuracy, precision, recall, F1, confusion matrix
  4. Uses 5-fold cross-validation to check for overfitting
  5. Plots model comparison and feature importance
  6. Saves results to model_results.json

Usage:
    python train_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 150, "font.size": 12})


# =============================================
# STEP 1: Load data and prepare train/test split
# =============================================
print("Step 1: Loading data and preparing train/test split...\n")

df = pd.read_csv("model_ready.csv")
print(f"  Loaded {len(df)} trials")

# These are our 26 input features (X)
FEATURE_COLS = [
    "duration_days", "num_sites", "num_countries", "num_arms",
    "num_interventions", "enrollment_per_site", "has_multiple_arms",
    "num_inclusion_criteria", "num_exclusion_criteria", "total_criteria",
    "criteria_text_length", "min_age_years", "max_age_years", "age_range_years",
    "sex_restricted", "is_phase2", "is_phase3", "is_industry_sponsored",
    "is_cancer", "is_diabetes", "is_hiv", "is_asthma",
    "is_pain", "is_hypertension", "is_depression", "is_schizophrenia",
]

# This is what we're predicting (y)
TARGET = "enrollment_success"

X = df[FEATURE_COLS].fillna(0)
y = df[TARGET]

# 80/20 split — stratified so both sets have ~59% success rate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features — important for KNN (uses distance) and Logistic Regression
# StandardScaler makes each feature have mean=0, std=1
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
)

print(f"  Training set: {len(X_train)} trials")
print(f"  Test set:     {len(X_test)} trials")
print(f"  Features:     {len(FEATURE_COLS)}")
print(f"  Train success rate: {y_train.mean():.1%}")
print(f"  Test success rate:  {y_test.mean():.1%}")


# =============================================
# STEP 2: Define and train all models
# =============================================
print("\nStep 2: Training models...\n")

# Each model and why we chose it:
#   KNN: simple baseline, classifies by looking at most similar trials
#   Decision Tree: interpretable rules, shows which features matter
#   Random Forest: ensemble of trees, usually more accurate
#   Logistic Regression: gives probability estimates, good for understanding feature effects
#   Naive Bayes: fast, probabilistic, assumes feature independence
models = {
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
}

results = {}

for name, model in models.items():
    print(f"  Training {name}...")

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_scaled)

    # 5-fold cross-validation on training data
    # This gives us a more reliable accuracy estimate than a single train/test split
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")

    # Calculate all metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "confusion_matrix": cm,
    }

    print(f"    Accuracy:    {acc:.4f}")
    print(f"    Precision:   {prec:.4f}")
    print(f"    Recall:      {rec:.4f}")
    print(f"    F1 Score:    {f1:.4f}")
    print(f"    CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()


# =============================================
# STEP 3: Print detailed classification reports
# =============================================
print("=" * 60)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 60)

for name, res in results.items():
    print(f"\n--- {name} ---")
    print(classification_report(y_test, res["y_pred"], target_names=["Failure", "Success"]))


# =============================================
# PLOT 9: Model comparison bar chart
# =============================================
print("\nGenerating plots...")

model_names = list(results.keys())
metrics = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(model_names))
width = 0.2

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    values = [results[m][metric] for m in model_names]
    ax.bar(x + i * width, values, width, label=label, alpha=0.85)

ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, rotation=15, ha="right")
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plot9_model_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved: plot9_model_comparison.png")


# =============================================
# PLOT 10: Confusion matrices for all models
# =============================================
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))

for ax, (name, res) in zip(axes, results.items()):
    ConfusionMatrixDisplay(
        confusion_matrix=res["confusion_matrix"],
        display_labels=["Failure", "Success"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}", fontsize=10)

plt.suptitle("Confusion Matrices", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("plot10_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("  Saved: plot10_confusion_matrices.png")


# =============================================
# PLOT 11: Feature importance (Random Forest)
# =============================================
# Random Forest gives us feature_importances_ — how much each feature
# contributes to the model's predictions. Higher = more important.

rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]  # sort by importance

fig, ax = plt.subplots(figsize=(10, 8))
top_n = min(15, len(FEATURE_COLS))
top_indices = indices[:top_n]

ax.barh(
    [FEATURE_COLS[i] for i in reversed(top_indices)],
    [importances[i] for i in reversed(top_indices)],
    color="#2196F3", edgecolor="black", alpha=0.8
)
ax.set_xlabel("Feature Importance")
ax.set_title("Top 15 Most Important Features (Random Forest)")
plt.tight_layout()
plt.savefig("plot11_feature_importance_rf.png", bbox_inches="tight")
plt.close()
print("  Saved: plot11_feature_importance_rf.png")


# =============================================
# PLOT 12: Feature importance (Logistic Regression coefficients)
# =============================================
# Logistic Regression coefficients tell us direction AND magnitude:
#   positive = increases chance of success
#   negative = decreases chance of success

lr_model = results["Logistic Regression"]["model"]
coefficients = lr_model.coef_[0]
coef_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "coefficient": coefficients
}).sort_values("coefficient")

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#F44336" if c < 0 else "#4CAF50" for c in coef_df["coefficient"]]
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors, edgecolor="black", alpha=0.8)
ax.set_xlabel("Coefficient (negative = hurts enrollment, positive = helps)")
ax.set_title("Logistic Regression Coefficients")
ax.axvline(x=0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig("plot12_feature_importance_lr.png", bbox_inches="tight")
plt.close()
print("  Saved: plot12_feature_importance_lr.png")


# =============================================
# STEP 4: Decision Tree rules (interpretability)
# =============================================
print("\n" + "=" * 60)
print("DECISION TREE RULES (top 3 levels)")
print("=" * 60)

dt_model = results["Decision Tree"]["model"]
tree_rules = export_text(dt_model, feature_names=FEATURE_COLS, max_depth=3)
print(tree_rules)


# =============================================
# PLOT 13: Cross-validation comparison
# =============================================
fig, ax = plt.subplots(figsize=(10, 6))
cv_means = [results[m]["cv_mean"] for m in model_names]
cv_stds = [results[m]["cv_std"] for m in model_names]

bars = ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5,
              color="#2196F3", edgecolor="black", alpha=0.8)
ax.set_xlabel("Model")
ax.set_ylabel("Cross-Validation Accuracy")
ax.set_title("5-Fold Cross-Validation Accuracy (± 1 std)")
ax.set_ylim(0, 1.1)
ax.set_xticklabels(model_names, rotation=15, ha="right")

# Add value labels on bars
for bar, mean in zip(bars, cv_means):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("plot13_cv_comparison.png", bbox_inches="tight")
plt.close()
print("\n  Saved: plot13_cv_comparison.png")


# =============================================
# STEP 5: Save results summary
# =============================================
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

summary = {}
print(f"\n{'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'CV Acc':>10}")
print("-" * 67)

for name, res in results.items():
    print(f"{name:<25} {res['accuracy']:>8.4f} {res['precision']:>8.4f} "
          f"{res['recall']:>8.4f} {res['f1']:>8.4f} {res['cv_mean']:>8.4f}±{res['cv_std']:.3f}")
    summary[name] = {
        "accuracy": round(res["accuracy"], 4),
        "precision": round(res["precision"], 4),
        "recall": round(res["recall"], 4),
        "f1": round(res["f1"], 4),
        "cv_mean": round(res["cv_mean"], 4),
        "cv_std": round(res["cv_std"], 4),
    }

best = max(results.items(), key=lambda x: x[1]["f1"])
print(f"\nBest model by F1: {best[0]} ({best[1]['f1']:.4f})")

# Save to JSON
with open("model_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved: model_results.json")


# =============================================
# STEP 6: Top feature insights
# =============================================
print("\n" + "=" * 60)
print("KEY FINDINGS — Feature Importance")
print("=" * 60)

# Top 5 from Random Forest
print("\nTop 5 features (Random Forest):")
for i in range(5):
    idx = indices[i]
    print(f"  {i+1}. {FEATURE_COLS[idx]}: {importances[idx]:.4f}")

# Direction from Logistic Regression
print("\nFactors that HELP enrollment (Logistic Regression):")
positive = coef_df[coef_df["coefficient"] > 0.1].sort_values("coefficient", ascending=False)
for _, row in positive.head(5).iterrows():
    print(f"  + {row['feature']}: {row['coefficient']:.3f}")

print("\nFactors that HURT enrollment (Logistic Regression):")
negative = coef_df[coef_df["coefficient"] < -0.1].sort_values("coefficient")
for _, row in negative.head(5).iterrows():
    print(f"  - {row['feature']}: {row['coefficient']:.3f}")

print("\nDone! All plots and results saved.")
