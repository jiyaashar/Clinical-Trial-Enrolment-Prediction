"""
advanced_models.py — Regression, Neural Network, and Hyperparameter Tuning.

Takes model_ready.csv and:
  1. Regression: predicts exact enrollment ratio (not just binary)
  2. Neural Network: MLP classifier for enrollment success
  3. Hyperparameter tuning: GridSearch on Decision Tree and Random Forest
  4. Compares everything and saves results

Usage:
    python advanced_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 150, "font.size": 12})


# =============================================
# LOAD DATA
# =============================================
print("Loading data...\n")

df = pd.read_csv("model_ready.csv")

FEATURE_COLS = [
    "duration_days", "num_sites", "num_countries", "num_arms",
    "num_interventions", "enrollment_per_site", "has_multiple_arms",
    "num_inclusion_criteria", "num_exclusion_criteria", "total_criteria",
    "criteria_text_length", "min_age_years", "max_age_years", "age_range_years",
    "sex_restricted", "is_phase2", "is_phase3", "is_industry_sponsored",
    "is_cancer", "is_diabetes", "is_hiv", "is_asthma",
    "is_pain", "is_hypertension", "is_depression", "is_schizophrenia",
]

X = df[FEATURE_COLS].fillna(0)
y_class = df["enrollment_success"]  # binary target for classification

# For regression we need enrollment_actual — predict log(enrollment)
# since enrollment is right-skewed
y_reg = np.log1p(df["enrollment_actual"])  # log(enrollment + 1)

# Split — same random state so train/test sets match across all models
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)
_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42, stratify=y_class
)

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index)

print(f"  Training: {len(X_train)} trials, Test: {len(X_test)} trials")


# =============================================
# PART 1: REGRESSION MODELS
# =============================================
# Instead of just predicting success/failure, we predict the actual
# enrollment number. This gives more granular predictions.
# We use log(enrollment) because enrollment is right-skewed.

print("\n" + "=" * 60)
print("PART 1: REGRESSION — Predicting log(enrollment)")
print("=" * 60)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),      # L2 regularization — shrinks coefficients
    "Lasso Regression": Lasso(alpha=0.1),       # L1 regularization — can zero out features
}

reg_results = {}

for name, model in reg_models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_scaled, y_train_reg)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test_reg, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)

    reg_results[name] = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "y_pred": y_pred}

    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    R²:   {r2:.4f}")

# PLOT 14: Predicted vs actual enrollment (best regression model)
best_reg = max(reg_results.items(), key=lambda x: x[1]["r2"])
print(f"\n  Best regression model: {best_reg[0]} (R² = {best_reg[1]['r2']:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: predicted vs actual (log scale)
axes[0].scatter(y_test_reg, best_reg[1]["y_pred"], alpha=0.3, s=10, color="#2196F3")
min_val = min(y_test_reg.min(), best_reg[1]["y_pred"].min())
max_val = max(y_test_reg.max(), best_reg[1]["y_pred"].max())
axes[0].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
axes[0].set_xlabel("Actual log(enrollment)")
axes[0].set_ylabel("Predicted log(enrollment)")
axes[0].set_title(f"Predicted vs Actual — {best_reg[0]} (R²={best_reg[1]['r2']:.3f})")
axes[0].legend()

# Residual plot — should be random scatter around 0
residuals = y_test_reg - best_reg[1]["y_pred"]
axes[1].scatter(best_reg[1]["y_pred"], residuals, alpha=0.3, s=10, color="#FF9800")
axes[1].axhline(y=0, color="red", linestyle="--")
axes[1].set_xlabel("Predicted log(enrollment)")
axes[1].set_ylabel("Residual (actual - predicted)")
axes[1].set_title("Residual Plot")

plt.tight_layout()
plt.savefig("plot14_regression_results.png", bbox_inches="tight")
plt.close()
print("  Saved: plot14_regression_results.png")


# =============================================
# PART 2: NEURAL NETWORK
# =============================================
# MLPClassifier = Multi-Layer Perceptron, a basic neural network.
# It has hidden layers of neurons that learn non-linear patterns.

print("\n" + "=" * 60)
print("PART 2: NEURAL NETWORK — MLP Classifier")
print("=" * 60)

# Try a few different architectures
nn_configs = {
    "MLP (64)":        MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42),
    "MLP (128, 64)":   MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    "MLP (128, 64, 32)": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42),
}

nn_results = {}

for name, model in nn_configs.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_scaled, y_train_class)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test_class, y_pred)
    prec = precision_score(y_test_class, y_pred, zero_division=0)
    rec = recall_score(y_test_class, y_pred, zero_division=0)
    f1 = f1_score(y_test_class, y_pred, zero_division=0)
    cv = cross_val_score(model, X_train_scaled, y_train_class, cv=5, scoring="accuracy")
    cm = confusion_matrix(y_test_class, y_pred)

    nn_results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "cv_mean": cv.mean(), "cv_std": cv.std(), "confusion_matrix": cm,
    }

    print(f"    Accuracy:    {acc:.4f}")
    print(f"    F1 Score:    {f1:.4f}")
    print(f"    CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")

# PLOT 15: Neural network confusion matrices
best_nn_name = max(nn_results, key=lambda x: nn_results[x]["f1"])
best_nn = nn_results[best_nn_name]
print(f"\n  Best NN: {best_nn_name} (F1 = {best_nn['f1']:.4f})")

fig, axes = plt.subplots(1, len(nn_results), figsize=(5 * len(nn_results), 4))
for ax, (name, res) in zip(axes, nn_results.items()):
    ConfusionMatrixDisplay(
        confusion_matrix=res["confusion_matrix"],
        display_labels=["Failure", "Success"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\nAcc={res['accuracy']:.3f}", fontsize=10)

plt.suptitle("Neural Network Confusion Matrices", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("plot15_nn_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("  Saved: plot15_nn_confusion_matrices.png")


# =============================================
# PART 3: HYPERPARAMETER TUNING
# =============================================
# GridSearchCV tries every combination of hyperparameters
# and picks the one with the best cross-validation score.

print("\n" + "=" * 60)
print("PART 3: HYPERPARAMETER TUNING — GridSearchCV")
print("=" * 60)

# --- Tune Decision Tree ---
print("\n  Tuning Decision Tree...")
dt_param_grid = {
    "max_depth": [3, 5, 8, 10, 15, None],       # how deep the tree can grow
    "min_samples_split": [2, 5, 10, 20],          # min samples to split a node
    "min_samples_leaf": [1, 2, 5, 10],             # min samples in a leaf
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    cv=5,               # 5-fold cross-validation
    scoring="f1",        # optimize for F1 score
    n_jobs=-1,           # use all CPU cores
    verbose=0
)
dt_grid.fit(X_train_scaled, y_train_class)

print(f"    Best params: {dt_grid.best_params_}")
print(f"    Best CV F1:  {dt_grid.best_score_:.4f}")

# Evaluate tuned model on test set
dt_tuned_pred = dt_grid.predict(X_test_scaled)
dt_tuned_acc = accuracy_score(y_test_class, dt_tuned_pred)
dt_tuned_f1 = f1_score(y_test_class, dt_tuned_pred)
print(f"    Test Accuracy: {dt_tuned_acc:.4f}")
print(f"    Test F1:       {dt_tuned_f1:.4f}")

# --- Tune Random Forest ---
print("\n  Tuning Random Forest...")
rf_param_grid = {
    "n_estimators": [50, 100, 200],               # number of trees
    "max_depth": [5, 10, 15, None],                # max tree depth
    "min_samples_split": [2, 5, 10],               # min samples to split
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train_scaled, y_train_class)

print(f"    Best params: {rf_grid.best_params_}")
print(f"    Best CV F1:  {rf_grid.best_score_:.4f}")

# Evaluate tuned model on test set
rf_tuned_pred = rf_grid.predict(X_test_scaled)
rf_tuned_acc = accuracy_score(y_test_class, rf_tuned_pred)
rf_tuned_f1 = f1_score(y_test_class, rf_tuned_pred)
print(f"    Test Accuracy: {rf_tuned_acc:.4f}")
print(f"    Test F1:       {rf_tuned_f1:.4f}")


# =============================================
# PLOT 16: All models comparison (original + tuned + NN)
# =============================================
print("\nGenerating final comparison plot...")

# Load original results
with open("model_results.json") as f:
    original = json.load(f)

# Combine everything
all_results = {}
for name, res in original.items():
    all_results[name] = res["f1"]

all_results["DT (tuned)"] = dt_tuned_f1
all_results["RF (tuned)"] = rf_tuned_f1
all_results[best_nn_name] = best_nn["f1"]

# Sort by F1
all_results = dict(sorted(all_results.items(), key=lambda x: x[1], reverse=True))

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#4CAF50" if "tuned" in name or "MLP" in name else "#2196F3" for name in all_results.keys()]
bars = ax.bar(all_results.keys(), all_results.values(), color=colors, edgecolor="black", alpha=0.85)

# Add value labels
for bar, val in zip(bars, all_results.values()):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel("Model")
ax.set_ylabel("F1 Score")
ax.set_title("All Models Comparison (green = advanced/tuned)")
ax.set_ylim(0, 1.1)
ax.set_xticklabels(all_results.keys(), rotation=20, ha="right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot16_all_models_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved: plot16_all_models_comparison.png")


# =============================================
# FINAL SUMMARY
# =============================================
print("\n" + "=" * 60)
print("FINAL SUMMARY — ALL MODELS")
print("=" * 60)

print(f"\n--- Classification Models (F1 Score) ---")
for name, f1_val in all_results.items():
    marker = " ← BEST" if f1_val == max(all_results.values()) else ""
    print(f"  {name:<30} F1 = {f1_val:.4f}{marker}")

print(f"\n--- Regression Models (R²) ---")
for name, res in reg_results.items():
    print(f"  {name:<30} R² = {res['r2']:.4f}, RMSE = {res['rmse']:.4f}")

print(f"\n--- Hyperparameter Tuning ---")
print(f"  Decision Tree best params: {dt_grid.best_params_}")
print(f"  Random Forest best params: {rf_grid.best_params_}")

print(f"\n--- Neural Network Architectures ---")
for name, res in nn_results.items():
    print(f"  {name:<30} Acc = {res['accuracy']:.4f}, F1 = {res['f1']:.4f}")

# Save all advanced results
advanced_summary = {
    "regression": {name: {"r2": round(r["r2"], 4), "rmse": round(r["rmse"], 4)} for name, r in reg_results.items()},
    "neural_networks": {name: {"accuracy": round(r["accuracy"], 4), "f1": round(r["f1"], 4)} for name, r in nn_results.items()},
    "tuned_decision_tree": {"params": dt_grid.best_params_, "f1": round(dt_tuned_f1, 4)},
    "tuned_random_forest": {"params": rf_grid.best_params_, "f1": round(rf_tuned_f1, 4)},
}

with open("advanced_results.json", "w") as f:
    json.dump(advanced_summary, f, indent=2)

print("\nSaved: advanced_results.json")
print("\nDone! All advanced models complete.")
