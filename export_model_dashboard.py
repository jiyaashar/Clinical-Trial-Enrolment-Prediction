"""
export_model_dashboard.py — Export a smoother Decision Tree for the dashboard.

The fully tuned tree (depth 15, 155 leaves) gives extreme 0%/100% probabilities
which makes the dashboard always show either "High Risk" or "Low Risk" with nothing
in between. This script trains a shallower tree (depth 5) that gives smoother
probability estimates while still being accurate.

Usage:
    python export_model_dashboard.py

Output:
    model_export.json (overwrites the previous version)
"""

import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

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
y = df["enrollment_success"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
scaler.fit(X_train)

# Shallower tree with more samples per leaf = smoother probabilities
# max_depth=5 gives ~32 leaves max, each with many samples
# min_samples_leaf=50 ensures every leaf has enough data for a reliable probability
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=50,
    random_state=42
)
model.fit(scaler.transform(X_train), y_train)

y_pred = model.predict(scaler.transform(X_test))
print(f"Dashboard model accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Dashboard model F1:       {f1_score(y_test, y_pred):.4f}")
print(f"Tree depth: {model.get_depth()}")
print(f"Tree leaves: {model.get_n_leaves()}")

# Check probability distribution
probs = model.predict_proba(scaler.transform(X_test))[:, 1]
print(f"Probability range: {probs.min():.3f} to {probs.max():.3f}")
print(f"Unique probabilities: {len(np.unique(probs.round(3)))}")

# Export
def tree_to_json(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i >= 0 else "leaf" for i in tree_.feature]

    def recurse(node):
        if tree_.feature[node] == -2:
            values = tree_.value[node][0]
            total = values.sum()
            return {
                "type": "leaf",
                "prediction": int(np.argmax(values)),
                "probability": round(float(values[1] / total), 4),
                "samples": int(total),
            }
        else:
            return {
                "type": "split",
                "feature": feature_name[node],
                "feature_index": int(tree_.feature[node]),
                "threshold": round(float(tree_.threshold[node]), 6),
                "left": recurse(tree_.children_left[node]),
                "right": recurse(tree_.children_right[node]),
            }
    return recurse(0)

tree_json = tree_to_json(model, FEATURE_COLS)

scaler_params = {
    "mean": {name: round(float(m), 6) for name, m in zip(FEATURE_COLS, scaler.mean_)},
    "std": {name: round(float(s), 6) for name, s in zip(FEATURE_COLS, scaler.scale_)},
}

export = {
    "features": FEATURE_COLS,
    "scaler": scaler_params,
    "tree": tree_json,
    "metadata": {
        "model": "DecisionTreeClassifier (dashboard version - smoother probabilities)",
        "params": {"max_depth": 5, "min_samples_leaf": 50},
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "note": "Shallower tree for smoother probability estimates in the dashboard. The full tuned model (depth 15, F1=0.974) is used for final results."
    }
}

with open("model_export.json", "w") as f:
    json.dump(export, f)

import os
size_kb = os.path.getsize("model_export.json") / 1024
print(f"\nSaved model_export.json ({size_kb:.0f} KB)")
