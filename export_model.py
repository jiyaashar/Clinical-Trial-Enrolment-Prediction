"""
export_model.py — Export the trained Decision Tree as JSON for the dashboard.

Trains the tuned Decision Tree (same as advanced_models.py) and exports
its complete structure so the dashboard can make real predictions.

Usage:
    python export_model.py

Output:
    model_export.json — contains tree structure, scaler parameters, and feature names
"""

import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# =============================================
# Train the same tuned Decision Tree
# =============================================
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

# Same split as train_models.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
scaler.fit(X_train)

# Train tuned Decision Tree (same params as GridSearchCV found)
model = DecisionTreeClassifier(max_depth=15, min_samples_leaf=5, min_samples_split=2, random_state=42)
model.fit(scaler.transform(X_train), y_train)

# Verify accuracy matches
from sklearn.metrics import accuracy_score, f1_score
y_pred = model.predict(scaler.transform(X_test))
print(f"Exported model accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Exported model F1:       {f1_score(y_test, y_pred):.4f}")


# =============================================
# Export tree structure as JSON
# =============================================
def tree_to_json(tree, feature_names):
    """Convert a sklearn Decision Tree to a nested JSON structure."""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i >= 0 else "leaf"
        for i in tree_.feature
    ]

    def recurse(node):
        if tree_.feature[node] == -2:  # leaf node
            # Return the class with the most samples
            values = tree_.value[node][0]
            total = values.sum()
            return {
                "type": "leaf",
                "prediction": int(np.argmax(values)),
                "probability": round(float(values[1] / total), 4),  # P(success)
                "samples": int(total),
            }
        else:
            return {
                "type": "split",
                "feature": feature_name[node],
                "feature_index": int(tree_.feature[node]),
                "threshold": round(float(tree_.threshold[node]), 6),
                "left": recurse(tree_.children_left[node]),   # <= threshold
                "right": recurse(tree_.children_right[node]),  # > threshold
            }

    return recurse(0)


tree_json = tree_to_json(model, FEATURE_COLS)

# Export scaler parameters so dashboard can standardize inputs
scaler_params = {
    "mean": {name: round(float(m), 6) for name, m in zip(FEATURE_COLS, scaler.mean_)},
    "std": {name: round(float(s), 6) for name, s in zip(FEATURE_COLS, scaler.scale_)},
}

export = {
    "features": FEATURE_COLS,
    "scaler": scaler_params,
    "tree": tree_json,
    "metadata": {
        "model": "DecisionTreeClassifier",
        "params": {"max_depth": 15, "min_samples_leaf": 5, "min_samples_split": 2},
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
    }
}

with open("model_export.json", "w") as f:
    json.dump(export, f)

# Print file size
import os
size_kb = os.path.getsize("model_export.json") / 1024
print(f"\nSaved model_export.json ({size_kb:.0f} KB)")
print(f"Tree depth: {model.get_depth()}")
print(f"Tree leaves: {model.get_n_leaves()}")
