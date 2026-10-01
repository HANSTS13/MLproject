# ==========================================
# TEAM 02: CLASS IMBALANCE INVESTIGATION
# Research Question:
# Can a classifier with high accuracy still be practically poor?
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. GENERATE AN IMBALANCED DATASET
# ==========================================

# 95% Majority Class (0)
# 5% Minority Class (1)

X, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    weights=[0.95, 0.05],
    random_state=42
)

# Split into training and testing sets
# stratify=y keeps the same class imbalance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Total samples        : {len(X)}")
print(f"Training samples     : {X_train.shape[0]}")
print(f"Testing samples      : {X_test.shape[0]}")

print("\nTraining Class Distribution:")
print(pd.Series(y_train).value_counts())
print(pd.Series(y_train).value_counts(normalize=True))

print("\nTesting Class Distribution:")
print(pd.Series(y_test).value_counts())
print(pd.Series(y_test).value_counts(normalize=True))


# ==========================================
# 2. MAJORITY-CLASS BASELINE
# ==========================================

# This classifier completely ignores the features.
# It always predicts the majority class (Class 0).

y_baseline = np.zeros_like(y_test)

baseline_acc = accuracy_score(y_test, y_baseline)
baseline_prec = precision_score(
    y_test, y_baseline, zero_division=0
)
baseline_rec = recall_score(
    y_test, y_baseline, zero_division=0
)
baseline_f1 = f1_score(
    y_test, y_baseline, zero_division=0
)

baseline_cm = confusion_matrix(y_test, y_baseline)

print("\n" + "=" * 60)
print("MAJORITY-CLASS BASELINE")
print("=" * 60)

print("This model predicts Class 0 for every observation.")

print(f"\nAccuracy  : {baseline_acc:.4f}")
print(f"Precision : {baseline_prec:.4f}")
print(f"Recall    : {baseline_rec:.4f}")
print(f"F1-Score  : {baseline_f1:.4f}")

print("\nBaseline Confusion Matrix:")
print(baseline_cm)

print("\nBaseline Classification Report:")
print(
    classification_report(
        y_test,
        y_baseline,
        zero_division=0
    )
)


# ==========================================
# 3. TRAIN LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)


# ==========================================
# 4. PREDICT TEST DATA
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 5. CALCULATE LOGISTIC REGRESSION METRICS
# ==========================================

acc = accuracy_score(y_test, y_pred)

prec = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

rec = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(y_test, y_pred)


# ==========================================
# 6. DISPLAY LOGISTIC REGRESSION RESULTS
# ==========================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# 7. CREATE COMPARISON TABLE
# ==========================================

results_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],

    "Majority Baseline": [
        baseline_acc,
        baseline_prec,
        baseline_rec,
        baseline_f1
    ],

    "Logistic Regression": [
        acc,
        prec,
        rec,
        f1
    ]
})

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# ==========================================
# 8. CONFUSION MATRICES
# ==========================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

# ------------------------------------------
# A. Majority-Class Baseline
# ------------------------------------------

sns.heatmap(
    baseline_cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    cbar=False,
    ax=axes[0]
)

axes[0].set_title(
    "Majority-Class Baseline\n(Always Predicts Class 0)"
)

axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")


# ------------------------------------------
# B. Logistic Regression
# ------------------------------------------

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    ax=axes[1]
)

axes[1].set_title(
    "Logistic Regression\nConfusion Matrix"
)

axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")


plt.tight_layout()
plt.show()


# ==========================================
# 9. METRIC COMPARISON BAR CHART
# ==========================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(
    figsize=(12, 6)
)

bars1 = ax.bar(
    x - width / 2,
    results_df["Majority Baseline"],
    width,
    label="Majority-Class Baseline",
    color="salmon"
)

bars2 = ax.bar(
    x + width / 2,
    results_df["Logistic Regression"],
    width,
    label="Logistic Regression",
    color="skyblue"
)

ax.set_title(
    "Performance Comparison: Accuracy vs. Other Metrics"
)

ax.set_ylabel("Score")
ax.set_xlabel("Evaluation Metric")

ax.set_xticks(x)
ax.set_xticklabels(metrics)

ax.set_ylim(0, 1.05)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)


# Add value labels to bars
for bar in bars1:
    height = bar.get_height()

    ax.annotate(
        f"{height:.3f}",
        xy=(
            bar.get_x() + bar.get_width() / 2,
            height
        ),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom"
    )


for bar in bars2:
    height = bar.get_height()

    ax.annotate(
        f"{height:.3f}",
        xy=(
            bar.get_x() + bar.get_width() / 2,
            height
        ),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom"
    )


plt.tight_layout()
plt.show()


# ==========================================
# 10. FINAL INTERPRETATION
# ==========================================

print("\n" + "=" * 60)
print("FINAL INTERPRETATION")
print("=" * 60)

print("""
This experiment demonstrates why accuracy alone can be misleading
when dealing with an imbalanced classification dataset.

The dataset contains approximately 95% Class 0 observations and
only 5% Class 1 observations.

The majority-class baseline predicts Class 0 for every observation.
As a result, it achieves approximately 95% accuracy simply because
Class 0 represents most of the dataset.

However, the baseline has:

- 0% recall for the minority Class 1
- 0% precision for the minority Class 1
- 0% F1-score for the minority Class 1

Therefore, the model completely fails to identify the minority class
despite having high accuracy.

The Logistic Regression model provides a more meaningful comparison
because it attempts to identify both classes using the input features.

This demonstrates that for imbalanced classification problems,
accuracy should not be considered by itself. Precision, recall,
F1-score, and the confusion matrix provide additional information
about how well the classifier handles the minority class.
""")
