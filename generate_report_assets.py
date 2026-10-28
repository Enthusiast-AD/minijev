"""
generate_report_assets.py
=========================
Trains the Banking Intent Classification ANN model, evaluates it,
generates all visualisation plots, and saves a metrics JSON for the report.

Run:
    python generate_report_assets.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────
# 0.  Paths
# ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "dataset"
OUTPUT_DIR  = BASE_DIR / "report_assets"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH      = DATA_DIR / "train.csv"
TEST_PATH       = DATA_DIR / "test.csv"
CATEGORIES_PATH = DATA_DIR / "categories.json"

# ────────────────────────────────────────────────────────────────────
# 1.  Load data
# ────────────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
    categories = json.load(f)

# Normalise columns
train_df.columns = train_df.columns.str.strip().str.lower()
test_df.columns  = test_df.columns.str.strip().str.lower()

aliases = {"query": "text", "utterance": "text", "category": "label", "intent": "label"}
train_df = train_df.rename(columns=aliases)
test_df  = test_df.rename(columns=aliases)

train_df = train_df.dropna(subset=["text", "label"]).copy()
test_df  = test_df.dropna(subset=["text", "label"]).copy()
train_df["text"] = train_df["text"].astype(str).str.strip()
test_df["text"]  = test_df["text"].astype(str).str.strip()
train_df = train_df[train_df["text"] != ""]
test_df  = test_df[test_df["text"] != ""]

X      = train_df["text"].to_numpy()
y      = train_df["label"].to_numpy()
X_test = test_df["text"].to_numpy()
y_test = test_df["label"].to_numpy()

print(f"Training samples : {len(X)}")
print(f"Test samples     : {len(X_test)}")
print(f"Classes          : {len(np.unique(y))}")

# ────────────────────────────────────────────────────────────────────
# 2.  Category names
# ────────────────────────────────────────────────────────────────────
if isinstance(categories, list):
    category_names = {str(i): name for i, name in enumerate(categories)}
elif isinstance(categories, dict):
    if all(isinstance(v, str) for v in categories.values()):
        category_names = {str(k): v for k, v in categories.items()}
    else:
        category_names = {str(v): str(k) for k, v in categories.items()}
else:
    category_names = {}

for label in np.unique(y):
    category_names.setdefault(str(label), str(label))

def display_label(label):
    return category_names.get(str(label), str(label))

# ────────────────────────────────────────────────────────────────────
# 3.  Train / validation split
# ────────────────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# ────────────────────────────────────────────────────────────────────
# 4.  Build and train pipeline
# ────────────────────────────────────────────────────────────────────
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )),
    ("ann", MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        verbose=True,
    )),
])

print("\nTraining …")
model.fit(X_train, y_train)

# ────────────────────────────────────────────────────────────────────
# 5.  Evaluate
# ────────────────────────────────────────────────────────────────────
val_pred  = model.predict(X_val)
test_pred = model.predict(X_test)

val_acc   = accuracy_score(y_val, val_pred)
test_acc  = accuracy_score(y_test, test_pred)
val_f1    = f1_score(y_val,  val_pred,  average="macro", zero_division=0)
test_f1   = f1_score(y_test, test_pred, average="macro", zero_division=0)
test_prec = precision_score(y_test, test_pred, average="macro", zero_division=0)
test_rec  = recall_score(y_test, test_pred, average="macro", zero_division=0)

labels      = model.named_steps["ann"].classes_
label_names = [display_label(l) for l in labels]

report_dict = classification_report(
    y_test, test_pred,
    labels=labels,
    target_names=label_names,
    zero_division=0,
    output_dict=True,
)
report_str = classification_report(
    y_test, test_pred,
    labels=labels,
    target_names=label_names,
    zero_division=0,
)

print("\n====  VALIDATION  ====")
print(f"Accuracy : {val_acc:.4f}")
print(f"Macro-F1 : {val_f1:.4f}")
print("\n====  TEST  ====")
print(f"Accuracy  : {test_acc:.4f}")
print(f"Precision : {test_prec:.4f}")
print(f"Recall    : {test_rec:.4f}")
print(f"Macro-F1  : {test_f1:.4f}")
print("\nClassification Report:\n", report_str)

# ────────────────────────────────────────────────────────────────────
# 6.  Save metrics JSON
# ────────────────────────────────────────────────────────────────────
metrics = {
    "val_accuracy" : round(val_acc,  4),
    "val_macro_f1" : round(val_f1,   4),
    "test_accuracy": round(test_acc, 4),
    "test_precision_macro": round(test_prec, 4),
    "test_recall_macro":    round(test_rec,  4),
    "test_macro_f1":        round(test_f1,   4),
    "num_classes": int(len(labels)),
    "train_samples": int(len(X_train)),
    "val_samples":   int(len(X_val)),
    "test_samples":  int(len(X_test)),
    "per_class": {
        name: {
            "precision": round(report_dict[name]["precision"], 4),
            "recall":    round(report_dict[name]["recall"],    4),
            "f1-score":  round(report_dict[name]["f1-score"],  4),
            "support":   int(report_dict[name]["support"]),
        }
        for name in label_names
        if name in report_dict
    },
}

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved metrics → {OUTPUT_DIR / 'metrics.json'}")

# ────────────────────────────────────────────────────────────────────
# 7.  PLOT 1 — Training loss curve
# ────────────────────────────────────────────────────────────────────
ann = model.named_steps["ann"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ann.loss_curve_, color="#4F6EF7", linewidth=2, label="Training Loss")
if hasattr(ann, "validation_scores_") and ann.validation_scores_:
    # validation_scores_ is accuracy – invert for "val loss" proxy
    ax.plot(
        [1 - s for s in ann.validation_scores_],
        color="#F76F6E", linewidth=2, linestyle="--", label="Validation Error (1−acc)"
    )
ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Loss", fontsize=13)
ax.set_title("Training Loss Curve — Banking Intent ANN", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_training_loss_curve.png", dpi=150)
plt.close(fig)
print("Saved → 01_training_loss_curve.png")

# ────────────────────────────────────────────────────────────────────
# 8.  PLOT 2 — Overall metrics bar chart
# ────────────────────────────────────────────────────────────────────
metric_names  = ["Accuracy", "Precision\n(Macro)", "Recall\n(Macro)", "F1-Score\n(Macro)"]
metric_values = [test_acc, test_prec, test_rec, test_f1]
colors = ["#4F6EF7", "#43C59E", "#F7C948", "#F76F6E"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metric_names, metric_values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, metric_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.4f}",
        ha="center", va="bottom", fontsize=13, fontweight="bold"
    )
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score", fontsize=13)
ax.set_title("Model Performance Metrics (Test Set)", fontsize=15, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_metrics_bar_chart.png", dpi=150)
plt.close(fig)
print("Saved → 02_metrics_bar_chart.png")

# ────────────────────────────────────────────────────────────────────
# 9.  PLOT 3 — Class distribution (train)
# ────────────────────────────────────────────────────────────────────
class_counts = pd.Series(y_train).value_counts().sort_index()
class_display = [display_label(l) for l in class_counts.index]

fig, ax = plt.subplots(figsize=(14, 7))
palette = sns.color_palette("husl", len(class_counts))
ax.barh(class_display, class_counts.values, color=palette, edgecolor="white")
ax.set_xlabel("Number of Training Samples", fontsize=12)
ax.set_title("Class Distribution in Training Set", fontsize=15, fontweight="bold")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_class_distribution.png", dpi=150)
plt.close(fig)
print("Saved → 03_class_distribution.png")

# ────────────────────────────────────────────────────────────────────
# 10.  PLOT 4 — Per-class F1 score
# ────────────────────────────────────────────────────────────────────
per_class_f1 = [report_dict[name]["f1-score"] for name in label_names if name in report_dict]
valid_names  = [name for name in label_names if name in report_dict]

sorted_pairs = sorted(zip(per_class_f1, valid_names), reverse=True)
sorted_f1, sorted_names = zip(*sorted_pairs)

fig, ax = plt.subplots(figsize=(14, 9))
palette2 = sns.color_palette("RdYlGn", len(sorted_f1))
ax.barh(sorted_names, sorted_f1, color=palette2[::-1], edgecolor="white")
ax.axvline(x=test_f1, color="navy", linestyle="--", linewidth=1.5, label=f"Macro F1 = {test_f1:.4f}")
ax.set_xlabel("F1-Score", fontsize=12)
ax.set_title("Per-Class F1-Score (Test Set)", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim(0, 1.05)
ax.grid(axis="x", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_per_class_f1.png", dpi=150)
plt.close(fig)
print("Saved → 04_per_class_f1.png")

# ────────────────────────────────────────────────────────────────────
# 11.  PLOT 5 — Confusion Matrix (top-20 most-confused classes)
# ────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, test_pred, labels=labels)

# pick top-20 classes by support (most examples in test set)
support        = cm.sum(axis=1)
top20_idx      = np.argsort(support)[::-1][:20]
cm_top20       = cm[np.ix_(top20_idx, top20_idx)]
top20_names    = [display_label(labels[i]) for i in top20_idx]

# normalise row-wise (recall per cell)
cm_norm = cm_top20.astype(float) / (cm_top20.sum(axis=1, keepdims=True) + 1e-9)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=top20_names,
    yticklabels=top20_names,
    ax=ax,
    linewidths=0.5,
    cbar_kws={"label": "Recall (row-normalised)"},
    annot_kws={"size": 7},
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix — Top-20 Classes (row-normalised)", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_confusion_matrix_top20.png", dpi=150)
plt.close(fig)
print("Saved → 05_confusion_matrix_top20.png")

# ────────────────────────────────────────────────────────────────────
# 12.  PLOT 6 — Val Accuracy curve (if available)
# ────────────────────────────────────────────────────────────────────
if hasattr(ann, "validation_scores_") and ann.validation_scores_:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ann.validation_scores_, color="#43C59E", linewidth=2, label="Validation Accuracy")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Validation Accuracy per Epoch", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_val_accuracy_curve.png", dpi=150)
    plt.close(fig)
    print("Saved → 06_val_accuracy_curve.png")

# ────────────────────────────────────────────────────────────────────
# 13.  Save model artifact
# ────────────────────────────────────────────────────────────────────
artifact_path = BASE_DIR / "mini_jev_choice.joblib"
joblib.dump({"model": model, "category_names": category_names}, artifact_path)
print(f"\nSaved model → {artifact_path}")

# ────────────────────────────────────────────────────────────────────
# 14.  Quick example prediction
# ────────────────────────────────────────────────────────────────────
test_messages = [
    "My transfer has not arrived",
    "I want to activate my card",
    "What is the exchange rate today?",
    "My card was stolen",
    "I forgot my PIN",
]

print("\n===  Example Predictions  ===")
for msg in test_messages:
    probs = model.predict_proba([msg])[0]
    best_idx   = int(np.argmax(probs))
    best_label = model.named_steps["ann"].classes_[best_idx]
    print(f'  "{msg}"')
    print(f"   → {display_label(best_label)}  (confidence: {probs[best_idx]:.3f})\n")

print("\n✅  All assets saved to:", OUTPUT_DIR)
print("✅  Metrics JSON:", OUTPUT_DIR / "metrics.json")
