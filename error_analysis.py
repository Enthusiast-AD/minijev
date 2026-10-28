"""
error_analysis.py
=================
Loads the trained model, runs error analysis:
  - Which intents are most often confused?
  - What do wrong predictions look like?
  - Are there duplicate messages between train and test?
  - Which classes are hardest / easiest?

Run:
    python error_analysis.py
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ────────────────────────────────────────────────────────────────────
# 0.  Paths
# ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "dataset"
OUTPUT_DIR  = BASE_DIR / "report_assets"
OUTPUT_DIR.mkdir(exist_ok=True)

ARTIFACT_PATH = BASE_DIR / "mini_jev_choice.joblib"

# ────────────────────────────────────────────────────────────────────
# 1.  Load model + data
# ────────────────────────────────────────────────────────────────────
print("Loading model...")
artifact    = joblib.load(ARTIFACT_PATH)
model       = artifact["model"]
cat_names   = artifact["category_names"]

def display(label): return cat_names.get(str(label), str(label))

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")

# normalise columns
for df in [train_df, test_df]:
    df.columns = df.columns.str.strip().str.lower()

aliases = {"query": "text", "utterance": "text", "category": "label", "intent": "label"}
train_df = train_df.rename(columns=aliases)
test_df  = test_df.rename(columns=aliases)

train_df = train_df.dropna(subset=["text", "label"]).copy()
test_df  = test_df.dropna(subset=["text", "label"]).copy()
train_df["text"] = train_df["text"].astype(str).str.strip()
test_df["text"]  = test_df["text"].astype(str).str.strip()
train_df = train_df[train_df["text"] != ""]
test_df  = test_df[test_df["text"] != ""]

X_test = test_df["text"].to_numpy()
y_test = test_df["label"].to_numpy()

print(f"Test samples: {len(X_test)}")

# ────────────────────────────────────────────────────────────────────
# 2.  Run predictions
# ────────────────────────────────────────────────────────────────────
print("Running predictions...")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)
classes = model.named_steps["ann"].classes_

# Build results dataframe
results_df = pd.DataFrame({
    "text":       X_test,
    "true_label": y_test,
    "pred_label": y_pred,
    "correct":    y_test == y_pred,
    "confidence": y_proba.max(axis=1).round(4),
    "true_name":  [display(l) for l in y_test],
    "pred_name":  [display(l) for l in y_pred],
})

wrong_df   = results_df[~results_df["correct"]].copy()
correct_df = results_df[results_df["correct"]].copy()

print(f"\nTotal correct   : {len(correct_df)} / {len(results_df)}")
print(f"Total incorrect : {len(wrong_df)} / {len(results_df)}")

# ────────────────────────────────────────────────────────────────────
# 3.  Most confused intent PAIRS
# ────────────────────────────────────────────────────────────────────
print("\n=== TOP 20 MOST CONFUSED PAIRS ===")
confusion_pairs = Counter()
for _, row in wrong_df.iterrows():
    pair = (row["true_name"], row["pred_name"])
    confusion_pairs[pair] += 1

top_pairs = confusion_pairs.most_common(20)
for i, ((true_n, pred_n), count) in enumerate(top_pairs, 1):
    print(f"  {i:2d}. '{true_n}' → predicted as '{pred_n}' ({count}x)")

# ────────────────────────────────────────────────────────────────────
# 4.  Error rate per class
# ────────────────────────────────────────────────────────────────────
class_errors = (
    results_df.groupby("true_name")["correct"]
    .agg(["count", "sum"])
    .rename(columns={"count": "total", "sum": "correct_count"})
)
class_errors["error_count"]  = class_errors["total"] - class_errors["correct_count"]
class_errors["error_rate"]   = (class_errors["error_count"] / class_errors["total"]).round(4)
class_errors["accuracy"]     = (class_errors["correct_count"] / class_errors["total"]).round(4)
class_errors = class_errors.sort_values("error_rate", ascending=False)

print("\n=== CLASSES WITH HIGHEST ERROR RATE ===")
print(class_errors.head(15).to_string())

print("\n=== CLASSES WITH ZERO ERRORS (PERFECT) ===")
perfect = class_errors[class_errors["error_rate"] == 0.0]
print(perfect.to_string())

# ────────────────────────────────────────────────────────────────────
# 5.  Sample wrong predictions to inspect
# ────────────────────────────────────────────────────────────────────
print("\n=== SAMPLE INCORRECT PREDICTIONS ===")
print(f"{'True Intent':<45} {'Predicted':<45} {'Conf':>6}  Text")
print("-" * 140)
for _, row in wrong_df.sort_values("confidence", ascending=False).head(30).iterrows():
    print(f"{row['true_name']:<45} {row['pred_name']:<45} {row['confidence']:>6.3f}  {row['text'][:60]}")

# ────────────────────────────────────────────────────────────────────
# 6.  Duplicate check between train and test
# ────────────────────────────────────────────────────────────────────
print("\n=== DUPLICATE ANALYSIS ===")
train_texts = set(train_df["text"].str.lower().str.strip())
test_texts  = set(test_df["text"].str.lower().str.strip())
overlap     = train_texts & test_texts

print(f"Train unique texts : {len(train_texts)}")
print(f"Test unique texts  : {len(test_texts)}")
print(f"Overlapping texts  : {len(overlap)}")

if overlap:
    print(f"\nSample overlapping messages (first 10):")
    for msg in list(overlap)[:10]:
        train_label = train_df[train_df["text"].str.lower().str.strip() == msg]["label"].values
        test_label  = test_df[test_df["text"].str.lower().str.strip() == msg]["label"].values
        print(f"  Train: {train_label} | Test: {test_label} | Msg: {msg[:70]}")

# Within-train duplicates
train_dup = train_df[train_df.duplicated(subset=["text"], keep=False)]
print(f"\nDuplicate messages within training set: {len(train_dup)}")

# Within-test duplicates
test_dup = test_df[test_df.duplicated(subset=["text"], keep=False)]
print(f"Duplicate messages within test set: {len(test_dup)}")

# ────────────────────────────────────────────────────────────────────
# 7.  Confidence analysis
# ────────────────────────────────────────────────────────────────────
print("\n=== CONFIDENCE ANALYSIS ===")
print(f"Correct predictions avg confidence   : {correct_df['confidence'].mean():.4f}")
print(f"Incorrect predictions avg confidence : {wrong_df['confidence'].mean():.4f}")
print(f"Wrong predictions with high conf (>0.7): {(wrong_df['confidence'] > 0.7).sum()}")
print(f"Wrong predictions with low conf (<0.3) : {(wrong_df['confidence'] < 0.3).sum()}")

# ────────────────────────────────────────────────────────────────────
# 8.  PLOT — Top confused pairs (horizontal bar)
# ────────────────────────────────────────────────────────────────────
top15_pairs = top_pairs[:15]
pair_labels = [f"{t}\n→ {p}" for (t, p), _ in top15_pairs]
pair_counts = [c for _, c in top15_pairs]

fig, ax = plt.subplots(figsize=(12, 8))
colors  = sns.color_palette("flare", len(pair_labels))
ax.barh(pair_labels[::-1], pair_counts[::-1], color=colors, edgecolor="white")
ax.set_xlabel("Number of confusions", fontsize=12)
ax.set_title("Top 15 Most Confused Intent Pairs", fontsize=15, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
for i, v in enumerate(pair_counts[::-1]):
    ax.text(v + 0.1, i, str(v), va="center", fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "07_top_confused_pairs.png", dpi=150)
plt.close(fig)
print("\nSaved → 07_top_confused_pairs.png")

# ────────────────────────────────────────────────────────────────────
# 9.  PLOT — Error rate per class (top 20 worst)
# ────────────────────────────────────────────────────────────────────
top20_worst = class_errors.head(20)

fig, ax = plt.subplots(figsize=(12, 8))
pal = sns.color_palette("RdYlGn_r", len(top20_worst))
ax.barh(top20_worst.index[::-1], top20_worst["error_rate"][::-1], color=pal, edgecolor="white")
ax.set_xlabel("Error Rate (proportion of test samples misclassified)", fontsize=12)
ax.set_title("Top 20 Classes with Highest Error Rate", fontsize=15, fontweight="bold")
ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1, alpha=0.5, label="50% error rate")
ax.set_xlim(0, 1)
ax.grid(axis="x", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "08_error_rate_per_class.png", dpi=150)
plt.close(fig)
print("Saved → 08_error_rate_per_class.png")

# ────────────────────────────────────────────────────────────────────
# 10.  PLOT — Confidence distribution: correct vs incorrect
# ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(correct_df["confidence"], bins=30, alpha=0.7, color="#43C59E", label="Correct predictions", edgecolor="white")
ax.hist(wrong_df["confidence"],   bins=30, alpha=0.7, color="#F76F6E", label="Incorrect predictions", edgecolor="white")
ax.set_xlabel("Model Confidence Score", fontsize=13)
ax.set_ylabel("Number of Predictions", fontsize=13)
ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "09_confidence_distribution.png", dpi=150)
plt.close(fig)
print("Saved → 09_confidence_distribution.png")

# ────────────────────────────────────────────────────────────────────
# 11.  Save full error report JSON
# ────────────────────────────────────────────────────────────────────
error_report = {
    "summary": {
        "total_test"      : int(len(results_df)),
        "correct"         : int(len(correct_df)),
        "incorrect"       : int(len(wrong_df)),
        "accuracy"        : round(len(correct_df) / len(results_df), 4),
        "avg_conf_correct": round(float(correct_df["confidence"].mean()), 4),
        "avg_conf_wrong"  : round(float(wrong_df["confidence"].mean()), 4),
        "duplicate_train_test_overlap": int(len(overlap)),
        "within_train_duplicates"     : int(len(train_dup)),
        "within_test_duplicates"      : int(len(test_dup)),
    },
    "top_confused_pairs": [
        {"true": t, "predicted_as": p, "count": int(c)}
        for (t, p), c in top_pairs
    ],
    "worst_classes": class_errors.head(15)[["error_rate", "error_count", "total", "accuracy"]].to_dict(orient="index"),
    "perfect_classes": list(perfect.index.tolist()),
    "sample_wrong_predictions": [
        {
            "text":     row["text"],
            "true":     row["true_name"],
            "predicted":row["pred_name"],
            "confidence": float(row["confidence"]),
        }
        for _, row in wrong_df.sort_values("confidence", ascending=False).head(50).iterrows()
    ],
}

with open(OUTPUT_DIR / "error_analysis.json", "w") as f:
    json.dump(error_report, f, indent=2)

print("Saved → error_analysis.json")
print("\n✅ Error analysis complete!")
print(f"   Accuracy: {error_report['summary']['accuracy']:.4f}")
print(f"   Errors  : {error_report['summary']['incorrect']} / {error_report['summary']['total_test']}")
print(f"   Train-Test duplicates: {len(overlap)}")
