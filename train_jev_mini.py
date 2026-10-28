
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. Load local dataset files
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
CATEGORIES_PATH = DATA_DIR / "categories.json"



for path in [TRAIN_PATH, TEST_PATH, CATEGORIES_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


with open(CATEGORIES_PATH, "r", encoding="utf-8") as file:
    categories = json.load(file)

print("Training columns:", train_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())
print("Category JSON type:", type(categories).__name__)


# Normalize column names
train_df.columns = train_df.columns.str.strip().str.lower()
test_df.columns = test_df.columns.str.strip().str.lower()

# Support common alternative column names
column_aliases = {
    "query": "text",
    "utterance": "text",
    "category": "label",
    "intent": "label",
}

train_df = train_df.rename(columns=column_aliases)
test_df = test_df.rename(columns=column_aliases)

required_columns = {"text", "label"}

if not required_columns.issubset(train_df.columns):
    raise ValueError(
        f"train.csv needs columns {required_columns}; "
        f"found {train_df.columns.tolist()}"
    )

if not required_columns.issubset(test_df.columns):
    raise ValueError(
        f"test.csv needs columns {required_columns}; "
        f"found {test_df.columns.tolist()}"
    )

# Drop missing and empty examples
train_df = train_df.dropna(subset=["text", "label"]).copy()
test_df = test_df.dropna(subset=["text", "label"]).copy()

train_df["text"] = train_df["text"].astype(str).str.strip()
test_df["text"] = test_df["text"].astype(str).str.strip()

train_df = train_df[train_df["text"] != ""]
test_df = test_df[test_df["text"] != ""]

X = train_df["text"].to_numpy()
y = train_df["label"].to_numpy()

X_test = test_df["text"].to_numpy()
y_test = test_df["label"].to_numpy()

print("\nTraining examples:", len(X))
print("Test examples:", len(X_test))
print("Training classes:", len(np.unique(y)))
print("Test classes:", len(np.unique(y_test)))


# --------------------------------------------------
# 2. Prepare category names
# --------------------------------------------------

def normalize_categories(data):
    """
    Convert common category JSON structures into
    a dictionary mapping label IDs/names to names.
    """
    if isinstance(data, dict):
        # Example: {"0": "activate_my_card", ...}
        if all(isinstance(v, str) for v in data.values()):
            return {str(k): v for k, v in data.items()}

        # Example: {"categories": {"0": "intent", ...}}
        for key in ("categories", "labels", "intents"):
            if key in data:
                return normalize_categories(data[key])

        # Example: {"activate_my_card": 0, ...}
        if all(isinstance(v, (int, float)) for v in data.values()):
            return {str(v): str(k) for k, v in data.items()}

    if isinstance(data, list):
        # Example: ["activate_my_card", "cash_withdrawal", ...]
        return {str(i): name for i, name in enumerate(data)}

    raise ValueError(
        "Unsupported categories.json format. "
        "Print the JSON structure and adapt normalize_categories()."
    )


category_names = normalize_categories(categories)

# If labels are already strings, they may themselves be the
# actual intent names, so use them as a fallback.
for label in np.unique(y):
    category_names.setdefault(str(label), str(label))


def display_label(label):
    return category_names.get(str(label), str(label))


# --------------------------------------------------
# 3. Split training data into train and validation
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# 4. Define TF-IDF + ANN model
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Train
# --------------------------------------------------

print("\nTraining mini Jev Choice model...")
model.fit(X_train, y_train)


# --------------------------------------------------
# 6. Validation evaluation
# --------------------------------------------------

val_pred = model.predict(X_val)

print("\n--- Validation results ---")
print("Accuracy:", accuracy_score(y_val, val_pred))
print(
    "Macro-F1:",
    f1_score(y_val, val_pred, average="macro", zero_division=0),
)


# --------------------------------------------------
# 7. Final test evaluation
# --------------------------------------------------

test_pred = model.predict(X_test)

print("\n--- Final test results ---")
print("Accuracy:", accuracy_score(y_test, test_pred))
print(
    "Macro-F1:",
    f1_score(y_test, test_pred, average="macro", zero_division=0),
)

labels = model.named_steps["ann"].classes_

print("\nClassification report:")
print(classification_report(
    y_test,
    test_pred,
    labels=labels,
    target_names=[display_label(label) for label in labels],
    zero_division=0,
))


# --------------------------------------------------
# 8. Save model and category metadata
# --------------------------------------------------

artifact_path = BASE_DIR / "mini_jev_choice.joblib"

joblib.dump({
    "model": model,
    "category_names": category_names,
}, artifact_path)

print(f"\nSaved model to: {artifact_path}")


# --------------------------------------------------
# 9. Try an example prediction
# --------------------------------------------------

message = "My transfer has not arrived"

probabilities = model.predict_proba([message])[0]
best_index = int(np.argmax(probabilities))
best_label = model.named_steps["ann"].classes_[best_index]

print("\n--- Example prediction ---")
print("Input:", message)
print("Choice:", display_label(best_label))
print("Class ID:", str(best_label))
print("Model probability:", round(float(probabilities[best_index]), 4))
