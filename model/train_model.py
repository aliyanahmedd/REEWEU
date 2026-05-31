"""
train_model.py
================
Trains the ReviewGuard fake-review detection model.

Pipeline:
    raw review text  ->  cleaning  ->  TF-IDF features  ->  Random Forest  ->  label

Run this ONCE before starting the web app:
    python model/train_model.py

It produces three .pkl files used by app.py at prediction time:
    - review_model.pkl     (the trained Random Forest classifier)
    - tfidf_vectorizer.pkl (the fitted TF-IDF vectorizer)
    - label_encoder.pkl    (maps class names <-> numbers)
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ----------------------------------------------------------------------------
# Paths — everything is resolved relative to this file so the script can be run
# from any working directory.
# ----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(THIS_DIR, "..", "data", "reviews.csv")
MODEL_PKL = os.path.join(THIS_DIR, "review_model.pkl")
TFIDF_PKL = os.path.join(THIS_DIR, "tfidf_vectorizer.pkl")
ENCODER_PKL = os.path.join(THIS_DIR, "label_encoder.pkl")


# Fallback: if the nltk stopwords corpus is missing, download it automatically.
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

STEMMER = PorterStemmer()


def clean_text(text):
    """
    Cleans a single review string so the model sees consistent input.

    Steps: lowercase -> strip HTML -> remove special chars -> remove stopwords
    -> Porter stemming -> collapse extra whitespace. The SAME cleaning must be
    applied at prediction time (app.py) or the TF-IDF features won't line up.
    """
    text = str(text).lower()                       # lowercase everything
    text = re.sub(r"<.*?>", " ", text)             # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters/spaces
    words = text.split()
    # drop stopwords ("the", "is", ...) then stem ("running" -> "run")
    words = [STEMMER.stem(w) for w in words if w not in STOPWORDS and len(w) > 1]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def load_dataset():
    """
    Loads reviews.csv and normalises it to two columns: text + label.

    The Kaggle "Amazon Fake Reviews" dataset uses a few different column names
    across versions, so we try the common ones. Labels are mapped to the three
    ReviewGuard classes: GENUINE / FAKE / SUSPICIOUS.
    """
    if not os.path.exists(DATA_CSV):
        print(f"[ERROR] Dataset not found at {DATA_CSV}")
        print("Download the 'Amazon Fake Reviews Dataset' from Kaggle and place")
        print("it as data/reviews.csv (needs a text column and a label column).")
        sys.exit(1)

    df = pd.read_csv(DATA_CSV)
    cols = {c.lower(): c for c in df.columns}

    # find the text column (try common names)
    text_col = next((cols[c] for c in
                     ("text", "review", "review_text", "text_", "body")
                     if c in cols), None)
    # find the label column
    label_col = next((cols[c] for c in
                      ("label", "labels", "class", "rating_label", "category")
                      if c in cols), None)

    if text_col is None or label_col is None:
        print(f"[ERROR] Could not find text/label columns. Found: {list(df.columns)}")
        sys.exit(1)

    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]

    # Normalise raw labels into GENUINE / FAKE / SUSPICIOUS.
    def map_label(v):
        s = str(v).strip().lower()
        if s in ("cg", "fake", "1", "computer-generated", "deceptive", "bot"):
            return "FAKE"
        if s in ("or", "genuine", "0", "original", "real", "truthful"):
            return "GENUINE"
        if s in ("suspicious", "2", "borderline"):
            return "SUSPICIOUS"
        # default heuristic: anything starting with 'f' = fake, else genuine
        return "FAKE" if s.startswith("f") else "GENUINE"

    df["label"] = df["label"].apply(map_label)
    return df


def main():
    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} reviews. Class distribution:")
    print(df["label"].value_counts())

    print("\nCleaning text (this may take a minute on 20K rows)...")
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() > 0]

    # --- Label encoding: turn class names into numbers for sklearn ---
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])

    # --- TF-IDF: converts review text to numbers based on word frequency.
    # Common words get a lower weight, rare/informative words get a higher
    # weight. ngram_range=(1,2) also captures two-word phrases like "love it".
    print("\nVectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X = vectorizer.fit_transform(df["clean"])

    # Train/test split so we can measure real accuracy on unseen reviews.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Random Forest: an ensemble of many decision trees that vote. It is
    # more accurate than a single tree, resists overfitting, and handles
    # high-dimensional text features well. class_weight='balanced' stops the
    # model from ignoring the smaller classes.
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 50)
    print(f"ACCURACY: {acc * 100:.2f}%")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Persist everything to disk for app.py to load later ---
    joblib.dump(model, MODEL_PKL)
    joblib.dump(vectorizer, TFIDF_PKL)
    joblib.dump(encoder, ENCODER_PKL)

    # Save the accuracy so the dashboard can display it.
    with open(os.path.join(THIS_DIR, "accuracy.txt"), "w") as f:
        f.write(f"{acc * 100:.2f}")

    print(f"\nSaved model    -> {MODEL_PKL}")
    print(f"Saved TF-IDF   -> {TFIDF_PKL}")
    print(f"Saved encoder  -> {ENCODER_PKL}")
    print("\nDone! You can now run: python app.py")


if __name__ == "__main__":
    main()
