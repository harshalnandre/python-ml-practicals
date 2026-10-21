"""Practical 10: Teaching baseline for text + image-aware fake-news classification.

CSV format:
  text,label,image_path (image_path optional)
Labels may be 0/1 or real/fake. Image features are simple image statistics;
this is a baseline, not a reliable fact-checker. Use lawful images and trusted labels.
"""
import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction import DictVectorizer

def image_stats(path):
    """Extract simple visual statistics, or NaNs if no readable image is available."""
    if not isinstance(path, str) or not path.strip() or not os.path.isfile(path):
        return {"img_available": 0.0, "img_mean_r": np.nan, "img_mean_g": np.nan,
                "img_mean_b": np.nan, "img_std": np.nan}
    try:
        image = np.asarray(Image.open(path).convert("RGB").resize((64, 64)), dtype=float) / 255.0
        return {"img_available": 1.0, "img_mean_r": float(image[:, :, 0].mean()),
                "img_mean_g": float(image[:, :, 1].mean()), "img_mean_b": float(image[:, :, 2].mean()),
                "img_std": float(image.std())}
    except Exception:
        return {"img_available": 0.0, "img_mean_r": np.nan, "img_mean_g": np.nan,
                "img_mean_b": np.nan, "img_std": np.nan}

def demo_data():
    rows = [
        ("Local council approves new public library after budget vote", "real"),
        ("Scientists publish study with methods and data available for review", "real"),
        ("Official agency announces updated public transport timetable", "real"),
        ("Viral post claims a miracle drink cures every disease overnight", "fake"),
        ("Anonymous message says celebrity secretly controls the weather", "fake"),
        ("Fabricated headline claims all banks will permanently close tomorrow", "fake"),
        ("University releases its annual research report", "real"),
        ("Shocking rumour claims ordinary fruit replaces every prescribed treatment", "fake"),
        ("City publishes audited results for its road improvement programme", "real"),
        ("Unverified post claims a hidden planet will hit Earth next week", "fake"),
        ("Public health department shares vaccination clinic dates", "real"),
        ("Click here: unbelievable secret proves every election result is fictional", "fake"),
        ("Court publishes a dated decision and official case summary", "real"),
        ("A post with no source claims a common food instantly makes people invisible", "fake"),
        ("Meteorological office issues a forecast based on current observations", "real"),
        ("Made-up report says the moon disappeared for several hours last night", "fake"),
    ]
    return pd.DataFrame(rows, columns=["text", "label"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="", help="CSV containing text,label and optional image_path columns")
    args = parser.parse_args()
    df = pd.read_csv(args.csv) if args.csv else demo_data()
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise SystemExit("CSV must include columns named 'text' and 'label'.")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    if df["label"].nunique() < 2:
        raise SystemExit("Need at least two distinct labels.")
    has_images = "image_path" in df.columns and df["image_path"].notna().any()
    if has_images:
        image_features = pd.DataFrame([image_stats(p) for p in df["image_path"]])
        # Numeric image features appended as special tokens is not a robust multimodal model;
        # this demo uses text TF-IDF and a separate numeric feature pipeline.
        X_text = df["text"].fillna("")
        X_img = image_features
        y = LabelEncoder().fit_transform(df["label"])
        X_text_train, X_text_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
            X_text, X_img, y, test_size=0.25, random_state=42, stratify=y
        )
        text_model = Pipeline([("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
                               ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        img_model = Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        text_model.fit(X_text_train, y_train)
        img_model.fit(X_img_train, y_train)
        # Combine modality probabilities; equal weighting is a simple late-fusion baseline.
        p_text = text_model.predict_proba(X_text_test)
        p_img = img_model.predict_proba(X_img_test)
        probs = 0.7 * p_text + 0.3 * p_img
        pred = probs.argmax(axis=1)
        print("Mode: text + basic image-statistics late fusion")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
        )
        model = Pipeline([("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
                          ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        y_test = LabelEncoder().fit(df["label"]).transform(y_test)
        print("Mode: text-only demo (no usable image_path column supplied)")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print("Classification report:\n", classification_report(y_test, pred, zero_division=0))
    print("Important: this is a classroom baseline, not a fact-checking system. Verify claims with reliable sources.")

if __name__ == "__main__":
    main()
