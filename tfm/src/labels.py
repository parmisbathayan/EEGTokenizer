"""Sentence-label matching for ZuCo Task 1."""

import re

import pandas as pd


def normalize_text(text):
    text = str(text).lower().strip()
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def load_labels(path):
    labels = pd.read_csv(path)
    required = {"sentence_id", "sentence", "sentiment_label"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"label file is missing columns: {sorted(missing)}")
    labels = labels[["sentence_id", "sentence", "sentiment_label"]].dropna().copy()
    labels["sentence_id"] = labels["sentence_id"].astype(int)
    labels["sentiment_label"] = labels["sentiment_label"].astype(int)
    if labels["sentence_id"].duplicated().any():
        raise ValueError("sentence_id must be unique")
    return labels.sort_values("sentence_id").reset_index(drop=True)


def label_lookup(path):
    labels = load_labels(path)
    return {
        normalize_text(row.sentence): (int(row.sentence_id), int(row.sentiment_label))
        for row in labels.itertuples()
    }

