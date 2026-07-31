"""Optional sentiment-label lookup for the ZuCo SR task."""

import csv

from .text import normalize_text


def load_label_lookup(path):
    """Return normalized text -> (sentence_id, sentiment_label)."""

    if path is None:
        return {}
    lookup = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sentence", "sentiment_label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"label CSV is missing columns: {sorted(missing)}")
        for ordinal, row in enumerate(reader, start=1):
            key = normalize_text(row["sentence"])
            if not key:
                raise ValueError(f"label row {ordinal} has empty normalized text")
            sentence_id = row.get("sentence_id") or ordinal
            value = (int(sentence_id), int(row["sentiment_label"]))
            previous = lookup.setdefault(key, value)
            if previous != value:
                raise ValueError(f"conflicting labels for normalized text {key!r}")
    return lookup
