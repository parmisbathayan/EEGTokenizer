"""Deterministic English text normalization and grouping."""

import hashlib
import re
import unicodedata


_QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "`": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }
)


def normalize_text(text):
    """Return the canonical form used for matching and split grouping."""

    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.translate(_QUOTE_TRANSLATION).casefold().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def text_hash(text):
    """Hash normalized text so duplicate grouping does not depend on row order."""

    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
