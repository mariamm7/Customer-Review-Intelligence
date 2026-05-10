import os
import re
import string
import bz2
from typing import List
import sys

import pandas as pd
from tqdm import tqdm
tqdm.pandas()  # Enable progress_apply for pandas DataFrames

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Optional: spaCy for better NER anonymization and POS-aware lemmatization
try:
    import spacy
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm")
    except Exception:
        _SPACY_NLP = None
except Exception:
    _SPACY_NLP = None


def _ensure_nltk_resources() -> None:
    """Download minimal NLTK resources if missing."""
    # include punkt_tab as some NLTK tokenizer variants request it
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    print("⏳ Checking NLTK resources...")
    for r in resources:
        try:
            nltk.data.find(r)
            print("  ✓ " + r)
        except LookupError:
            print("  ⬇️  Downloading " + r + "...")
            nltk.download(r, quiet=True)
            print("  ✓ " + r)


_ensure_nltk_resources()

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def load_data(file_path: str, max_reviews: int = None) -> pd.DataFrame:
    """Load fastText-style bz2 file into a DataFrame with columns ['review', 'sentiment'].

    Expected input lines like: "__label__1 This is a review text..."
    
    Args:
        file_path: path to the bz2 file
        max_reviews: limit number of reviews to load (None = load all)
    """
    print("📂 Loading data from: " + file_path)
    with bz2.open(file_path, "rt", encoding="utf-8") as f:
        lines = f.readlines()

    # Limit to max_reviews if specified
    if max_reviews is not None:
        lines = lines[:max_reviews]
        print("⚡ Limited to " + str(max_reviews) + " reviews")

    data = []
    for line in tqdm(lines, desc="Parsing lines"):
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            try:
                label = int(parts[0].replace("__label__", ""))
            except ValueError:
                # skip malformed lines
                continue
            text = parts[1]
            data.append((text, label))

    df = pd.DataFrame(data, columns=["review", "label"]) if data else pd.DataFrame(columns=["review", "label"])
    df["sentiment"] = df["label"].map({1: "negative", 2: "positive"})
    df = df.drop(columns=["label"])
    return df


def anonymize_text(text: str) -> str:
    """Simple regex-based anonymization for emails, phones, urls, handles, and numbers."""
    if not isinstance(text, str):
        return text
    # If spaCy NER is available, use it for more robust anonymization
    if _SPACY_NLP is not None:
        try:
            doc = _SPACY_NLP(text)
            # replace named entities of interest with a generic token
            entities_of_interest = {"PERSON", "GPE", "LOC", "ORG", "NORP", "FAC", "PRODUCT"}
            new_text = text
            # iterate reversed so replacements don't shift offsets
            for ent in reversed(list(doc.ents)):
                if ent.label_ in entities_of_interest:
                    start, end = ent.start_char, ent.end_char
                    new_text = new_text[:start] + f"<{ent.label_}>" + new_text[end:]
            # still run regex-based anonymization for emails/phones/urls
            text = new_text
        except Exception:
            # fallback to regex anonymization below
            pass

    # Emails
    text = re.sub(r"[a-zA-Z0-9.+_-]+@[a-zA-Z0-9._-]+\.[a-zA-Z]+", "<EMAIL>", text)
    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
    # Handles like @username
    text = re.sub(r"@\w+", "<HANDLE>", text)
    # Phone numbers (simple patterns)
    text = re.sub(r"\+?\d[\d\-() ]{7,}\d", "<PHONE>", text)
    # Long digit sequences (credit cards, ids)
    text = re.sub(r"\b\d{6,}\b", "<NUM>", text)
    return text


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, remove HTML tags, anonymize, remove punctuation, normalize whitespace."""
    if not isinstance(text, str):
        return ""

    t = text.lower()
    # remove HTML tags
    t = re.sub(r"<[^>]*>", "", t)
    # anonymize before stripping punctuation/urls
    t = anonymize_text(t)
    # remove remaining URLs (redundant but safe)
    t = re.sub(r"https?://\S+|www\.\S+", "", t)
    # remove punctuation
    t = t.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize_with_spacy(text: str, remove_stopwords: bool) -> List[str] | None:
    """Attempt to tokenize using spaCy, return None on failure so caller can fallback."""
    try:
        doc = _SPACY_NLP(text)
    except Exception:
        return None

    toks: List[str] = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        if remove_stopwords and token.is_stop:
            continue
        lemma = token.lemma_.lower().strip()
        if lemma:
            toks.append(lemma)
    return toks


def _tokenize_with_nltk(text: str, remove_stopwords: bool) -> List[str]:
    """Tokenize using NLTK and lemmatize tokens."""
    tokens = word_tokenize(text)
    normalized: List[str] = []
    for tok in tokens:
        tok = tok.lower()
        if remove_stopwords and tok in _STOPWORDS:
            continue
        # lemmatize; for verbs we could POS-tag, but keep simple
        lemma = _LEMMATIZER.lemmatize(tok)
        if lemma:
            normalized.append(lemma)
    return normalized


def tokenize_normalize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize text, remove stopwords (optional), and lemmatize tokens."""
    if not isinstance(text, str) or not text:
        return []

    # Prefer spaCy if available; fallback to NLTK on any issue
    if _SPACY_NLP is not None:
        spacy_result = _tokenize_with_spacy(text, remove_stopwords)
        if spacy_result is not None:
            return spacy_result

    return _tokenize_with_nltk(text, remove_stopwords)


def preprocess_and_save(raw_path: str, save_path: str, max_reviews: int = None) -> None:
    """Run the full preprocessing pipeline and save results to CSV.

    Adds the following columns to the output CSV:
    - clean_review: cleaned and anonymized string
    - tokens: token list (stored as space-joined string)
    
    Args:
        raw_path: path to input bz2 file
        save_path: path to output CSV file
        max_reviews: limit number of reviews to process (None = all)
    """
    print("🚀 Starting preprocessing...")
    df = load_data(raw_path, max_reviews=max_reviews)
    print("✅ Loaded " + str(len(df)) + " reviews")

    # Clean and anonymize text
    print("🧹 Cleaning reviews...")
    df["clean_review"] = df["review"].progress_apply(clean_text)

    # Tokenize, remove stopwords and lemmatize
    print("🔤 Tokenizing and lemmatizing...")
    df["tokens"] = df["clean_review"].progress_apply(lambda t: tokenize_normalize(t, remove_stopwords=True))

    # Save tokens as a space-joined string for CSV friendliness
    print("💾 Preparing output...")
    df["tokens_text"] = df["tokens"].apply(lambda toks: " ".join(toks))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8")
    print("💾 Saved cleaned dataset to " + save_path)


if __name__ == "__main__":
    raw_path = r"C:/ChatterLens/data/raw/test.ft.txt.bz2"
    save_path = r"C:/ChatterLens/data/processed/amazon_reviews_clean.csv"
    
    # Limit to 100K reviews for now, scale later
    max_reviews = 100000
    
    preprocess_and_save(raw_path, save_path, max_reviews=max_reviews)
