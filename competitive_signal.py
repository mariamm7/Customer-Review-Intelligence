# src/competitive_signal_service.py

import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from textblob import TextBlob
from tqdm import tqdm

tqdm.pandas()

# -------------------------------
# CONFIG
# -------------------------------
MODEL_DIR = "models/ner_model"
MODEL_NAME_NER = "dbmdz/bert-large-cased-finetuned-conll03-english"
DATA_PATH = r"C:/ChatterLens/data/processed/amazon_reviews_clean.csv"  # optional batch test
BATCH_TEST_SIZE = 1000

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI(title="ChatterLens Competitive Signal API")

class ReviewRequest(BaseModel):
    review: str

class BatchReviewRequest(BaseModel):
    reviews: List[str]

# -------------------------------
# MODEL LOADING / SAVING
# -------------------------------
def load_or_download_ner_model():
    if not os.path.exists(MODEL_DIR):
        print("Downloading and saving NER model locally...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_NER)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME_NER)
        tokenizer.save_pretrained(MODEL_DIR)
        model.save_pretrained(MODEL_DIR)
    else:
        print("Loading saved NER model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    
    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    print("NER model ready.\n")
    return ner_pipe

ner_pipeline = load_or_download_ner_model()

# -------------------------------
# FUNCTIONS
# -------------------------------
def extract_competitors_ner(text: str) -> str:
    """Extract ORG entities from text using NER model."""
    ner_results = ner_pipeline(str(text))
    competitors = [ent['word'] for ent in ner_results if ent['entity_group'] == 'ORG']
    return ", ".join(competitors) if competitors else "None"

def get_sentiment(text: str) -> str:
    """Return sentiment of the text using TextBlob."""
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

# -------------------------------
# BATCH TESTING ON REVIEWS CSV
# -------------------------------
def batch_test_reviews():
    if not os.path.exists(DATA_PATH):
        print("No dataset found for batch testing.")
        return
    
    df = pd.read_csv(DATA_PATH)
    if len(df) < BATCH_TEST_SIZE:
        test_df = df.copy()
    else:
        test_df = df.sample(BATCH_TEST_SIZE, random_state=42)

    results = []
    print(f"Running batch test on {len(test_df)} reviews...")
    for review in tqdm(test_df["clean_review"].tolist(), desc="Processing"):
        competitor = extract_competitors_ner(review)
        sentiment = get_sentiment(review)
        results.append({
            "review": review,
            "mentioned_competitor": competitor,
            "review_sentiment": sentiment
        })
    
    print("Batch testing completed.\n")
    return results

# -------------------------------
# FASTAPI ENDPOINTS
# -------------------------------
@app.post("/analyze_review")
def analyze_review(request: ReviewRequest):
    competitor = extract_competitors_ner(request.review)
    sentiment = get_sentiment(request.review)
    return {
        "mentioned_competitor": competitor,
        "review_sentiment": sentiment
    }

@app.post("/analyze_reviews_batch")
def analyze_reviews_batch(request: BatchReviewRequest):
    results = []
    for review in request.reviews:
        competitor = extract_competitors_ner(review)
        sentiment = get_sentiment(review)
        results.append({
            "review": review,
            "mentioned_competitor": competitor,
            "review_sentiment": sentiment
        })
    return {"results": results}

@app.get("/")
def root():
    return {"message": "ChatterLens Competitive Signal API is running!"}

# -------------------------------
# MAIN ENTRY (for direct testing)
# -------------------------------
if __name__ == "__main__":
    # Test single sample
    sample_text = "I love the features of Apple but Samsung is catching up fast."
    print("Competitors:", extract_competitors_ner(sample_text))
    print("Sentiment:", get_sentiment(sample_text))

    # Optional: batch test from CSV
    batch_results = batch_test_reviews()
    if batch_results:
        print(f"Sample batch result: {batch_results[:3]}")
