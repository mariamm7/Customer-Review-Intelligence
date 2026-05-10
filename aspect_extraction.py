import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import spacy
from rake_nltk import Rake
import nltk
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# -------------------------------
# Download stopwords
# -------------------------------
nltk.download("stopwords")

# -------------------------------
# Load ABSA model
# -------------------------------
MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# -------------------------------
# Load spaCy and RAKE
# -------------------------------
nlp = spacy.load("en_core_web_sm")
rake = Rake()

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Lovable AI ABSA API", version="1.0")

# -------------------------------
# Request schema
# -------------------------------
class ReviewRequest(BaseModel):
    reviews: List[str]

# -------------------------------
# Aspect Extraction Functions
# -------------------------------
def extract_aspects_spacy(text: str) -> List[str]:
    doc = nlp(text.lower())
    aspects = set()
    for chunk in doc.noun_chunks:
        modifiers = [tok.text for tok in chunk.root.lefts if tok.dep_ == "amod"]
        aspect_phrase = " ".join(modifiers + [chunk.root.text]).strip()
        aspects.add(aspect_phrase)
    return list(aspects)

def extract_aspects_rake(text: str) -> List[str]:
    rake.extract_keywords_from_text(text.lower())
    return rake.get_ranked_phrases()

def extract_aspects(text: str, top_n: int = 10) -> List[str]:
    aspects_spacy = extract_aspects_spacy(text)
    aspects_rake = extract_aspects_rake(text)
    combined = list(dict.fromkeys(aspects_spacy + aspects_rake))  # deduplicate
    return combined[:top_n] or ["overall"]

# -------------------------------
# Sentiment Analysis Function (single review)
# -------------------------------
def analyze_review(review: str, max_len: int = 128) -> Dict[str, Dict]:
    aspects = extract_aspects(review)
    if not aspects:
        aspects = ["overall"]
    batch_texts = [f"[CLS] {review} [SEP] {aspect}" for aspect in aspects]
    inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    with torch.no_grad():
        outputs = model(**inputs)
        scores_batch = torch.softmax(outputs.logits, dim=1).numpy()
    results = {}
    for aspect, scores in zip(aspects, scores_batch):
        sentiment = ["negative", "neutral", "positive"][np.argmax(scores)]
        confidence = float(np.max(scores))
        results[aspect] = {"sentiment": sentiment, "confidence": confidence}
    return results

# -------------------------------
# Batched Sentiment Analysis (for speed)
# -------------------------------
def analyze_reviews_batched(reviews: List[str], batch_size: int = 32, max_len: int = 128):
    predicted_labels = []
    for i in tqdm(range(0, len(reviews), batch_size), desc="Processing reviews", ncols=100):
        batch = reviews[i:i+batch_size]
        for review in batch:
            result = analyze_review(review, max_len)
            overall_sentiments = [v["sentiment"] for v in result.values()]
            final_sentiment = max(set(overall_sentiments), key=overall_sentiments.count)
            predicted_labels.append(final_sentiment)
    return predicted_labels

# -------------------------------
# FastAPI Endpoint
# -------------------------------
@app.post("/predict")
def predict_sentiment(req: ReviewRequest):
    response = []
    for review in req.reviews:
        result = analyze_review(review)
        overall_sentiments = [v["sentiment"] for v in result.values()]
        final_sentiment = max(set(overall_sentiments), key=overall_sentiments.count)
        response.append({
            "review": review,
            "overall_sentiment": final_sentiment,
            "aspects": result
        })
    return {"predictions": response}

# -------------------------------
# Run predictions on first 1000 reviews
# -------------------------------
if __name__ == "__main__":
    df = pd.read_csv("C:/ChatterLens/data/processed/amazon_reviews_clean.csv")
    n = 1000
    test_reviews = df['clean_review'].iloc[:n].tolist()
    true_labels = df['sentiment'].iloc[:n].tolist()

    predicted_labels = analyze_reviews_batched(test_reviews, batch_size=32)

    # -------------------------------
    # Compute metrics
    # -------------------------------
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="macro", zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)

    print(f"\nEvaluated {n} reviews:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
