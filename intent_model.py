# src/intent_api.py

import os
import json
import pandas as pd
from tqdm import tqdm
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
DATA_PATH = r"C:/ChatterLens/data/processed/amazon_reviews_clean.csv"
MODEL_DIR = r"C:/ChatterLens/models/intent_classification_model"

MODEL_FILE = os.path.join(MODEL_DIR, "intent_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
INTENT_MAP_FILE = os.path.join(MODEL_DIR, "intent_map.json")

# ----------------------------------------------------
# Load model + vectorizer + intent map
# ----------------------------------------------------
print("Loading model...")
clf = joblib.load(MODEL_FILE)

print("Loading vectorizer...")
vectorizer = joblib.load(VECTORIZER_FILE)

print("Loading intent map...")
with open(INTENT_MAP_FILE, "r") as f:
    intent_map = json.load(f)

# reverse map (id → label)
id2label = {int(k): v for k, v in intent_map.items()}

# ----------------------------------------------------
# AUTO-LABEL FUNCTION (same as training)
# ----------------------------------------------------
INTENTS = {
    "praise": ["love", "great", "perfect", "excellent", "amazing"],
    "complaint": ["bad", "poor", "terrible", "disappointed", "waste"],
    "question": ["how", "what", "why", "where", "when", "does"],
    "refund_request": ["refund", "return", "replace", "money back", "exchange"],
    "product_missing_parts": ["missing", "parts", "broken", "piece"],
    "product_damage": ["damage", "scratched", "cracked", "broken"],
    "delivery_issue": ["late", "delay", "shipping", "arrival", "courier"],
    "packaging_issue": ["packaging", "box", "wrapped", "open"],
    "warranty_claim": ["warranty", "guarantee", "support", "service"],
    "recommendation": ["recommend", "suggest", "advice", "buy again"],
    "feature_request": ["feature", "option", "add", "improve"],
    "size_issue": ["size", "fit", "small", "large", "tight", "loose"],
    "price_comment": ["expensive", "cheap", "price", "worth"],
    "usability_issue": ["difficult", "hard", "use", "operate", "setup"],
    "comparison": ["better than", "worse than", "compared", "similar"]
}

def auto_label_intent(text):
    text_lower = str(text).lower()
    for intent, keywords in INTENTS.items():
        if any(kw in text_lower for kw in keywords):
            return intent
    return "other"

# ----------------------------------------------------
# FastAPI App
# ----------------------------------------------------
app = FastAPI(title="Intent Classification API")

class IntentRequest(BaseModel):
    texts: list

# Prediction function
def predict_intent(texts):
    X = vectorizer.transform(texts)
    preds = clf.predict(X)
    return [id2label[int(p)] for p in preds]

# ----------------------------------------------------
# API Endpoint for GUI
# ----------------------------------------------------
@app.post("/predict_intent")
def predict(payload: IntentRequest):
    preds = predict_intent(payload.texts)
    return {"predictions": preds}

# ----------------------------------------------------
# SELF-VALIDATION: TEST FIRST 1000 REVIEWS
# ----------------------------------------------------
print("\n===== LOADING DATASET FOR SELF-VALIDATION =====")
df = pd.read_csv(DATA_PATH)

test_df = df.head(1000).copy()
texts = test_df["clean_review"].tolist()

print("\n===== GENERATING PSEUDO-GROUND-TRUTH USING AUTO LABELS =====")
true_labels = [auto_label_intent(t) for t in texts]

print("\n===== MODEL PREDICTING 1000 REVIEWS IN BATCHES =====")
batch_preds = []
batch_size = 64

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_preds.extend(predict_intent(batch))

# ----------------------------------------------------
# METRICS
# ----------------------------------------------------
print("\n===== SELF-CHECK METRICS (Model vs Auto-Labels) =====")

acc = accuracy_score(true_labels, batch_preds)
print("Accuracy:", round(acc, 4))

print("\nClassification Report:")
print(classification_report(true_labels, batch_preds))

print("\n🚀 Intent API ready. Launch using:  uvicorn src.intent_api:app --reload")
