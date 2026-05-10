import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Enable tqdm for pandas
tqdm.pandas()

# ---------------------------
# CONFIG
# ---------------------------
DATA_PATH = r"C:/ChatterLens/data/processed/amazon_reviews_clean.csv"
MODEL_DIR = r"C:/ChatterLens/models/churn_model"
MODEL_FILE = os.path.join(MODEL_DIR, "churn_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

clf = None
vectorizer = None

# ---------------------------
# TRAINING PIPELINE
# ---------------------------

def train_and_save_model():
    global clf, vectorizer

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} samples\n")

    print("Creating churn labels...")
    df['churn'] = df['sentiment'].progress_apply(
        lambda x: 1 if str(x).lower() == "negative" else 0
    )

    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df["clean_review"])
    y = df["churn"]

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForest classifier...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    print("\n---------------------------")
    print("MODEL EVALUATION")
    print("---------------------------\n")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nSaving model and vectorizer...")
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print("Model saved successfully.\n")


def load_model():
    global clf, vectorizer

    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        print("Loading saved churn model...")
        clf = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        print("Model loaded.\n")
    else:
        print("Model not found → Training new churn model...")
        train_and_save_model()


def predict_churn(text_list):
    features = vectorizer.transform(text_list)
    preds = clf.predict(features)
    return preds.tolist()


# ---------------------------
# BATCH TESTING ON 1000 REVIEWS
# ---------------------------
def test_on_1000_reviews():
    print("\n---------------------------")
    print("RUNNING 1000-REVIEW BATCH TEST")
    print("---------------------------\n")

    df = pd.read_csv(DATA_PATH)

    if len(df) < 1000:
        print("Dataset has fewer than 1000 samples — testing on full dataset.\n")
        test_df = df.copy()
    else:
        test_df = df.sample(1000, random_state=42)

    predictions = []
    
    for review in tqdm(test_df["clean_review"].tolist(), desc="Predicting"):
        pred = predict_churn([review])[0]
        predictions.append(pred)

    churn_rate = (sum(predictions) / len(predictions)) * 100

    print(f"\nTotal reviews tested: {len(predictions)}")
    print(f"Churn Risk in batch: {churn_rate:.2f}%\n")


# ---------------------------
# FASTAPI ENDPOINT
# ---------------------------
app = FastAPI(title="Churn Prediction API")

class ReviewItem(BaseModel):
    review: str

@app.post("/predict_churn")
def churn_api(item: ReviewItem):
    pred = predict_churn([item.review])[0]
    return {
        "churn_prediction": int(pred),
        "churn_risk": "YES" if pred == 1 else "NO"
    }


# ---------------------------
# MAIN ENTRY
# ---------------------------
if __name__ == "__main__":
    load_model()
    test_on_1000_reviews()
