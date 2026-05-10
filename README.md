# Customer-Review-Intelligence
An AI pipeline that turns raw customer reviews into actionable business intelligence.

## What It Does
It analyzes customer reviews to answer five real business questions:
- What do customers feel about specific product aspects?
- Which customers are at risk of churning?
- Which customers are showing purchase intent?
- Are customers mentioning competitor products?

## Project Modules

| File | What it does |
|---|---|
| data_preprocessing.py | Cleans raw Amazon review text — removes noise, standardizes format |
| aspect_extraction.py | Extracts product aspects using spaCy + RAKE, classifies sentiment per aspect using DeBERTa |
| churn_model.py | Predicts churn risk using TF-IDF + Random Forest, served via FastAPI |
| purchase_intent_model.py | Detects purchase intent signals from review language |
| competitive_signal.py | Flags competitor brand mentions within reviews |

## Sample Output
```json
{
  "review": "The camera is amazing but battery dies too fast",
  "overall_sentiment": "mixed",
  "aspects": {
    "camera": {"sentiment": "positive", "confidence": 0.94},
    "battery": {"sentiment": "negative", "confidence": 0.91}
  },
  "churn_risk": "NO",
  "purchase_intent": "YES",
  "competitor_mention": "NONE"
}
```

## Tech Stack
- Python
- HuggingFace Transformers (DeBERTa)
- spaCy + RAKE
- scikit-learn (TF-IDF, Random Forest)
- FastAPI
- pandas, numpy

## Dataset
Amazon customer reviews — preprocessed locally. Raw data available on Kaggle.

## Author
Mariam Maqsood | Final Year AI Student | NED University, Karachi
