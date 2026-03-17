# Reddit Sentiment Analysis for Stock Movement Prediction 📈🤖

A web mining and NLP project that analyzes Reddit posts to predict daily stock return direction using a **multi-model sentiment fusion pipeline** combining an LLM, XGBoost, and Logistic Regression.

---

## 📌 Problem Statement

Can Reddit sentiment predict stock return direction? With the rise of retail investor communities like **r/wallStreetBets**, Reddit has become a powerful driver of market behavior. This project builds a sentiment-driven stock forecasting pipeline by mining Reddit posts related to 7 major tickers and combining multiple NLP models to predict whether a stock goes **up**, **down**, or stays **neutral** on a given day.

---

## 🎯 Objectives

- Extract and preprocess Reddit posts for 7 stock tickers
- Train sentiment classifiers (XGBoost + Logistic Regression) on the Sentiment140 dataset
- Apply an LLM for context-aware sentiment classification
- Fuse predictions from all 3 models using a rule-based strategy
- Merge sentiment labels with daily stock return data
- Predict stock return direction using an XGBoost classifier

---

## 📁 Project Structure
```
reddit-sentiment-stock/
├── notebooks/
│   └── web_mining_final_project.ipynb   # Full pipeline: preprocessing, modeling, results
├── reports/
│   ├── Final_Report_Group_7.pdf         # Full written report
│   └── Presentation.pptx               # Presentation slides
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| Reddit Stock Posts | Kaggle | Posts for AAPL, TSLA, GME, NFLX, MSFT, MCD, NVDA |
| Sentiment140 | Kaggle | 1.6M labeled tweets (positive/negative) for classifier training |
| Stock Price Data | Stooq API (via pandas-datareader) | Daily OHLC prices Jan 2018 – Jan 2023 |

---

## 🤖 Models & Pipeline

### Stage 1 — Sentiment Classification (3 models)

| Model | Role | Strength |
|-------|------|---------|
| **LLM** (FLAN-T5 / RoBERTa) | Context-aware sentiment | Handles sarcasm, slang, financial jargon |
| **XGBoost** + TF-IDF | Trained on Sentiment140 | Non-linear patterns, high precision |
| **Logistic Regression** + TF-IDF | Trained on Sentiment140 | Fast, interpretable baseline |

### Stage 2 — Fusion Strategy
```
All 3 disagree      → Discard post
Any predicts neutral → Final = Neutral
2/3 agree           → Final = that label
```

### Stage 3 — Stock Return Prediction
- XGBoost classifier trained on fused sentiment + post text
- Target: daily return direction (up / down / neutral)
- Features: TF-IDF vectors + sentiment label

---

## 📈 Results

### Logistic Regression (Sentiment140 Training)
| Metric | Score |
|--------|-------|
| Accuracy | **77%** |
| Precision (macro avg) | 0.77 |
| Recall (macro avg) | 0.77 |
| F1-Score (macro avg) | 0.77 |

### Stock Return Prediction (XGBoost)
| Metric | Score |
|--------|-------|
| Overall Accuracy | **66%** |
| Macro Avg Precision | 64% |
| Macro Avg Recall | 65% |
| Macro Avg F1-Score | 62% |

> ⚠️ Model struggles with minority classes (up/down) due to class imbalance — most posts and returns are neutral.

---

## ⚙️ Methodology

1. **Data Collection** — Reddit posts (Kaggle) + Sentiment140 + Stooq stock prices
2. **Preprocessing** — Text cleaning, TF-IDF vectorization, UNIX timestamp parsing
3. **Sentiment Training** — XGBoost & Logistic Regression on Sentiment140
4. **Sentiment Prediction** — All 3 models label each Reddit post
5. **Fusion** — Rule-based merging of 3 predictions into 1 final label
6. **Stock Alignment** — Match posts to daily return direction by ticker + date
7. **Return Prediction** — XGBoost classifier on final labeled dataset
8. **Real-Time Module** — PRAW + RoBERTa + XGBoost for live predictions

---

## 🔧 Requirements
```bash
pip install pandas numpy scikit-learn xgboost transformers praw pandas-datareader matplotlib wordcloud
```

---

## 🚀 Getting Started
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/reddit-sentiment-stock.git
cd reddit-sentiment-stock

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebooks/web_mining_final_project.ipynb
```

---

## 💡 Key Findings

- **NVDA, AAPL, MSFT, TSLA** had the highest Reddit post volumes (~1,200 each)
- Reddit activity **surged dramatically post-2020**, peaking around 2024–2025
- Top terms in posts: *stock, company, market, revenue, share, buy, click*
- Sentiment alone is a useful but **insufficient** predictor — additional financial features (volume, volatility, macro indicators) are needed

---

## 🔮 Future Work

- Fine-tune LLMs on finance-specific corpora (FinBERT, BloombergGPT)
- Add features: trading volume, news sentiment, earnings dates, macro indicators
- Explore LSTM/Transformer models for time-series sentiment aggregation
- Build a real-time web dashboard for live ticker sentiment + return forecasting
- Expand to non-English subreddits for global market coverage

---

## 📚 References

1. Mittal & Goel (2012) — Sentiment Analysis of Twitter Data for Stock Market Prediction
2. Bollen, Mao & Zeng (2011) — Sentiment Analysis on Social Media for Stock Movement Prediction
3. Mahajan (2017) — Stock Market Prediction Using Twitter Sentiment Analysis
4. Hugging Face Transformers — https://huggingface.co/transformers
5. Doe & Smith (2025) — A Multimodal Deep Learning Framework for Stock Movement Forecasting Using Reddit

---
