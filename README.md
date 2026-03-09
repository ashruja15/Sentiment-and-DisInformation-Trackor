# Sentiment-and-DisInformation-Trackor

🚀 TruthPulseAI
AI-Powered YouTube Comment Intelligence Dashboard

TruthPulseAI is an AI-driven dashboard that analyzes YouTube comments to detect sentiment, toxicity, and potential bot activity using advanced NLP models.

It transforms raw comment data into actionable intelligence for content creators, researchers, and digital analysts.

📌 Features

✅ Sentiment Analysis (Positive / Neutral / Negative)

✅ Toxicity Detection

✅ Bot Activity Risk Scoring

✅ Sentiment Trend Over Time

✅ Word Cloud Visualization

✅ Interactive Dashboard (Streamlit)

✅ CSV Export of Full Analysis

🧠 How It Works

User enters a YouTube video URL.

Comments are fetched using YouTube Data API v3.

AI models analyze:

Sentiment (RoBERTa-based model)

Toxicity (BERT-based model)

A behavioral rule engine calculates Bot Risk Score.

Dashboard visualizes insights in real time.

🤖 AI Models Used
Sentiment Model

cardiffnlp/twitter-xlm-roberta-base-sentiment

Toxicity Model

unitary/toxic-bert

🛠️ Tech Stack

- Python

- Streamlit

- HuggingFace Transformers

- PyTorch

- YouTube Data API v3

- Plotly

- WordCloud

- Pandas

📊 Bot Detection Logic

Bot Risk Score is calculated using behavioral anomaly detection:

Suspicious keywords (crypto, giveaway, telegram, etc.)

Repeated comments

Very short comments

High toxicity

Zero likes

Each condition increases risk score, producing a final percentage (0–100%).