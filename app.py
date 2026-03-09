# ============================
# IMPORTS
# ============================

import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="TruthPulseAI",
    page_icon="🚀",
    layout="wide"
)

import pandas as pd
import urllib.parse as urlparse
from urllib.parse import parse_qs
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from wordcloud import WordCloud
import plotly.graph_objects as go
import re

# ============================
# CONFIG
# ============================

API_KEY = "YOUR API KEY"

if not API_KEY:
    st.error("⚠️ YouTube API Key missing")
    st.stop()

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY,
    cache_discovery=False
)

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
TOXIC_MODEL = "unitary/toxic-bert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# LOAD MODELS
# ============================

@st.cache_resource
def load_models():

    sent_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    sent_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

    tox_tokenizer = AutoTokenizer.from_pretrained(TOXIC_MODEL)
    tox_model = AutoModelForSequenceClassification.from_pretrained(TOXIC_MODEL)

    sent_model.to(device)
    tox_model.to(device)

    sent_model.eval()
    tox_model.eval()

    return sent_tokenizer, sent_model, tox_tokenizer, tox_model


sent_tokenizer, sent_model, tox_tokenizer, tox_model = load_models()

# ============================
# HELPER FUNCTIONS
# ============================

def extract_video_id(url):

    parsed_url = urlparse.urlparse(url)

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    if parsed_url.hostname in ("www.youtube.com","youtube.com","m.youtube.com"):
        return parse_qs(parsed_url.query).get("v",[None])[0]

    return None


def clean_text(text):

    text = re.sub(r"http\S+","",text)
    text = re.sub(r"@\w+","",text)

    return text.strip()


# ============================
# FETCH COMMENTS
# ============================

def fetch_all_comments(video_id,max_comments):

    comments=[]
    next_page_token=None

    try:

        while len(comments)<max_comments:

            request=youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            )

            response=request.execute()

            for item in response.get("items",[]):

                snippet=item["snippet"]["topLevelComment"]["snippet"]

                comments.append({
                    "Username":snippet["authorDisplayName"],
                    "Comment":snippet["textOriginal"],
                    "Likes":snippet["likeCount"],
                    "Published":snippet["publishedAt"]
                })

                if len(comments)>=max_comments:
                    break

            next_page_token=response.get("nextPageToken")

            if not next_page_token:
                break

    except HttpError as e:

        st.error(f"YouTube API Error: {e}")
        return []

    return comments


# ============================
# SENTIMENT ANALYSIS
# ============================

def batch_sentiment(texts,batch_size=64):

    results=[]
    labels=["Negative","Neutral","Positive"]

    for i in range(0,len(texts),batch_size):

        batch=texts[i:i+batch_size]

        inputs=sent_tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        inputs={k:v.to(device) for k,v in inputs.items()}

        with torch.no_grad():
            outputs=sent_model(**inputs)

        scores=F.softmax(outputs.logits,dim=1)
        predictions=torch.argmax(scores,dim=1)

        for j in range(len(batch)):
            results.append(labels[predictions[j].item()])

    return results


# ============================
# TOXICITY ANALYSIS
# ============================

def batch_toxicity(texts,batch_size=64):

    results=[]

    for i in range(0,len(texts),batch_size):

        batch=texts[i:i+batch_size]

        inputs=tox_tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        inputs={k:v.to(device) for k,v in inputs.items()}

        with torch.no_grad():
            outputs=tox_model(**inputs)

        scores=torch.sigmoid(outputs.logits)

        for j in range(len(batch)):
            results.append(round(scores[j][0].item()*100,2))

    return results


# ============================
# BOT DETECTION
# ============================

def detect_bot(text,likes,toxicity,comment_counts):

    score=0

    suspicious_words=[
        "crypto","bitcoin","giveaway",
        "telegram","whatsapp","earn money"
    ]

    if any(word in text.lower() for word in suspicious_words):
        score+=2

    if len(text.split())<=2:
        score+=1

    if comment_counts.get(text,0)>3:
        score+=2

    if toxicity>80:
        score+=1

    if likes==0:
        score+=1

    return min(score*20,100)


# ============================
# UI
# ============================

st.title("🚀 TruthPulseAI - AI Comment Intelligence Dashboard")

video_url=st.text_input("Enter YouTube Video URL")

max_comments=st.slider(
    "Number of Comments",
    50,
    500,
    200
)


# ============================
# ANALYSIS
# ============================

if video_url and st.button("Analyze Comments"):

    video_id=extract_video_id(video_url)

    if not video_id:
        st.error("Invalid YouTube URL")
        st.stop()

    with st.spinner("Fetching & Processing Comments..."):

        raw_comments=fetch_all_comments(video_id,max_comments)

        if not raw_comments:
            st.warning("No comments found.")
            st.stop()

        texts=[clean_text(c["Comment"]) for c in raw_comments]

        comment_counts={}
        for t in texts:
            comment_counts[t]=comment_counts.get(t,0)+1

        sentiment=batch_sentiment(texts)
        toxicity=batch_toxicity(texts)

        data=[]

        for i,item in enumerate(raw_comments):

            bot_score=detect_bot(
                texts[i],
                item["Likes"],
                toxicity[i],
                comment_counts
            )

            data.append({
                "Username":item["Username"],
                "Comment":texts[i],
                "Likes":item["Likes"],
                "Sentiment":sentiment[i],
                "Toxicity %":toxicity[i],
                "Bot Score %":bot_score,
                "Published":item["Published"]
            })

        df=pd.DataFrame(data)

# ============================
# EXECUTIVE SUMMARY
# ============================

    st.subheader("📊 Executive Summary")

    pos=(df["Sentiment"]=="Positive").mean()*100
    neg=(df["Sentiment"]=="Negative").mean()*100
    neu=(df["Sentiment"]=="Neutral").mean()*100
    bot=df["Bot Score %"].mean()

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Positive %",f"{pos:.2f}%")
    c2.metric("Negative %",f"{neg:.2f}%")
    c3.metric("Neutral %",f"{neu:.2f}%")
    c4.metric("Avg Bot Risk",f"{bot:.2f}%")

# ============================
# SENTIMENT TREND
# ============================

    st.subheader("📈 Sentiment Trend")

    df["Published"]=pd.to_datetime(df["Published"])
    df.sort_values("Published",inplace=True)

    sentiment_time=(
        df.groupby(pd.Grouper(key="Published",freq="1H"))["Sentiment"]
        .value_counts()
        .unstack()
        .fillna(0)
    )

    fig=go.Figure()

    for s in ["Positive","Neutral","Negative"]:
        if s in sentiment_time.columns:
            fig.add_trace(go.Scatter(
                x=sentiment_time.index,
                y=sentiment_time[s],
                mode="lines+markers",
                name=s
            ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        hovermode="x unified"
    )

    st.plotly_chart(fig,use_container_width=True)

# ============================
# WORD CLOUD
# ============================

    st.subheader("☁ Word Cloud")

    text_blob=" ".join(df["Comment"])

    wc=WordCloud(width=800,height=400).generate(text_blob)

    fig_wc,ax_wc=plt.subplots()
    ax_wc.imshow(wc)
    ax_wc.axis("off")

    st.pyplot(fig_wc)

# ============================
# COMMENTS TABLE
# ============================

    st.subheader("💬 All Comments")

    st.dataframe(df,use_container_width=True)

# ============================
# DOWNLOAD CSV
# ============================

    st.download_button(
        "Download Full Analysis CSV",
        df.to_csv(index=False),
        "truthpulse_full_analysis.csv",
        "text/csv"

    )
