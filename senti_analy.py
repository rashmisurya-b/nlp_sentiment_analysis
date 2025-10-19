# AI Echo — Conversational Partner (One-File Streamlit App)
import streamlit as st
import joblib, os, time, requests
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import requests

# -------------------------------------------------------
# PAGE CONFIG & THEME
# -------------------------------------------------------
st.set_page_config(page_title="AI Echo — Conversational Partner", layout="wide")
sns.set_theme(style="whitegrid")


THEME_CSS = """
<style>
body {
  background: linear-gradient(135deg, #f6f9fc, #e8f0fe);
  color: #333;
  font-family: "Poppins", sans-serif;
}
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #f6f9fc, #e8f0fe);
}
div[data-testid="stVerticalBlock"] > div {
  background: rgba(255,255,255,0.9);
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  backdrop-filter: blur(4px);
}
button[kind="primary"] {
  background: linear-gradient(90deg,#00C9FF,#92FE9D);
  color:black; border:none; font-weight:bold;
  border-radius:10px; transition:transform .2s;
}
button[kind="primary"]:hover {
  transform:scale(1.05);
  background:linear-gradient(90deg,#4facfe,#00f2fe);
  color:white;
}
h1,h2,h3 {
  color:#1a1a1a;
  text-shadow:none;
}
p, label, span, div {
  color:#333 !important;
}
footer {visibility:hidden;}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# -------------------------------------------------------
# SPLASH SCREEN
# -------------------------------------------------------
with st.empty():
    st.markdown("<h1 style='text-align:center; color:#fff;'>👋 Hello! AI Echo is warming up...</h1>", unsafe_allow_html=True)
    st.progress(0)
    for i in range(0,101,5):
        st.progress(i)
        time.sleep(0.05)
    time.sleep(0.4)

# -------------------------------------------------------
# LOAD ASSETS & MODEL
# -------------------------------------------------------
@st.cache_resource
def load_assets():
    vect = joblib.load("C:\\Users\\B Rashmi Surya Vetri\\Desktop\\Chatgpt\\models\\tfidf_vectorizer.pkl")
    model = joblib.load("C:\\Users\\B Rashmi Surya Vetri\\Desktop\\Chatgpt\\models\\logisticregression.pkl")
    le = joblib.load("C:\\Users\\B Rashmi Surya Vetri\\Desktop\\Chatgpt\\models\\label_encoder.pkl")
    df = pd.read_csv("C:\\Users\\B Rashmi Surya Vetri\\Desktop\\Chatgpt\\cleaned_reviews.csv")
    return vect, model, le, df

vect, model, le, df = load_assets()



# -------------------------------------------------------
# TRY IT YOURSELF SECTION
# -------------------------------------------------------
st.subheader("💭 Try It Yourself — Instant Sentiment Prediction")
user_text = st.text_area("Type or paste any text:", height=120, placeholder="Example: 'I love using AI Echo!'")

if st.button("🔍 Analyze Sentiment"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        X = vect.transform([user_text])
        pred = model.predict(X)
        probs = model.predict_proba(X)
        label = le.inverse_transform(pred)[0]
        conf = np.max(probs)*100
        emoji = {"positive":"😊","neutral":"😐","negative":"😞"}[label]
        msg = {"positive":"Positive energy detected!",
               "neutral":"Balanced feedback detected.",
               "negative":"Some dissatisfaction detected."}[label]
        st.success(f"*Predicted Sentiment:* {label.upper()} {emoji}")
        st.write(f"*Confidence:* {conf:.2f}%")
        st.info(msg)
        st.balloons()

st.markdown("---")

# -------------------------------------------------------
# DASHBOARD SECTION
# -------------------------------------------------------
st.subheader("📊 Sentiment Dashboard — Explore Key Insights")

# Prepare predictions for dataset
@st.cache_data
def predict_dataset(df):
    X = vect.transform(df["cleaned_review_basic"].astype(str))
    preds = model.predict(X)
    df["predicted_sentiment"] = le.inverse_transform(preds)
    return df
df = predict_dataset(df)

questions = [
 "1️⃣ Overall sentiment of user reviews",
 "2️⃣ How does sentiment vary by rating?",
 "3️⃣ Which keywords are most associated with each sentiment?",
 "4️⃣ How has sentiment changed over time?",
 "5️⃣ Do verified users tend to leave more positive reviews?",
 "6️⃣ Are longer reviews more likely to be negative or positive?",
 "7️⃣ Which locations show the most positive or negative sentiment?",
 "8️⃣ Is there a difference in sentiment across platforms (Web vs Mobile)?",
 "9️⃣ Which ChatGPT versions have higher/lower sentiment?",
 "🔟 What are the most common negative feedback themes?"
]

if "q_index" not in st.session_state:
    st.session_state.q_index = 0

col1, col2, col3 = st.columns([1,2,1])
with col1:
    if st.button("⬅️ Previous"):
        st.session_state.q_index = (st.session_state.q_index-1)%len(questions)
        st.balloons()
with col2:
    choice = st.selectbox("Select a question:", questions, index=st.session_state.q_index)
    st.session_state.q_index = questions.index(choice)
with col3:
    if st.button("Next ➡️"):
        st.session_state.q_index = (st.session_state.q_index+1)%len(questions)
        st.balloons()

q = questions[st.session_state.q_index]
st.header(q)

# -------- Visualization logic --------
if q == questions[0]:
    counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, palette="coolwarm", ax=ax)
    st.pyplot(fig)
    st.info(f"💡 Most reviews are {counts.idxmax()}.")

elif q == questions[1]:
    if "rating_num" in df.columns:
        pivot = pd.crosstab(df["rating_num"], df["sentiment"], normalize="index")
        st.bar_chart(pivot)
        st.info("💡 Higher ratings align with positive text sentiment.")
    else:
        st.warning("Rating column missing.")

elif q == questions[2]:
    def top_words(sent):
        text = " ".join(df.loc[df["sentiment"]==sent,"cleaned_review_basic"]).split()
        return pd.DataFrame(Counter(text).most_common(10), columns=["Word","Count"])
    col1,col2,col3=st.columns(3)
    for s,c in zip(["positive","neutral","negative"],[col1,col2,col3]):
        with c:
            st.subheader(s.capitalize())
            st.dataframe(top_words(s))
    st.info("💡 Words differ clearly across sentiments.")

elif q == questions[3]:
    if "date" in df.columns:
        df["date"]=pd.to_datetime(df["date"],errors="coerce")
        df["month"]=df["date"].dt.to_period("M")
        trend=df.groupby("month")["sentiment"].value_counts(normalize=True).unstack().fillna(0)
        st.line_chart(trend)
        st.info("💡 Observe monthly satisfaction trends.")
    else:
        st.warning("No date column found.")

elif q == questions[4]:
    if "verified_purchase_bool" in df.columns:
        tab=df.groupby("verified_purchase_bool")["sentiment"].value_counts(normalize=True).unstack().fillna(0)
        st.bar_chart(tab)
        st.info("💡 Verified users often post more positive reviews.")
    else:
        st.warning("verified_purchase not found.")

elif q == questions[5]:
    df["length"]=df["cleaned_review_basic"].apply(lambda x: len(str(x).split()))
    fig,ax=plt.subplots()
    sns.boxplot(x="sentiment", y="length", data=df,
                order=["negative","neutral","positive"], ax=ax)
    st.pyplot(fig)
    st.info("💡 Longer reviews show stronger emotions.")

elif q == questions[6]:
    if "location" in df.columns:
        top=df["location"].value_counts().head(10).index
        avg=df[df["location"].isin(top)].groupby("location")["sentiment"].value_counts(normalize=True).unstack().fillna(0)
        st.bar_chart(avg)
        st.info("💡 Regional sentiment differences observed.")
    else:
        st.warning("No location column found.")

elif q == questions[7]:
    if "platform" in df.columns:
        tab=df.groupby("platform")["sentiment"].value_counts(normalize=True).unstack().fillna(0)
        st.bar_chart(tab)
        st.info("💡 Compare experience on Web vs Mobile.")
    else:
        st.warning("Platform column not found.")

elif q == questions[8]:
    if "version" in df.columns:
        topv=df["version"].value_counts().head(10).index
        ver=df[df["version"].isin(topv)].groupby("version")["sentiment"].value_counts(normalize=True).unstack().fillna(0)
        st.bar_chart(ver)
        st.info("💡 Track satisfaction across versions.")
    else:
        st.warning("Version column not found.")

elif q == questions[9]:
    neg=df[df["sentiment"]=="negative"]
    words=" ".join(neg["cleaned_review_basic"]).split()
    freq=Counter(words).most_common(15)
    if freq:
        w,c=zip(*freq)
        fig,ax=plt.subplots()
        sns.barplot(x=list(c), y=list(w), palette="rocket", ax=ax)
        st.pyplot(fig)
        st.info("💡 Frequent negative words show key issues.")
    else:
        st.info("No negative reviews found.")

st.markdown("---")
st.caption("✨ AI Echo — Conversational Partner · Built with Streamlit & Lottie")