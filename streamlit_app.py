# streamlit_app.py
import streamlit as st
import pickle, re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------- load artefacts --------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------- recreate preprocess() exactly as in training --------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text: str) -> str:
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

# -------- Streamlit UI --------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="📊")
st.title("📊 Product-Review Sentiment Analyzer")
st.write("Type or paste a product review and click **Predict**.")

review = st.text_area("Your review:", height=150)

if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned = preprocess(review)
        vect = vectorizer.transform([cleaned])
        pred  = model.predict(vect)[0]
        if pred == "positive":
            st.success("🌟 Sentiment: **Positive**")
        else:
            st.error("🙁 Sentiment: **Negative**")
