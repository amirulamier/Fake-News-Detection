import streamlit as st
import joblib
import re

# Load models
lr_model = joblib.load("model/logistic_model.pkl")
nb_model = joblib.load("model/naive_bayes_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Detect whether a news article is **Fake** or **Real** using AI models.")

# Model selector
model_choice = st.selectbox(
    "Select AI Model",
    ("Logistic Regression", "Naive Bayes")
)

user_input = st.text_area("News Content", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        if model_choice == "Logistic Regression":
            prediction = lr_model.predict(vectorized)[0]
            confidence = lr_model.predict_proba(vectorized).max()
        else:
            prediction = nb_model.predict(vectorized)[0]
            confidence = nb_model.predict_proba(vectorized).max()

        if prediction == 1:
            st.success(f"✅ REAL NEWS (Confidence: {confidence:.2f})")
        else:
            st.error(f"🚨 FAKE NEWS (Confidence: {confidence:.2f})")
