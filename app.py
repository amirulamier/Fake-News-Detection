import streamlit as st
import joblib
import re
import numpy as np

# Load models
lr_model = joblib.load("model/logistic_model.pkl")
nb_model = joblib.load("model/naive_bayes_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

def risk_level(conf):
    if conf < 0.65:
        return "🟡 UNCERTAIN"
    elif conf < 0.80:
        return "🟠 LIKELY"
    else:
        return "🔴 HIGH RISK"

def explain_prediction(model, vectorized_text, top_n=5):
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    indices = vectorized_text.toarray()[0].argsort()[-top_n:][::-1]
    return [feature_names[i] for i in indices]

st.set_page_config(page_title="Fake News Risk Analyzer", layout="centered")

st.title("📰 Fake News Risk Analyzer")
st.caption("An AI-based decision-support system for detecting misinformation")

st.markdown("### 📥 Input News Text")

col1, col2 = st.columns(2)

with col1:
    if st.button("Load Fake Example"):
        st.session_state.text = (
            "Breaking: Leaked documents reveal secret plans by global leaders "
            "to control the population using hidden technology."
        )

with col2:
    if st.button("Load Real Example"):
        st.session_state.text = (
            "The Ministry of Health announced new public health guidelines "
            "aimed at improving disease prevention nationwide."
        )

news_text = st.text_area(
    "Paste news article or headline below:",
    value=st.session_state.get("text", ""),
    height=200
)

st.caption("⚠️ For best results, use English news-style text (at least 15–20 words).")

if st.button("Analyze"):
    if len(news_text.split()) < 10:
        st.warning("Text too short for reliable analysis.")
    else:
        cleaned = clean_text(news_text)
        vectorized = vectorizer.transform([cleaned])

        # Predictions
        lr_pred = lr_model.predict(vectorized)[0]
        lr_conf = lr_model.predict_proba(vectorized).max()

        nb_pred = nb_model.predict(vectorized)[0]
        nb_conf = nb_model.predict_proba(vectorized).max()

        # Agreement check
        agreement = lr_pred == nb_pred

        # Final decision (use Logistic Regression as primary)
        final_pred = lr_pred
        final_conf = lr_conf

        st.markdown("---")
        st.markdown("## 🧠 AI Assessment Result")

        if final_pred == 1:
            st.success("🟢 REAL-STYLE NEWS")
        else:
            st.error("🔴 FAKE-STYLE NEWS")

        st.markdown(f"**Confidence:** {final_conf:.2f}")
        st.markdown(f"**Risk Level:** {risk_level(final_conf)}")

        if not agreement:
            st.warning(
                "⚠️ Model Disagreement Detected\n\n"
                "Logistic Regression and Naive Bayes produced different predictions. "
                "Result should be interpreted with caution."
            )

        # Explainability
        st.markdown("### 🔍 Key Influential Words (Explainable AI)")
        keywords = explain_prediction(lr_model, vectorized)
        st.write(", ".join(keywords))

        # Ethical disclaimer
        st.markdown("---")
        st.info(
            "ℹ️ **Important Notice**\n\n"
            "This system detects misinformation based on linguistic patterns "
            "and writing style, not factual verification. "
            "It should be used as a decision-support tool rather than a final authority."
        )
