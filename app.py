import streamlit as st
import joblib
import numpy as np
import re
import string
import nltk

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========== Download NLTK resources ==========

nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# ========== Load Artifacts ==========
@st.cache_resource 
def load_artifacts():
    tfidf = joblib.load("saved_models/tfidf_vectorizer.joblib")
    log_reg = joblib.load("saved_models/log_reg_model.joblib")
    bow_vect = joblib.load("saved_models/bow_vectorizer.joblib")
    nb = joblib.load("saved_models/nb_model.joblib")
    tokenizer = joblib.load("saved_models/tokenizer.joblib")
    label_encoder = joblib.load("saved_models/label_encoder.joblib")
    lstm_model = load_model("saved_models/lstm_model.h5")
    return tfidf, log_reg, bow_vect, nb, tokenizer, label_encoder, lstm_model


tfidf, log_reg, bow_vect, nb, tokenizer, label_encoder, lstm_model = load_artifacts()

# ========== Text Cleaning ==========
def clean_text(text: str) -> str:
    # Basic normalization and cleaning 
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    return " ".join(tokens)

def preprocess_for_models(raw_text: str) -> str:
    return clean_text(raw_text)

# ========== Prediction Function ==========
def predict_sentiment(text: str):
    cleaned = preprocess_for_models(text)

    # Logistic Regression (TF-IDF)
    X_tfidf = tfidf.transform([cleaned])
    pred_lr = log_reg.predict(X_tfidf)[0]
    proba_lr = (
        log_reg.predict_proba(X_tfidf)[0]
        if hasattr(log_reg, "predict_proba")
        else None
    )

    # Naive Bayes (BoW)
    X_bow = bow_vect.transform([cleaned])
    pred_nb = nb.predict(X_bow)[0]
    proba_nb = nb.predict_proba(X_bow)[0] if hasattr(nb, "predict_proba") else None

    # LSTM
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=100, padding="post", truncating="post")
    proba_lstm = lstm_model.predict(pad, verbose=0)[0]  # returns array of probs
    pred_lstm_idx = int(np.argmax(proba_lstm))
    pred_lstm = label_encoder.inverse_transform([pred_lstm_idx])[0]

    return {
        "cleaned": cleaned,
        "lr": {"label": pred_lr, "proba": proba_lr},
        "nb": {"label": pred_nb, "proba": proba_nb},
        "lstm": {"label": pred_lstm, "proba": proba_lstm},
    }


st.set_page_config(page_title="University Review Sentiment Analyzer", page_icon="🎓")
st.title("University Review Sentiment Analyzer")
st.write("Compare **Machine Learning** vs **Deep Learning** models on university/course reviews.")

st.markdown(
    """
This app predicts whether a review is **positive**, **neutral**, or **negative** using:
- Logistic Regression (TF-IDF)
- Naive Bayes (Bag-of-Words)
- LSTM (word sequences)
"""
)

# Example texts for quick testing
with st.expander("Try example reviews"):
    examples = {
        "Very positive": "The course was amazing and the professor was extremely helpful.",
        "Mixed": "The content was good but the assignments were too time-consuming.",
        "Negative": "The lectures were boring and I learned almost nothing."
    }
    example_choice = st.radio("Examples", list(examples.keys()), horizontal=True)
    if st.button("Use selected example"):
        st.session_state["example_text"] = examples[example_choice]

default_text = st.session_state.get(
    "example_text",
    "The course content was well structured and the instructor explained concepts clearly."
)
user_input = st.text_area("Enter a university/course review:", value=default_text, height=150)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        results = predict_sentiment(user_input)

        st.subheader("Model predictions")
        col1, col2, col3 = st.columns(3)
        col1.metric("Logistic Regression", results["lr"]["label"])
        col2.metric("Naive Bayes", results["nb"]["label"])
        col3.metric("LSTM", results["lstm"]["label"])

        # Show cleaned text
        st.write("---")
        st.subheader("Preprocessed text")
        st.code(results["cleaned"])

        # Confidence table
        st.write("---")
        class_names = list(label_encoder.classes_)
        rows = []
        if results["lr"]["proba"] is not None:
            rows.append(
                ["Logistic Regression"]
                + [float(p) for p in results["lr"]["proba"]]
            )
        if results["nb"]["proba"] is not None:
            rows.append(
                ["Naive Bayes"]
                + [float(p) for p in results["nb"]["proba"]]
            )
        rows.append(
            ["LSTM"] + [float(p) for p in results["lstm"]["proba"]]
        )

        st.table(
            { "Model": [r[0] for r in rows],
              **{cls: [r[i+1] for r in rows] for i, cls in enumerate(class_names)}
            }
        )

      
