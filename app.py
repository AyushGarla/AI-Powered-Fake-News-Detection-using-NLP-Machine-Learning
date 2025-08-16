import re
import joblib
import pandas as pd
import streamlit as st

# --- NLTK setup ---
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download once (cached)
@st.cache_resource
def _download_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass
_download_nltk()

# --- Preprocessing (must match training) ---
STOPWORDS = set(stopwords.words('english')) - {'no', 'nor', 'not'}
LEMMATIZER = WordNetLemmatizer()

def _wn_pos(tb):
    if tb.startswith('J'): return wordnet.ADJ
    if tb.startswith('V'): return wordnet.VERB
    if tb.startswith('N'): return wordnet.NOUN
    if tb.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

def preprocess_text(text: str):
    s = text.lower()
    s = re.sub(r"^b['\"]", "", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\b[a-z]\b", " ", s)
    s = re.sub(r"^[a-z]\s+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def nlp_preprocess_text(text: str):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    pos_tags = pos_tag(tokens)
    return [LEMMATIZER.lemmatize(w, _wn_pos(p)) for w, p in pos_tags]

def preprocess_for_vectorizer(texts: pd.Series) -> pd.Series:
    cleaned = texts.apply(preprocess_text)
    tokens = cleaned.apply(nlp_preprocess_text)
    return tokens.apply(lambda toks: " ".join(toks))

# --- Load model + vectorizer ---
@st.cache_resource
def load_artifacts(model_path: str, vec_path: str):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

# --- UI ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detection (NLP + ML)")
st.write("Enter a headline/article or upload a CSV to classify as **Fake (0)** or **Real (1)**.")

# Sidebar - model artifacts
st.sidebar.header("Model Artifacts")
model_file = st.sidebar.file_uploader("Upload model (.pkl)", type=["pkl"], key="model")
vec_file   = st.sidebar.file_uploader("Upload vectorizer (.pkl)", type=["pkl"], key="vectorizer")

if model_file and vec_file:
    with open("uploaded_model.pkl", "wb") as f: f.write(model_file.read())
    with open("uploaded_vectorizer.pkl", "wb") as f: f.write(vec_file.read())
    model, vectorizer = load_artifacts("uploaded_model.pkl", "uploaded_vectorizer.pkl")
else:
    # Fallback to local files if present
    try:
        model, vectorizer = load_artifacts("fake_news_svm_model.pkl", "tfidf_vectorizer.pkl")
    except Exception as e:
        st.warning("Upload model & vectorizer in the sidebar or place 'fake_news_svm_model.pkl' and 'tfidf_vectorizer.pkl' next to app.py.")
        st.stop()

tab1, tab2 = st.tabs(["🔎 Single Prediction", "📦 Batch CSV Prediction"])

with tab1:
    st.subheader("Single Text / Headline + Article")
    title_input = st.text_input("Title", "")
    text_input = st.text_area("Article Text", "", height=180)
    if st.button("Predict"):
        combined = pd.Series([f"{title_input} {text_input}"])
        prepped = preprocess_for_vectorizer(combined)
        X = vectorizer.transform(prepped)
        pred = int(model.predict(X)[0])
        label = "Real (1)" if pred == 1 else "Fake (0)"
        st.success(f"Prediction: **{label}**")

with tab2:
    st.subheader("Batch Prediction from CSV")
    st.caption("CSV must include **title** and **text** columns. (Other columns are ignored.)")
    csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv")
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        if not {"title","text"}.issubset(df.columns):
            st.error("CSV must contain 'title' and 'text' columns.")
        else:
            combined = (df["title"].astype(str) + " " + df["text"].astype(str))
            prepped = preprocess_for_vectorizer(combined)
            X = vectorizer.transform(prepped)
            preds = model.predict(X)
            out = df.copy()
            out["label"] = preds  # 0=fake, 1=real
            st.write("Preview of results:")
            st.dataframe(out.head(20))
            st.download_button(
                "⬇️ Download predictions CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption("Model: Linear SVM | Features: TF‑IDF | Preprocessing: regex clean, stopwords(-not/no/nor), POS‑lemmatization")
