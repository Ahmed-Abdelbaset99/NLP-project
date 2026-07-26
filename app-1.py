import streamlit as st
import joblib
import nltk
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load('banking_assistant_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    label_names = joblib.load('label_names.pkl')
    return model, vectorizer, label_names

model, vectorizer, label_names = load_artifacts()


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def get_accuracy():
    try:
        with open('model_evaluation_report.txt', 'r') as f:
            first_line = f.readline()
            match = re.search(r'Accuracy:\s*([\d.]+)', first_line)
            if match:
                return float(match.group(1))
    except Exception:
        pass
    return 0.8218  # fallback

accuracy = get_accuracy()

# Page config
st.set_page_config(page_title="Banking77 Intent Classifier", layout="centered")


st.markdown("""
    <style>
    .big-font {
        font-size: 40px !important;
        font-weight: bold;
        color: #1f4e79;
    }
    .subtitle {
        font-size: 18px;
        color: #555555;
    }
    .result-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }
    .label-name {
        font-size: 24px;
        font-weight: bold;
        color: #1f4e79;
    }
    .label-number {
        font-size: 20px;
        color: #333333;
    }
    .accuracy-text {
        font-size: 16px;
        color: #2e8b57;
        font-weight: 600;
    }
    .processed-text {
        font-size: 16px;
        color: #555555;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">🏦 Banking77 Intent Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter a banking-related query and the model will predict the intent label and its number.</p>', unsafe_allow_html=True)

st.markdown("---")

query = st.text_area("✍️ Your Query", placeholder="e.g. I want to activate my new card", height=100)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_clicked = st.button("🔍 Predict Intent", use_container_width=True)

if predict_clicked:
    if not query.strip():
        st.warning("⚠️ Please enter a query first.")
    else:
        with st.spinner("🤖 Analyzing your query..."):
            processed = preprocess_text(query)
            X = vectorizer.transform([processed])
            pred = model.predict(X)[0]
            label_name = label_names[pred]

        st.markdown("---")
        st.markdown("### ✅ Prediction Result")
        st.markdown(f"""
        <div class="result-box">
            <p class="label-name">🏷️ Label: {label_name}</p>
            <p class="label-number">🔢 Label Number: {pred}</p>
            <p class="processed-text">📝 Processed Text: {processed}</p>
            <p class="accuracy-text">🎯 Model Accuracy: {accuracy*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Powered by LogisticRegression + TF-IDF on the Banking77 dataset 🚀")
