import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Load model and preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Initialize preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Streamlit page
st.set_page_config(page_title="Toxic Comment Classifier")

st.title("🛡️ Toxic Comment Classification")
st.write("Enter a comment below and click Predict.")

comment = st.text_area("Enter your comment")

if st.button("Predict"):

    clean_comment = clean_text(comment)

    scores = sia.polarity_scores(comment)

    input_df = pd.DataFrame({
        "clean_comment":[clean_comment],
        "word_count":[len(comment.split())],
        "char_count":[len(comment)],
        "neg":[scores["neg"]],
        "neu":[scores["neu"]],
        "pos":[scores["pos"]],
        "compound":[scores["compound"]]
    })

    processed = preprocessor.transform(input_df)

    prediction = model.predict(processed)

    if prediction[0] == 1:
        st.error("🚨 Toxic Comment")
    else:
        st.success("✅ Non Toxic Comment")