import streamlit as st
import pickle
import nltk

@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

setup_nltk()

from text_vectorization import clean

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit UI
st.title("🍽️ Review Sentiment Classifier")
st.write("Enter a review to check if it is Positive or Negative")

user_input = st.text_area("Enter Review")

if st.button("Predict"):
    if user_input.strip():
        cleaned = clean(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")
    else:
        st.warning("Please enter some text")
