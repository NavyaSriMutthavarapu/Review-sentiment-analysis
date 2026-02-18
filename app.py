import streamlit as st
import pickle
from text_vectorization import clean

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("🍽️ Review Sentiment Classifier")

user_input = st.text_area("Enter your review:")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")


