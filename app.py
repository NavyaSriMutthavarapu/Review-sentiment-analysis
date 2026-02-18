import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
stop_words=set(stopwords.words('english'))
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
def clean(doc):
    doc=doc.lower()
    doc=re.sub(r'[^a-zA-Z\s]','',doc)
    tokens=word_tokenize(doc)
    stop_words=set(stopwords.words('english'))
    filtered=[word for word in tokens if word not in stop_words]
    lemmatizer=WordNetLemmatizer()
    lemmatized=[lemmatizer.lemmatize(word) for word in filtered]
    return " ".join(lemmatized)
# Streamlit UI
st.title("🍽️ Review Sentiment Classifier")
st.write("Enter a review to check if it is Positive or Negative")

user_input = st.text_area("Enter Review")

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned = clean(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")
    else:
        st.warning("Please enter some text")