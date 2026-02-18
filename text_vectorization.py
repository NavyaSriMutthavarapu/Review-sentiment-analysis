import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean(doc):
    doc = doc.lower()
    doc = re.sub(r'[^a-zA-Z\s]', '', doc)
    doc = re.sub(r'\s+', ' ', doc)

    words = doc.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)



