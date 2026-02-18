import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('wordnet')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean(doc):
    doc = doc.lower()
    doc = re.sub(r'[^a-zA-Z\s]', '', doc)
    doc = re.sub(r'\s+', ' ', doc).strip()

    tokens = word_tokenize(doc)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)



