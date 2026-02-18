import nltk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words=set(stopwords.words('english'))
def clean(doc):
    doc=doc.lower()
    doc=re.sub(r'[^a-zA-Z\s]','',doc)
    doc=re.sub(r'\s+',' ',doc)
    tokens=word_tokenize(doc)
    filtered=[word for word in tokens if word not in stop_words]
    lemmatizer=WordNetLemmatizer()
    lemmatized=[lemmatizer.lemmatize(word) for word in filtered]
    return " ".join(lemmatized)