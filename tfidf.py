import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")
nltk.download("stopwords")

text_data = ["tiger bites deer", "deer bites grass", "grass is green"]

def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    text = ', '.join(words)
    return text

text_data_preprocessed = [preprocess_text(text) for text in text_data]
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(text_data_preprocessed)
features = tfidf.get_feature_names_out()

print("Features:", features)
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())
