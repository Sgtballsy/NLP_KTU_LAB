import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


nltk.download("movie_reviews")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return {word: True for word in tokens}

pos_reviews = [(movie_reviews.raw(fileid), "positive") for fileid in movie_reviews.fileids("pos")]
neg_reviews = [(movie_reviews.raw(fileid), "negative") for fileid in movie_reviews.fileids("neg")]
all_reviews = pos_reviews + neg_reviews

processed_data = [(preprocess(text), label) for (text, label) in all_reviews]

train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)

classifier = NaiveBayesClassifier.train(train_data)

new_texts = [
    "The movie was amazing and full of great performances",
    "The plot was boring and the acting was bad",
    "I really enjoyed the visuals and soundtrack",
    "It was a waste of time and money"
]

print("\n--- Sentiment Predictions ---")
for text in new_texts:
    features = preprocess(text)
    prediction = classifier.classify(features)
    print(f"'{text}' → {prediction}")
