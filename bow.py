from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "this is the first document",
    "this is the second document",
    "this is the third document",
    "is this the first document"
]

vectorizer = CountVectorizer()

vectorizer.fit(corpus)
vocabulary = vectorizer.vocabulary_
bow = vectorizer.transform(corpus)

print("Vocabulary:", vocabulary)

for i in range(len(corpus)):
    print(f"BoW({i+1}) = {bow[i].toarray()[0]}")
