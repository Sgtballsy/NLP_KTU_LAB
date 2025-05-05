import nltk

nltk.download('averaged_perceptron_tagger') 
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download('maxent_ne_chunker_tab')

text = "Shawn Murphy is a good doctor"

tokens = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)

for entity in entities:
    if hasattr(entity, "label"):
        print(entity.label(), " ".join(c[0] for c in entity.leaves()))