import nltk
from nltk.corpus import brown
from nltk.probability import LidstoneProbDist

nltk.download('brown')
nltk.download('punkt')

def train_hmm_tagger():
    tagged_sentences = brown.tagged_sents()
    symbols = set()
    states = set()

    for sentence in tagged_sentences:
        for word, tag in sentence:
            symbols.add(word)
            states.add(tag)

    trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=states, symbols=symbols)
    hmm_tagger = trainer.train_supervised(
        tagged_sentences,
        estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins)
    )

    return hmm_tagger

def pos_tag_sentence(sentence, hmm_tagger):
    tokens = nltk.word_tokenize(sentence)
    tagged_tokens = hmm_tagger.tag(tokens)
    return tagged_tokens

hmm_tagger = train_hmm_tagger()
sentence = input("Enter the sentence to be tagged: ")
tagged = pos_tag_sentence(sentence, hmm_tagger)

print(tagged)
