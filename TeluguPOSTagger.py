import nltk
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger
from nltk.corpus import indian
from nltk.tag import tnt

class TeluguPOSTagger:
    def __init__(self):
        # Load annotated Telugu POS tagged data
        tagged_sentences = indian.tagged_sents('telugu.pos')
        
        # Split the data into training and testing sets
        train_data = tagged_sentences[100:]
        test_data = tagged_sentences[:100]

        # Initialize and train POS tagger
        self.tagger = self.train_tagger(train_data)

    def train_tagger(self, train_data):
        # Initialize taggers
        default_tagger = DefaultTagger('NN')
        unigram_tagger = UnigramTagger(train_data, backoff=default_tagger)
        bigram_tagger = BigramTagger(train_data, backoff=unigram_tagger)
        tnt_tagger = tnt.TnT()

        # Train TnT tagger
        tnt_tagger.train(train_data)

        return tnt_tagger

    def tag(self, sentence):
        # Tokenize input sentence
        tokens = nltk.word_tokenize(sentence)

        # Tag tokens using the trained tagger
        tagged_tokens = self.tagger.tag(tokens)

        return tagged_tokens

if __name__ == "__main__":
    # Initialize POS tagger
    pos_tagger = TeluguPOSTagger()

    # Test the POS tagger with a sample sentence
    sentence = "నేను ఒక మంచి ప్రాంతంలో ఉంది."
    tagged_sentence = pos_tagger.tag(sentence)
    print(tagged_sentence)
