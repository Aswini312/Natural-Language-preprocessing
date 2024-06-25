import spacy
from nltk.stem import PorterStemmer

# Load SpaCy's English language model
nlp = spacy.load("en_core_web_sm")

def preprocess_text_spacy(text):
    # Tokenization, Stopword Removal, and Lemmatization using SpaCy
    doc = nlp(text)
    tokens = [token.text for token in doc]
    tokens_no_stopwords = [token.text for token in doc if not token.is_stop]
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]

    # Stemming using NLTK's PorterStemmer
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens_no_stopwords]

    print("Tokens:", tokens)
    print("Tokens after Stopword Removal:", tokens_no_stopwords)
    print("Stemmed Tokens:", stemmed_tokens)
    print("Lemmatized Tokens:", lemmatized_tokens)

    return {
        "tokens": tokens,
        "tokens_no_stopwords": tokens_no_stopwords,
        "stemmed_tokens": stemmed_tokens,
        "lemmatized_tokens": lemmatized_tokens
    }

# Example usage
text = "The striped bats are hanging on their feet for best"
processed_text = preprocess_text_spacy(text)
print("Processed Text:", processed_text)
