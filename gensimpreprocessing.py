from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS, preprocess_string, strip_punctuation, strip_numeric
from gensim.corpora import Dictionary
from gensim.parsing.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Custom filters for preprocessing
CUSTOM_FILTERS = [lambda x: x.lower(), strip_punctuation, strip_numeric]

def preprocess_text_gensim(text):
    # Tokenization and stopword removal
    tokens = simple_preprocess(text)
    tokens_no_stopwords = [token for token in tokens if token not in STOPWORDS]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens_no_stopwords]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens_no_stopwords]

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
processed_text = preprocess_text_gensim(text)
print("Processed Text:", processed_text)
