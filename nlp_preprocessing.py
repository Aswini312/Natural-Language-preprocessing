import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    print("Tokens:", tokens)

    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    tokens_no_stopwords = [token for token in tokens if token.lower() not in stop_words]
    print("Tokens after Stopword Removal:", tokens_no_stopwords)

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens_no_stopwords]
    print("Stemmed Tokens:", stemmed_tokens)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens_no_stopwords]
    print("Lemmatized Tokens:", lemmatized_tokens)

    return {
        "tokens": tokens,
        "tokens_no_stopwords": tokens_no_stopwords,
        "stemmed_tokens": stemmed_tokens,
        "lemmatized_tokens": lemmatized_tokens
    }

# Example usage
text = "The striped bats are hanging on their feet for best"
processed_text = preprocess_text(text)
print("Processed Text:", processed_text)
