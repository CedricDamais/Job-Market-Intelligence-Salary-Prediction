import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import AutoTokenizer

# Downloads for NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Stopwords
stop_words = set(stopwords.words('english'))

# Preloaded tokenizers
BYTE_TOKENIZER = AutoTokenizer.from_pretrained("gpt2")  # Byte-level BPE
BPE_TOKENIZER = AutoTokenizer.from_pretrained("roberta-base")  # WordPiece/BPE hybrid


# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d{10,}', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Vectorization (Count / TF-IDF)
def vectorize(texts, method="count", **kwargs):
    joined = [' '.join(tokenize(t, method="nltk", remove_stopwords=True)) for t in texts]
    if method == "count":
        vectorizer = CountVectorizer(**kwargs)
    elif method == "tfidf":
        vectorizer = TfidfVectorizer(**kwargs)
    else:
        raise ValueError(f"Unsupported vectorization method: {method}")
    vectors = vectorizer.fit_transform(joined)
    return vectors, vectorizer


# Sequences for Keras models
def prepare_sequences(texts, tokenizer=None, vocab_size=10000, max_len=100):
    if tokenizer is None:
        tokenizer = KerasTokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded, tokenizer


# Transformer-compatible encoding
def encode_transformer(texts, model_name="gpt2", max_len=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors='tf'
    )
    return encoded, tokenizer
