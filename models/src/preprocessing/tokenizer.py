"""
Utility functions for loading and using Hugging Face tokenizers.
"""
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer


nltk.download('punkt')
nltk.download('stopwords')

# Stopwords
stop_words = set(stopwords.words('english'))

# Preloaded tokenizers
BYTE_TOKENIZER = AutoTokenizer.from_pretrained("gpt2")  # Byte-level BPE
BPE_TOKENIZER = AutoTokenizer.from_pretrained("roberta-base")  # WordPiece/BPE hybrid

def tokenize(text, method="nltk", remove_stopwords=False):
    if method == "nltk":
        tokens = word_tokenize(text)
    elif method == "split":
        tokens = text.split()
    elif method == "byte":
        tokens = BYTE_TOKENIZER.tokenize(text)
    elif method == "bpe":
        tokens = BPE_TOKENIZER.tokenize(text)
    else:
        raise ValueError(f"Unsupported tokenization method: {method}")

    if remove_stopwords and method in ["nltk", "split"]:
        tokens = [t for t in tokens if t not in stop_words]

    return tokens

def tokenize_data_frame(df, column_names:list, method="byte", remove_stopwords=False):
    """
    Tokenize a column of a DataFrame, and then add the tokenized column to the DataFrame.
    the name of the new column is the name of the original column with "_tokenized" suffix.
    """
    for column_name in column_names:
        df[column_name + "_tokenized"] = df[column_name].apply(lambda x: tokenize(x, method, remove_stopwords))
    return df