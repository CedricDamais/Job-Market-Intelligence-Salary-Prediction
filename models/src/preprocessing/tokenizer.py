"""
Utility functions for loading and using Hugging Face tokenizers.
"""

from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import os
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Downloads for NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.warning(f"Could not download NLTK data: {e}")

# Stopwords
try:
    stop_words = set(stopwords.words('english'))
except Exception as e:
    logging.warning(f"Could not load stopwords: {e}")
    stop_words = set()

# Preloaded tokenizers
try:
    BYTE_TOKENIZER = AutoTokenizer.from_pretrained("gpt2")  # Byte-level BPE
    BPE_TOKENIZER = AutoTokenizer.from_pretrained("roberta-base")  # WordPiece/BPE hybrid
except Exception as e:
    logging.warning(f"Could not load pretrained tokenizers: {e}")
    BYTE_TOKENIZER = None
    BPE_TOKENIZER = None

# Cache directory for tokenizers (optional, but good practice)
CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "tokenizers_custom"
)
os.makedirs(CACHE_DIR, exist_ok=True)


def load_tokenizer(
    tokenizer_name: str = "gpt2", cache_dir: str | None = CACHE_DIR
) -> Tokenizer:
    """
    Loads a pre-trained tokenizer from the Hugging Face Hub.

    Args:
        tokenizer_name: The name of the tokenizer on the Hugging Face Hub
                        (e.g., "gpt2", "bert-base-uncased").
        cache_dir: Directory to cache downloaded tokenizer files. Defaults to
                   ~/.cache/huggingface/tokenizers_custom. Set to None to use
                   the default cache location of the huggingface_hub library.

    Returns:
        An instance of the Hugging Face Tokenizer.

    Raises:
        Exception: If the tokenizer cannot be loaded.
    """
    logging.info(f"Attempting to load tokenizer: {tokenizer_name}")
    try:
        try:
            config_path = hf_hub_download(
                repo_id=tokenizer_name,
                filename="tokenizer.json",
                cache_dir=cache_dir,
                library_name="nlp-linkedin-offers",
                library_version="0.1.0",
            )
            tokenizer = Tokenizer.from_file(config_path)
            logging.info(
                f"Successfully loaded tokenizer '{tokenizer_name}' from tokenizer.json."
            )
            return tokenizer
        except Exception as e:
            logging.warning(
                f"Could not load '{tokenizer_name}' directly from tokenizer.json: {e}. "
                "Attempting legacy loading (might be slower or require transformers)."
            )
            raise FileNotFoundError(
                f"Could not find or load tokenizer.json for '{tokenizer_name}'. "
                "Consider installing the 'transformers' library and using AutoTokenizer for broader compatibility."
            )

    except Exception as e:
        logging.error(f"Failed to load tokenizer '{tokenizer_name}': {e}")
        raise


def tokenize(text, method="nltk", remove_stopwords=False):
    """
    Tokenize text using various methods.
    
    Args:
        text: Input text to tokenize
        method: Tokenization method ("nltk", "split", "byte", "bpe")
        remove_stopwords: Whether to remove stopwords (only for nltk and split methods)
    
    Returns:
        List of tokens
    """
    if method == "nltk":
        tokens = word_tokenize(text)
    elif method == "split":
        tokens = text.split()
    elif method == "byte" and BYTE_TOKENIZER is not None:
        tokens = BYTE_TOKENIZER.tokenize(text)
    elif method == "bpe" and BPE_TOKENIZER is not None:
        tokens = BPE_TOKENIZER.tokenize(text)
    else:
        raise ValueError(f"Unsupported tokenization method: {method}")

    if remove_stopwords and method in ["nltk", "split"]:
        tokens = [t for t in tokens if t not in stop_words]

    return tokens


def tokenize_data_frame(df, column_names: list, method="byte", remove_stopwords=False):
    """
    Tokenize a column of a DataFrame, and then add the tokenized column to the DataFrame.
    the name of the new column is the name of the original column with "_tokenized" suffix.
    
    Args:
        df: DataFrame to process
        column_names: List of column names to tokenize
        method: Tokenization method to use
        remove_stopwords: Whether to remove stopwords
    
    Returns:
        DataFrame with additional tokenized columns
    """
    for column_name in column_names:
        df[column_name + "_tokenized"] = df[column_name].apply(
            lambda x: tokenize(x, method, remove_stopwords)
        )
    return df
