from nltk.tokenize import word_tokenize  # More robust than split(" ")
import re
from collections import Counter

def get_pruned_vocab(df, column, top_k=50000, min_freq=5):
    token_counter = Counter()
    special_tokens = {"__START__", "__END__", "<UNK>", "<PAD>"}

    for text in df[column]:
        # Normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

        # Tokenize with NLTK (handles contractions, hyphens better)
        tokens = word_tokenize(text)

        # Filter special tokens from data
        tokens = [t for t in tokens if t not in special_tokens]
        token_counter.update(tokens)

    # Filter by min frequency AND top_k
    filtered_tokens = [token for token, count in token_counter.most_common(top_k)
                       if count >= min_freq]

    # Build vocab (special tokens first)
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    vocab.update({token: idx + len(special_tokens) for idx, token in enumerate(filtered_tokens)})

    return vocab, len(vocab)