# -*- coding: utf-8 -*-
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import nltk

import sys
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logging.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)

class LSTMLanguageModelS(nn.Module):
    def __init__(self, embedding_model, decoder_vocab_size, embedding_dim=384,
                 decoder_embed_dim=256, hidden_dim=512, num_layers=2, dropout_p=0.2):
        super().__init__()

        self.title_embedding_transformer = embedding_model

        # Freeze the transformer parameters if the model is passed
        # I dont wan the model to forget the rich semantics from the transformer
        if self.title_embedding_transformer is not None:
            for param in self.title_embedding_transformer.parameters():
                param.requires_grad = False

        # The different dimensions of the neural network
        self.embedding_dim = embedding_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.decoder_vocab_size = decoder_vocab_size

        # Linear projections for initializing LSTM states from title embedding
        # Corrected to use hidden_dim * num_layers for multi-layer LSTM state initialization
        self.title_to_hidden = nn.Linear(embedding_dim, hidden_dim * num_layers)
        self.title_to_cell = nn.Linear(embedding_dim, hidden_dim * num_layers)

        # Embedding layer for decoder input tokens (descriptions)
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, decoder_embed_dim)

        # Projection for combined input to LSTM
        # (decoder_token_embedding + title_embedding) -> lstm_input_size
        self.lstm_input_proj = nn.Linear(decoder_embed_dim + embedding_dim, decoder_embed_dim)
        
        self.dropout = nn.Dropout(dropout_p)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=decoder_embed_dim, # Input to LSTM is the projected combined embedding
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0
        )

        # Final output projection (deeper than a single linear layer)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_p), # Apply dropout within the projection too
            nn.Linear(hidden_dim * 2, decoder_vocab_size)
        )

    def forward(self, title_embeddings, decoder_inputs, lengths=None):
        batch_size = title_embeddings.size(0)

        # 1. Initialize LSTM states from title embeddings
        h_0_proj = self.title_to_hidden(title_embeddings)  # [batch_size, hidden_dim * num_layers]
        c_0_proj = self.title_to_cell(title_embeddings)    # [batch_size, hidden_dim * num_layers]

        # Reshape for multi-layer LSTM: [num_layers, batch_size, hidden_dim]
        h_0 = h_0_proj.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c_0 = c_0_proj.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        # 2. Embed decoder inputs (description tokens)
        embedded_desc_tokens = self.decoder_embedding(decoder_inputs)  # [batch_size, seq_len, decoder_embed_dim]
        embedded_desc_tokens = self.dropout(embedded_desc_tokens)

        # 3. Condition LSTM input with title embeddings at each step
        # I did this because I want the model to be able to use the title embedding at each step of the LSTM
        # To keep the context of the title in the description
        # Expand title_embeddings to match the sequence length of decoder_inputs
        seq_len = embedded_desc_tokens.size(1)
        title_expanded = title_embeddings.unsqueeze(1).expand(-1, seq_len, -1) # [batch_size, seq_len, embedding_dim]

        # Concatenate current token embedding with title embedding
        combined_embeddings = torch.cat([embedded_desc_tokens, title_expanded], dim=-1) # [batch_size, seq_len, decoder_embed_dim + embedding_dim]
        
        # Project this combined embedding to be the input size for LSTM
        # This is done to reduce the dimensionality of the input to the LSTM
        # This is a common practice in LSTM models
        lstm_input = self.lstm_input_proj(combined_embeddings) # Remember this mfk size [batch_size, seq_len, decoder_embed_dim]

        # 4. Handle packed sequences if lengths are provided (typically during training)
        if lengths is not None:
            # Ensure lengths tensor is on CPU I dont want to have any GPU memory issues
            lstm_input = torch.nn.utils.rnn.pack_padded_sequence(
                lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=True
            )

        # 5. Pass through LSTM
        # This is the main part of the LSTM
        lstm_out, _ = self.lstm(lstm_input, (h_0, c_0)) # lstm_out shape: [batch_size, seq_len, hidden_dim] or PackedSequence

        # 6. Unpack sequences if they were packed
        if lengths is not None:
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # 7. Apply activation and project to vocabulary size
        # The ReLU is applied to the raw LSTM output before the final projection layers.
        hidden_activated = torch.relu(lstm_out)
        output_logits = self.output_projection(hidden_activated) # [batch_size, seq_len, decoder_vocab_size]

        return output_logits

    def predict_next_token(self, title_embedding, current_token_id_tensor, hidden_state_tuple=None):
        # Ensure title_embedding has batch dimension [1, embedding_dim] wack ahh errror
        if title_embedding.dim() == 1:
            title_embedding = title_embedding.unsqueeze(0)
        
        # Ensure current_token_id_tensor is correctly shaped [batch_size, 1] (usually batch_size=1 for generation)
        # And on the correct device, and long type for embedding.
        current_token_id_tensor = current_token_id_tensor.to(next(self.parameters()).device).long()
        if current_token_id_tensor.dim() == 0: # scalar
            current_token_id_tensor = current_token_id_tensor.view(1, 1) # Shape [1, 1]
        elif current_token_id_tensor.dim() == 1: # e.g. [1]
            current_token_id_tensor = current_token_id_tensor.unsqueeze(0) # Shape [1, 1] if batch_size is 1

        batch_size = title_embedding.size(0) # Should be 1 during typical generation

        # Initialize hidden state if first step
        # This is done to initialize the hidden state of the LSTM that means that 
        # The model can use the title embedding at each step keeping the context of the title
        if hidden_state_tuple is None:
            h_0_proj = self.title_to_hidden(title_embedding)
            c_0_proj = self.title_to_cell(title_embedding)
            h_0 = h_0_proj.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
            c_0 = c_0_proj.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
            hidden_state_tuple = (h_0, c_0)

        # Embed the current token
        embedded_token = self.decoder_embedding(current_token_id_tensor) # [batch_size, 1, decoder_embed_dim]

        # Condition with title embedding (similar to forward pass but for a single step)
        # title_embedding is [batch_size, embedding_dim], unsqueeze to add seq_len of 1
        title_expanded_for_step = title_embedding.unsqueeze(1) # [batch_size, 1, embedding_dim]
        
        combined_for_step = torch.cat([embedded_token, title_expanded_for_step], dim=-1)
        lstm_input_at_step = self.lstm_input_proj(combined_for_step) # [batch_size, 1, decoder_embed_dim]

        # Pass through LSTM
        # lstm_out_at_step will be [batch_size, 1, hidden_dim]
        lstm_out_at_step, new_hidden_state_tuple = self.lstm(lstm_input_at_step, hidden_state_tuple)

        # Project to vocabulary
        # Apply activation consistent with forward pass
        hidden_activated_at_step = torch.relu(lstm_out_at_step)
        # Squeeze the sequence dimension (dim=1) as we are predicting for a single token
        output_logits = self.output_projection(hidden_activated_at_step.squeeze(1)) # [batch_size, decoder_vocab_size]
        
        return output_logits, new_hidden_state_tuple


def load_inference_model(checkpoint_path, device='cpu', YourLSTMLanguageModelClass=None):
    """
    Loads a LSTMLanguageModelS checkpoint for inference.
    """
    if YourLSTMLanguageModelClass is None:
        logging.error("You must provide the model class (e.g., LSTMLanguageModelS) to `YourLSTMLanguageModelClass` argument.")
        return None, None, None
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found at {checkpoint_path}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
        return None, None, None

    st_name = checkpoint.get('sentence_transformer_name')
    constructor_args = checkpoint.get('constructor_args')
    word_to_idx = checkpoint.get('word_to_idx')
    model_state_dict = checkpoint.get('model_state_dict')

    if not all([st_name, constructor_args, word_to_idx, model_state_dict]):
        logging.error("Checkpoint is missing one or more required keys: "
                      "'sentence_transformer_name', 'constructor_args', 'word_to_idx', 'model_state_dict'.")
        return None, None, None

    try:
        sentence_transformer_instance = SentenceTransformer(st_name, device=device)
    except Exception as e:
        logging.error(f"Error initializing SentenceTransformer model '{st_name}': {e}")
        return None, None, None

    st_embedding_dim = sentence_transformer_instance.get_sentence_embedding_dimension()
    if constructor_args.get('embedding_dim') != st_embedding_dim:
        logging.warning(
            f"Overriding 'embedding_dim' in constructor_args (was {constructor_args.get('embedding_dim')}) "
            f"with dimension from loaded SentenceTransformer: {st_embedding_dim}"
        )
        constructor_args['embedding_dim'] = st_embedding_dim
    
    if constructor_args.get('decoder_vocab_size') != len(word_to_idx):
        logging.warning(
            f"Overriding 'decoder_vocab_size' in constructor_args (was {constructor_args.get('decoder_vocab_size')}) "
            f"with size from loaded word_to_idx: {len(word_to_idx)}"
        )
        constructor_args['decoder_vocab_size'] = len(word_to_idx)

    try:
        loaded_model = YourLSTMLanguageModelClass(
            embedding_model=sentence_transformer_instance,
            **constructor_args
        ).to(device)
    except Exception as e:
        logging.error(f"Error initializing model with constructor_args: {e}")
        logging.error(f"Args: {constructor_args}")
        return None, None, None

    try:
        loaded_model.load_state_dict(model_state_dict)
    except Exception as e:
        logging.error(f"Error loading model state_dict. Ensure architecture matches: {e}")
        return None, None, None
        
    loaded_model.eval()
    logging.info(f"Model loaded successfully from: {checkpoint_path}")
    return loaded_model, word_to_idx, sentence_transformer_instance


def generate_description(model, title, word_to_idx, embedding_model_st_instance,
                         max_len=200, temperature=0.7, top_k=20):
    """
    Generate text with temperature and top-k sampling.
    It works much better then greedy search.
    """
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    device = next(model.parameters()).device
    model.eval()

    unk_idx = word_to_idx.get('<UNK>', -1)
    pad_idx = word_to_idx.get('<PAD>', -1)
    start_token_val = word_to_idx.get('__START__', -1)
    end_token_val = word_to_idx.get('__END__', -1)

    indices_to_filter_logits = [idx for idx in [unk_idx, pad_idx, start_token_val] if idx >= 0]

    with torch.no_grad():
        if isinstance(title, str):
            title_embedding = embedding_model_st_instance.encode([title], convert_to_tensor=True, device=device)
        else:
            title_embedding = title.to(device)

        if title_embedding.dim() == 1:
            title_embedding = title_embedding.unsqueeze(0)

    if start_token_val == -1:
        logging.error("Critical: '__START__' token not found in word_to_idx.")
        return "[ERROR: Missing __START__ token in vocabulary]"
        
    current_token_tensor = torch.tensor([[start_token_val]], device=device, dtype=torch.long)
    generated_indices = []
    hidden_state = None

    for _ in range(max_len):
        logits, hidden_state = model.predict_next_token(title_embedding, current_token_tensor, hidden_state)
        
        if temperature <= 0: 
            _, next_token_idx_tensor = torch.topk(logits, 1, dim=-1)
            next_token_idx = next_token_idx_tensor.item()
        else:
            logits = logits / temperature
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                if not top_k_logits.numel():
                    logging.warning("Top_k filtering resulted in empty logits, picking most probable token if any.")
                    _, next_token_idx_tensor = torch.topk(logits.view(-1), 1)
                    if next_token_idx_tensor.numel():
                        next_token_idx = next_token_idx_tensor.item()
                    else:
                        logging.error("Cannot select any token, vocabulary might be an issue.")
                        break
                else:
                    # Sa cree un nouveau tensor avec des -inf partout
                    filtered_logits = torch.full_like(logits, float('-inf'))
                    # Et ici je remet les logits qui sont dans top_k_indices a la valeur de top_k_logits
                    filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
                    logits = filtered_logits

                    for idx_to_filter in indices_to_filter_logits:
                        if 0 <= idx_to_filter < logits.size(1):
                            logits[0, idx_to_filter] = float('-inf')

                    probabilities = F.softmax(logits, dim=-1)

                    # Shiii si sa arrive weird yo
                    if not probabilities.numel() or torch.isinf(probabilities).any() or torch.isnan(probabilities).any() or probabilities.sum() == 0:
                        logging.warning("Probabilities are NaN, Inf or sum to zero. Attempting to pick highest logit or breaking.")
                        non_inf_logits = logits[0][torch.isfinite(logits[0])]
                        if non_inf_logits.numel() > 0:
                            next_token_idx = torch.argmax(logits[0]).item()
                        else:
                            logging.error("All logits are -Inf. Cannot sample. Stopping generation.")
                            break
                    else:
                        next_token_idx = torch.multinomial(probabilities, 1).item()


        if next_token_idx == end_token_val:
            break

        generated_indices.append(next_token_idx)
        current_token_tensor = torch.tensor([[next_token_idx]], device=device, dtype=torch.long)

    # Filter les tokens (UNK, PAD, START, END)
    final_word_filter_indices = [idx for idx in [unk_idx, pad_idx, start_token_val, end_token_val] if idx >=0]
    words = [idx_to_word.get(idx) for idx in generated_indices if idx not in final_word_filter_indices and idx in idx_to_word]
    return " ".join(filter(None, words))


def lstm_generate_description(job_title):
    logging.info("Starting description generation script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_file_path = os.path.join(script_dir, "model_cache/my_lstm_deep_for_inference.pt")

    if not os.path.exists(checkpoint_file_path):
        logging.error(f"Model checkpoint '{checkpoint_file_path}' not found. " 
                      "Please ensure the path is correct and the model file exists.")
        logging.error("If you haven't trained and saved a model yet, you'll need to do that first.")
        return

    model, word_to_idx, st_model = load_inference_model(
        checkpoint_path=checkpoint_file_path,
        device=device,
        YourLSTMLanguageModelClass=LSTMLanguageModelS
    )

    if not model:
        logging.error("Failed to load the model. Exiting.")
        return

    titles_to_test = [
        job_title
    ]

    for job_title in titles_to_test:
        logging.info(f"Generating description for title: '{job_title}'")
        generated_description = generate_description(
            model=model,
            title=job_title,
            word_to_idx=word_to_idx,
            embedding_model_st_instance=st_model,
            max_len=200,
            temperature=1.2,    
            top_k=20
        )
        logging.info(f"--- Generated Description for '{job_title}' ---")
        print(generated_description)
        logging.info(f"--- End of Description for '{job_title}' ---")
        return generated_description