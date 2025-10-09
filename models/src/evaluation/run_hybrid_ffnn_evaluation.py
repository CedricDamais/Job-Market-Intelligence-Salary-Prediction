import logging
import os
import sys
import pandas as pd
import torch
import nltk # For potential NLTK downloads by dependencies
import json
import numpy as np

# --- Setup Project Root Path ---
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))
    if PROJECT_ROOT not in sys.path:
         sys.path.append(PROJECT_ROOT)

# Now import your project modules
from models.src.generation.neural_networks import HybridFFNN, BenchmarkModel
from models.src.evaluation.custom_evaluate import evaluate_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logging.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)


def load_vocab(filepath):
    """Load vocabulary from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_embeddings(filepath):
    """Load pretrained embeddings from a .npy file."""
    return np.load(filepath)


def run_hybrid_ffnn_evaluation(
    model_checkpoint_path: str,
    test_data_path: str,
    vocab_path: str,
    embeddings_path: str = None,
    num_samples_to_eval: int = 50,
    device_str: str = "auto"
):
    """
    Loads a trained HybridFFNN model, generates descriptions for a test set,
    and then evaluates the generated descriptions.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logging.info(f"Using device: {device}")

    logging.info(f"Loading test data from: {test_data_path}")
    if not os.path.exists(test_data_path):
        logging.error(f"Test data file not found: {test_data_path}")
        return
    try:
        if test_data_path.endswith(".parquet"):
            df_test_full = pd.read_parquet(test_data_path)
        elif test_data_path.endswith(".csv"):
            df_test_full = pd.read_csv(test_data_path)
        else:
            logging.error(f"Unsupported test data format: {test_data_path}. Please use .parquet or .csv")
            return
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        return

    if 'title' not in df_test_full.columns or 'description' not in df_test_full.columns:
        logging.error("Test data must contain 'title' and 'description' columns.")
        return
        
    logging.info(f"Loaded {len(df_test_full)} records from test data.")
    
    if num_samples_to_eval > 0 and num_samples_to_eval < len(df_test_full):
        df_test_sampled = df_test_full.sample(n=num_samples_to_eval, random_state=42)
        logging.info(f"Sampled {num_samples_to_eval} records for evaluation.")
    else:
        df_test_sampled = df_test_full
        logging.info(f"Using all {len(df_test_sampled)} records for evaluation.")

    # Load vocabulary
    logging.info(f"Loading vocabulary from: {vocab_path}")
    vocab = load_vocab(vocab_path)
    
    # Load pretrained embeddings if provided
    pretrained_embeddings = None
    if embeddings_path and os.path.exists(embeddings_path):
        logging.info(f"Loading pretrained embeddings from: {embeddings_path}")
        pretrained_embeddings = load_embeddings(embeddings_path)
    
    # Initialize model with same parameters as training
    model = HybridFFNN(
        vocab=vocab,
        hidden_size=512,
        dropout=0.5,
        num_layers=2,
        context_size=3,
        lr=1e-4,
        pretrained_embeddings=pretrained_embeddings
    )
    
    # Load model weights
    logging.info(f"Loading model from checkpoint: {model_checkpoint_path}")
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create benchmark model
    benchmark_model = BenchmarkModel(df_test_sampled, model)
    
    evaluation_data = []
    logging.info(f"Generating descriptions for {len(df_test_sampled)} samples...")
    
    for _, row in df_test_sampled.iterrows():
        job_title = row['title']
        original_description = row['description']

        # Create context window from job title
        title_words = job_title.split()
        # Take first two words, or pad with the last word if title is shorter
        if len(title_words) >= 2:
            context_window = title_words[:2]
        else:
            # If title has only one word, repeat it
            context_window = [title_words[0], title_words[0]]
        # Add _START_ token
        context_window.append("_START_")
        
        predicted_description = benchmark_model.predict_job_description(
            context_window=context_window,
            max_length=150,
            temperature=0.7
        )
        
        evaluation_data.append({
            "inputs": " ".join(context_window),
            "predictions": " ".join(predicted_description),
            "ground_truth": original_description
        })

    df_for_evaluation = pd.DataFrame(evaluation_data)
    logging.info(f"Generated {len(df_for_evaluation)} predictions.")

    if df_for_evaluation.empty:
        logging.error("No predictions were generated. Cannot proceed with evaluation.")
        return

    logging.info("Starting evaluation using custom_evaluate.evaluate_dataset...")
    try:
        results = evaluate_dataset(dataframe=df_for_evaluation) 
        logging.info("Evaluation complete.")
        if results:
            logging.info(f"MLflow Metrics: {results.metrics}")
        else:
            logging.info("evaluate_dataset did not return results.")
    except Exception as e:
        logging.error(f"Error during custom_evaluate.evaluate_dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    MODEL_CHECKPOINT_RELATIVE_PATH = "models/src/generation/model_cache/hybrid_ffnn_model.pth"
    TEST_DATA_RELATIVE_PATH = "models/src/generation/data/processed/df_test_for_evaluation.parquet"
    VOCAB_RELATIVE_PATH = "models/src/generation/data/processed/vocab.json"
    EMBEDDINGS_RELATIVE_PATH = "models/src/generation/data/processed/pretrained_embeddings.npy"
    
    MODEL_CHECKPOINT = os.path.join(PROJECT_ROOT, MODEL_CHECKPOINT_RELATIVE_PATH)
    TEST_DATA = os.path.join(PROJECT_ROOT, TEST_DATA_RELATIVE_PATH)
    VOCAB_PATH = os.path.join(PROJECT_ROOT, VOCAB_RELATIVE_PATH)
    EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, EMBEDDINGS_RELATIVE_PATH)

    NUM_SAMPLES = 10

    if not os.path.exists(MODEL_CHECKPOINT):
        logging.error(f"Model checkpoint not found: {MODEL_CHECKPOINT}")
    elif not os.path.exists(TEST_DATA):
        logging.error(f"Test data file not found: {TEST_DATA}")
        logging.error("Please create/save your test data (e.g., as df_test_for_evaluation.parquet) "
                      "in the specified path. It must include 'title' and 'description' columns.")
    elif not os.path.exists(VOCAB_PATH):
        logging.error(f"Vocabulary file not found: {VOCAB_PATH}")
    else:
        run_hybrid_ffnn_evaluation(
            model_checkpoint_path=MODEL_CHECKPOINT,
            test_data_path=TEST_DATA,
            vocab_path=VOCAB_PATH,
            embeddings_path=EMBEDDINGS_PATH,
            num_samples_to_eval=NUM_SAMPLES
        ) 