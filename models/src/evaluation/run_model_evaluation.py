import logging
import os
import sys
import pandas as pd
import torch
import nltk # For potential NLTK downloads by dependencies

# --- Setup Project Root Path ---
# This allows the script to be run from the project root (e.g., using poetry run)
# and still correctly import modules from within the 'models' package.
try:
    # Assuming this script is in models/src/evaluation/
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except NameError: # __file__ is not defined (e.g. in a notebook directly)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))
    if PROJECT_ROOT not in sys.path:
         sys.path.append(PROJECT_ROOT)

# Now import your project modules
from models.src.generation.lstm import LSTMLanguageModelS, load_inference_model, generate_description
from models.src.evaluation.custom_evaluate import evaluate_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logging.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)


def run_evaluation(
    model_checkpoint_path: str,
    test_data_path: str,
    num_samples_to_eval: int = 50,
    device_str: str = "auto"
):
    """
    Loads a trained model, generates descriptions for a test set,
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

    logging.info(f"Loading model from checkpoint: {model_checkpoint_path}")
    model, word_to_idx, st_model = load_inference_model(
        checkpoint_path=model_checkpoint_path,
        device=device,
        YourLSTMLanguageModelClass=LSTMLanguageModelS
    )
    if not model:
        logging.error("Failed to load the model. Exiting.")
        return
    logging.info("Model loaded successfully.")

    evaluation_data = []
    logging.info(f"Generating descriptions for {len(df_test_sampled)} samples...")
    
    for _, row in df_test_sampled.iterrows():
        job_title = row['title']
        original_description = row['description']

        predicted_description = generate_description(
            model=model,
            title=job_title,
            word_to_idx=word_to_idx,
            embedding_model_st_instance=st_model,
            max_len=150,
            temperature=0.7,
            top_k=20
        )
        
        evaluation_data.append({
            "inputs": job_title,
            "predictions": predicted_description,
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
    MODEL_CHECKPOINT_RELATIVE_PATH = "models/src/generation/model_cache/my_lstm_deep_for_inference.pt"
    TEST_DATA_RELATIVE_PATH = "models/src/generation/data/processed/df_test_for_evaluation.parquet"
    
    MODEL_CHECKPOINT = os.path.join(PROJECT_ROOT, MODEL_CHECKPOINT_RELATIVE_PATH)
    TEST_DATA = os.path.join(PROJECT_ROOT, TEST_DATA_RELATIVE_PATH)

    NUM_SAMPLES = 10

    if not os.path.exists(MODEL_CHECKPOINT):
        logging.error(f"Model checkpoint not found: {MODEL_CHECKPOINT}")
    elif not os.path.exists(TEST_DATA):
        logging.error(f"Test data file not found: {TEST_DATA}")
        logging.error("Please create/save your test data (e.g., as df_test_for_evaluation.parquet) "
                      "in the specified path. It must include 'title' and 'description' columns.")
    else:
        run_evaluation(
            model_checkpoint_path=MODEL_CHECKPOINT,
            test_data_path=TEST_DATA,
            num_samples_to_eval=NUM_SAMPLES
        ) 