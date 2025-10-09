from metrics import (
    grammar_metric,
    long_range_coherence_metric,
    novelty_metric,
    semantic_richness_metric,
    professionalism_metric,
)
import mlflow
import pandas as pd
import os
import dotenv

dotenv.load_dotenv()

def evaluate_dataset(dataframe: str | pd.DataFrame):
    """
    Evaluate the dataset statically using various metrics.
    """
    if isinstance(dataframe, str):
        dataframe = pd.read_csv(dataframe)
        # ensure that dataframe contains 'predictions' and 'ground_truth' columns
        if (
            "predictions" not in dataframe.columns
            or "ground_truth" not in dataframe.columns
        ):
            raise ValueError(
                "DataFrame must contain 'predictions' and 'ground_truth' columns."
            )
        # Get 50 rows of the dataframe
        dataframe = dataframe.sample(n=50, random_state=42)
    # Evaluate the dataset using the defined metrics
    extra_metrics = [
        grammar_metric,
        long_range_coherence_metric,
        novelty_metric,
        semantic_richness_metric,
        professionalism_metric,
        mlflow.metrics.rougeL(),
        mlflow.metrics.bleu(),
        mlflow.metrics.flesch_kincaid_grade_level(),
    ]
    
    # Use local file system for MLflow tracking instead of a server
    mlflow_tracking_dir = os.path.join(os.getcwd(), "mlruns")
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")
    mlflow.set_experiment("genai-eval")
    
    # Evaluate the dataset using the defined metrics
    with mlflow.start_run():
        results = mlflow.evaluate(
            model=None,
            data=dataframe,
            predictions="predictions",
            targets="ground_truth",
            extra_metrics=extra_metrics,
            evaluator_config={"col_mapping": {"inputs": "inputs"}},
        )
    print(results.metrics)
    print(results.tables["eval_results_table"])

    return results