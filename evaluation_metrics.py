"""
MLflow Evaluation Metrics Module for Text Summarization
========================================================

This module provides evaluation metrics for text summarization tasks using MLflow.
It implements ROUGE, toxicity, ARI grade level, and Flesch-Kincaid grade level metrics.

Requirements:
- mlflow
- evaluate
- torch
- transformers
- nltk
- rouge-score
- textstat

Install with:
pip install mlflow evaluate torch transformers nltk rouge-score textstat
"""

import mlflow
import mlflow.metrics
from typing import Dict, Any, Optional
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextSummarizationEvaluator:
    """
    Evaluator class for text summarization tasks using MLflow metrics.

    This class provides methods to evaluate generated summaries against reference texts
    using various metrics appropriate for text summarization tasks.
    """

    def __init__(self):
        """Initialize the evaluator with required metrics."""
        self.metrics = self._setup_metrics()
        logger.info(
            "TextSummarizationEvaluator initialized with metrics: %s",
            list(self.metrics.keys()),
        )

    def _setup_metrics(self) -> Dict[str, Any]:
        """
        Set up MLflow metrics for text summarization evaluation.

        Returns:
            Dict[str, Any]: Dictionary of metric names to MLflow metric objects
        """
        metrics = {}

        try:
            # ROUGE metrics - primary metric for summarization
            metrics["rouge_1"] = mlflow.metrics.rouge1()
            metrics["rouge_2"] = mlflow.metrics.rouge2()
            metrics["rouge_l"] = mlflow.metrics.rougeL()
            metrics["rouge_lsum"] = mlflow.metrics.rougeLsum()
            logger.info("ROUGE metrics loaded successfully")

        except Exception as e:
            logger.error("Failed to load ROUGE metrics: %s", str(e))
            logger.info(
                "Make sure you have installed: pip install evaluate nltk rouge-score"
            )
            return

        try:
            # Toxicity metric - measures harmful content
            metrics["toxicity"] = mlflow.metrics.toxicity()
            logger.info("Toxicity metric loaded successfully")

        except Exception as e:
            logger.error("Failed to load toxicity metric: %s", str(e))
            logger.info(
                "Make sure you have installed: pip install evaluate torch transformers"
            )

        try:
            # Readability metrics
            metrics["ari_grade_level"] = mlflow.metrics.ari_grade_level()
            metrics["flesch_kincaid_grade_level"] = (
                mlflow.metrics.flesch_kincaid_grade_level()
            )
            logger.info("Readability metrics loaded successfully")

        except Exception as e:
            logger.error("Failed to load readability metrics: %s", str(e))
            logger.info("Make sure you have installed: pip install textstat")

        return metrics

    def evaluate_summary(
        self, generated_summary: str, reference_text: str, run_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a generated summary against a reference text.

        Args:
            generated_summary (str): The generated summary text
            reference_text (str): The original reference text
            run_id (str, optional): MLflow run ID to log metrics to

        Returns:
            Dict[str, float]: Dictionary of metric names to scores
        """
        if not generated_summary or not reference_text:
            logger.warning("Empty generated_summary or reference_text provided")
            return {}

        # Prepare data for evaluation
        eval_data = pd.DataFrame(
            {"predictions": [generated_summary], "targets": [reference_text]}
        )

        results = {}

        # Evaluate each metric
        for metric_name, metric_func in self.metrics.items():
            try:
                # Calculate metric score
                score = self._calculate_metric_score(metric_func, eval_data)
                results[metric_name] = score

                # Log to MLflow if run_id is provided
                if run_id:
                    mlflow.log_metric(metric_name, score, run_id=run_id)

                logger.info("Metric %s: %.4f", metric_name, score)

            except Exception as e:
                logger.error("Failed to calculate %s: %s", metric_name, str(e))
                results[metric_name] = None

        return results

    def _calculate_metric_score(self, metric_func, eval_data: pd.DataFrame) -> float:
        """
        Calculate the score for a specific metric.

        Args:
            metric_func: MLflow metric function
            eval_data (pd.DataFrame): Evaluation data with predictions and targets

        Returns:
            float: Calculated metric score
        """
        # Use MLflow's evaluate function to calculate the metric
        eval_results = mlflow.evaluate(
            data=eval_data,
            targets="targets",
            predictions="predictions",
            extra_metrics=[metric_func],
            evaluator_config={
                "col_mapping": {"predictions": "predictions", "targets": "targets"}
            },
        )

        # Extract the metric score from results
        metric_name = metric_func.name
        return eval_results.metrics[metric_name]

    def batch_evaluate(
        self,
        generated_summaries: list,
        reference_texts: list,
        run_id: Optional[str] = None,
    ) -> Dict[str, list]:
        """
        Evaluate multiple summaries in batch.

        Args:
            generated_summaries (list): List of generated summary texts
            reference_texts (list): List of reference texts
            run_id (str, optional): MLflow run ID to log metrics to

        Returns:
            Dict[str, list]: Dictionary of metric names to lists of scores
        """
        if len(generated_summaries) != len(reference_texts):
            raise ValueError(
                "Generated summaries and reference texts must have the same length"
            )

        batch_results = {metric_name: [] for metric_name in self.metrics.keys()}

        for i, (summary, reference) in enumerate(
            zip(generated_summaries, reference_texts)
        ):
            logger.info("Evaluating summary %d/%d", i + 1, len(generated_summaries))

            results = self.evaluate_summary(summary, reference, run_id)

            for metric_name, score in results.items():
                batch_results[metric_name].append(score)

        # Log average scores if run_id is provided
        if run_id:
            for metric_name, scores in batch_results.items():
                valid_scores = [s for s in scores if s is not None]
                if valid_scores:
                    avg_score = sum(valid_scores) / len(valid_scores)
                    mlflow.log_metric(f"avg_{metric_name}", avg_score, run_id=run_id)

        return batch_results

    def get_metric_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all available metrics.

        Returns:
            Dict[str, str]: Dictionary of metric names to descriptions
        """
        descriptions = {
            "rouge_1": "ROUGE-1: Overlap of unigrams between generated and reference text",
            "rouge_2": "ROUGE-2: Overlap of bigrams between generated and reference text",
            "rouge_l": "ROUGE-L: Longest common subsequence between generated and reference text",
            "rouge_lsum": "ROUGE-Lsum: ROUGE-L calculated at summary level",
            "toxicity": "Toxicity: Measures presence of harmful/toxic content (0-1, lower is better)",
            "ari_grade_level": "ARI Grade Level: Automated Readability Index grade level",
            "flesch_kincaid_grade_level": "Flesch-Kincaid Grade Level: Reading grade level",
        }
        return descriptions

    def available_metrics(self) -> list:
        """
        Get list of successfully loaded metrics.

        Returns:
            list: List of available metric names
        """
        return list(self.metrics.keys())


# Convenience function for quick evaluation
def evaluate_text_summary(
    generated_summary: str, reference_text: str, run_id: Optional[str] = None
) -> Dict[str, float]:
    """
    Convenience function to quickly evaluate a single summary.

    Args:
        generated_summary (str): The generated summary text
        reference_text (str): The original reference text
        run_id (str, optional): MLflow run ID to log metrics to

    Returns:
        Dict[str, float]: Dictionary of metric names to scores
    """
    evaluator = TextSummarizationEvaluator()
    return evaluator.evaluate_summary(generated_summary, reference_text, run_id)


# Example usage and setup verification
def verify_setup():
    """
    Verify that all required packages are installed and metrics can be loaded.
    """
    print("Verifying MLflow Text Summarization Evaluator setup...")

    try:
        evaluator = TextSummarizationEvaluator()
        available = evaluator.available_metrics()

        print(f"✓ Successfully loaded {len(available)} metrics:")
        for metric in available:
            print(f"  - {metric}")

        descriptions = evaluator.get_metric_descriptions()
        print("\nMetric descriptions:")
        for metric in available:
            print(f"  {metric}: {descriptions.get(metric, 'No description available')}")

        return True

    except Exception as e:
        print(f"✗ Setup verification failed: {e}")
        return False


if __name__ == "__main__":
    verify_setup()
