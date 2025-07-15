from datetime import datetime
import os
import mlflow
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import pandas as pd
from get_vector_db import get_vector_db

from evaluation_metrics import TextSummarizationEvaluator

# Initialize the evaluator once (outside your loop)
evaluator = TextSummarizationEvaluator()

LLM_MODEL = os.getenv("LLM_MODEL", "mistral")


# Function to get the prompt templates for generating alternative questions and answering based on context
def get_prompt():
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    return QUERY_PROMPT, prompt


with open("opekepe.md", "r", encoding="utf-8") as file:
    reference_text = file.read()


# Main function to handle the query process
def query(input):
    print(f"input: {type(input)}")
    # mlflow.log_param("question", input)
    with mlflow.start_run(
        run_name=f"simple_query_metrics"
        # run_name=f"simple_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ) as run:
        # run.info.run_id

        mlflow.log_param("input", input)

        if input:
            # Initialize the language model with the specified model name
            llm = ChatOllama(model=LLM_MODEL)
            # Get the vector database instance
            db = get_vector_db()
            # Get the prompt templates
            QUERY_PROMPT, prompt = get_prompt()

            # Set up the retriever to generate multiple queries using the language model and the query prompt
            retriever = MultiQueryRetriever.from_llm(
                db.as_retriever(), llm, prompt=QUERY_PROMPT
            )

            # Define the processing chain to retrieve context, generate the answer, and parse the output
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            response = chain.invoke(input)

            # ==== Evaluation ====
            text_metrics = {
                "ari_grade_level": mlflow.metrics.ari_grade_level(),
                "flesch_kincaid_grade_level": mlflow.metrics.flesch_kincaid_grade_level(),
            }

            # Prepare data for evaluation
            eval_data = pd.DataFrame(
                {"predictions": [response], "targets": [reference_text]}
            )

            results = {}
            for metric_name, metric_func in text_metrics.items():
                try:
                    # Calculate metric score
                    score = _calculate_metric_score(metric_func, eval_data)
                    print(f"{metric_name} score: {score}")
                    results[metric_name] = score

                except Exception as e:
                    # logger.error("Failed to calculate %s: %s", metric_name, str(e))
                    results[metric_name] = None

            mlflow.log_param("response", response)

            question_answering_metrics = {
                "rouge1": mlflow.metrics.rouge1(),
                "rouge2": mlflow.metrics.rouge2(),
                "rougeL": mlflow.metrics.rougeL(),
                "rougeLsum": mlflow.metrics.rougeLsum(),
                "toxicity": mlflow.metrics.toxicity(),
                "latency": mlflow.metrics.latency(),
                # "retrieval_precision": mlflow.metrics.re,
                # "faithfulness": mlflow.metrics.f,
                # "response_relevancy": response_relevancy_metric,
            }
            for metric_name, metric_func in question_answering_metrics.items():
                try:
                    # Calculate metric score
                    score = _calculate_metric_score(metric_func, eval_data)
                    print(f"{metric_name} score: {score}")
                    results[metric_name] = score

                except Exception as e:
                    # logger.error("Failed to calculate %s: %s", metric_name, str(e))
                    results[metric_name] = None

            valid_results = {k: v for k, v in results.items() if v is not None}
            if valid_results:
                mlflow.log_metrics(valid_results)
            evaluation_response = {
                "index": "index_name",
                "metrics": {
                    "retrieval_precision": "precision_result",
                    "faithfulness": "faithfulness_result",
                    "response_relevancy": "response_relevancy_result",
                },
                "evaluation_sample": {
                    "question": "ragas_sample.user_input",
                    "response": "ragas_sample.response",
                    "reference": "ragas_sample.reference",
                    "retrieved_contexts": "context",
                    "reference_contexts": "ragas_sample.reference_contexts",
                },
            }

            mlflow.log_dict(evaluation_response, "evaluation_response")
            # mlflow.log_dict(evaluation_response, "evaluation_response")
            # mlflow.log_metrics(results)

            return response

    return None


def _calculate_metric_score(metric_func, eval_data: pd.DataFrame) -> float:
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
    print(f"eval_results: {eval_results}")
    # Extract the metric score from results
    metric_name = metric_func.name
    return eval_results.metrics[metric_name]


def evaluation(input):
    def _build_metrics():
        metrics = {}

        try:
            # ROUGE metrics - primary metric for summarization
            metrics["rouge_1"] = mlflow.metrics.rouge1()
            metrics["rouge_2"] = mlflow.metrics.rouge2()
            metrics["rouge_l"] = mlflow.metrics.rougeL()
            metrics["rouge_lsum"] = mlflow.metrics.rougeLsum()

        except Exception as e:
            print(f"Error loading ROUGE metrics: {e}")

        try:
            metrics["ari_grade_level"] = mlflow.metrics.ari_grade_level()
            metrics["flesch_kincaid_grade_level"] = (
                mlflow.metrics.flesch_kincaid_grade_level()
            )
        except Exception as e:
            print(f"Failed to load readability metrics: {e}")

        try:
            metrics["toxicity"] = mlflow.metrics.toxicity()
        except Exception as e:
            print(f"Failed to load toxicity metrics: {e}")
        return metrics

    metrics = _build_metrics()
