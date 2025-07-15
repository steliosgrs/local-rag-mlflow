import os
from groq import Groq
import pandas as pd
import mlflow
from dotenv import load_dotenv
from configs import prompts_to_test
from evaluation_metrics import TextSummarizationEvaluator

# Initialize the evaluator once (outside your loop)
evaluator = TextSummarizationEvaluator()


def check():
    if evaluator:
        exit()


# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print(api_key)
# Turn on auto tracing for Groq by calling mlflow.groq.autolog()
mlflow.groq.autolog()

mlflow.set_experiment("Groq Evalution")


with open("opekepe.md", "r", encoding="utf-8") as file:
    article = file.read()
    # print(article)


@mlflow.trace(
    name="llm_judge",
)
def evaluate_summary_with_mock_llm_judge(
    original_text: str, prompt_used: str, generated_summary: str
) -> dict:
    """
    A mock LLM judge to evaluate the quality of a generated summary.
    In a real application, this would involve calling another LLM (e.g., GPT-4, Claude)
    with a specific prompt to rate the summary.
    """
    evaluation_results = {
        "conciseness_score": 0.0,
        "relevance_score": 0.0,
        "coherence_score": 0.0,
        "overall_score": 0.0,
    }

    # Simulate conciseness: shorter summaries (within reason) get higher scores
    original_word_count = len(original_text.split())
    summary_word_count = len(generated_summary.split())

    if summary_word_count < 0.2 * original_word_count:
        evaluation_results["conciseness_score"] = 9.0  # Very concise
    elif summary_word_count < 0.4 * original_word_count:
        evaluation_results["conciseness_score"] = 7.0  # Moderately concise
    else:
        evaluation_results["conciseness_score"] = 4.0  # Less concise

    # Simulate relevance: check for keywords from the original text
    keywords = ["AI", "healthcare", "finance", "challenges", "innovation"]
    relevant_keywords_found = sum(
        1 for kw in keywords if kw.lower() in generated_summary.lower()
    )
    evaluation_results["relevance_score"] = (
        relevant_keywords_found / len(keywords)
    ) * 10.0

    # Simulate coherence: check for basic sentence structure / length (very basic mock)
    # A real LLM judge would assess flow, grammar, etc.
    if len(generated_summary.split(".")) > 1 and len(generated_summary.split(".")) < 5:
        evaluation_results["coherence_score"] = 8.0
    else:
        evaluation_results["coherence_score"] = 5.0

    evaluation_results["overall_score"] = (
        evaluation_results["conciseness_score"]
        + evaluation_results["relevance_score"]
        + evaluation_results["coherence_score"]
    ) / 3.0

    return evaluation_results


client = Groq(api_key=api_key)
for i, prompt_data in enumerate(prompts_to_test):
    if i != 0:
        continue
    prompt_name = prompt_data["name"]
    raw_prompt_template = prompt_data["text"]

    # Fill in the article text into the prompt template
    current_prompt = raw_prompt_template.format(article_text=article)

    with mlflow.start_run() as run:
        # --- Log Parameters ---
        mlflow.log_param("prompt_strategy", prompt_name)
        mlflow.log_param("raw_prompt_template", raw_prompt_template)
        mlflow.log_param("full_input_prompt", current_prompt)
        mlflow.log_param("model_name", "llama3-8b-8192")
        mlflow.log_param("temperature", 0.7)
        mlflow.log_param("max_tokens", 800)

        try:
            # --- Call Groq API ---
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": current_prompt}],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=800,
            )
            generated_text = chat_completion.choices[0].message.content
            print(f"Generated Summary ({prompt_name}):\n{generated_text}\n")

            # --- Log Generated Text as Artifact ---
            with open(
                f"generated_summary_{prompt_name.replace(' ', '_')}.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(generated_text)
            mlflow.log_artifact(
                f"generated_summary_{prompt_name.replace(' ', '_')}.txt"
            )
            # --- Evaluate with LLM Judge and Log Metrics ---
            evaluation_scores = evaluate_summary_with_mock_llm_judge(
                article, current_prompt, generated_text
            )
            for metric_name, score in evaluation_scores.items():
                mlflow.log_metric(f"judge_{metric_name}", score)
            print(f"Judge Scores: {evaluation_scores}")

            # # --- ADD EVALUATION HERE ---
            # # Evaluate the generated summary against the original article
            # evaluation_results = evaluator.evaluate_summary(
            #     generated_summary=generated_text,
            #     reference_text=article,  # Using original article as reference
            #     run_id=run.info.run_id,
            # )

            # # Print evaluation results
            # print(f"Evaluation Results for {prompt_name}:")
            # for metric_name, score in evaluation_results.items():
            #     if score is not None:
            #         print(f"  {metric_name}: {score:.4f}")

            # Optionally log evaluation results as a JSON artifact
            import json

            with open(
                f"evaluation_results_{prompt_name.replace(' ', '_')}.json", "w"
            ) as f:
                json.dump(evaluation_results, f, indent=2)
            mlflow.log_artifact(
                f"evaluation_results_{prompt_name.replace(' ', '_')}.json"
            )

        except Exception as e:
            print(f"❌ Error during generation for '{prompt_name}': {e}")
            mlflow.log_param("generation_error", str(e))
            mlflow.set_tag("status", "failed")
    print("\n✅ All MLflow runs completed for prompt optimization.")
# Use predefined question-answering metrics to evaluate our model.
# results = mlflow.evaluate(
#     logged_model_info.model_uri,
#     eval_data,
#     targets="ground_truth",
#     model_type="question-answering",
# )
# print(f"See aggregated evaluation results below: \n{results.metrics}")

# # Evaluation result for each data record is available in `results.tables`.
# eval_table = results.tables["eval_results_table"]
# print(f"See evaluation table below: \n{eval_table}")
