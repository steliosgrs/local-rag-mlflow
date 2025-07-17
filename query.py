from datetime import datetime
import os
import mlflow
from models import OllamaJudgeModel
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
from mlflow.metrics.genai import answer_correctness, faithfulness, relevance
from mlflow.metrics.genai import EvaluationExample

LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
LLM_AS_JUDGE_MODEL = os.getenv("LLM_AS_JUDGE_MODEL", "gemma3")
LLM_AS_JUDGE_URI = os.getenv("LLM_AS_JUDGE_URI", "http://localhost:11434")
from mlflow.deployments import set_deployments_target

set_deployments_target("http://localhost:11434")  # Your Ollama server


# Create good and bad examples for faithfulness evaluation based on ΟΠΕΚΕΠΕ document
faithfulness_examples = [
    # Good example - High faithfulness (score 5)
    EvaluationExample(
        input="Ποιο είναι το νομικό πλαίσιο που διέπει τη λειτουργία του ΟΠΕΚΕΠΕ;",
        output="Το νομικό πλαίσιο λειτουργίας του ΟΠΕΚΕΠΕ καθορίζεται από τον βασικό Νόμο 2520/1997 για τη σύσταση του οργανισμού, ο οποίος τροποποιήθηκε με τον Νόμο 3874/2010 που τον αναδιοργάνωσε ως Νομικό Πρόσωπο Δημοσίου Δικαίου. Επίσης διέπεται από τους κανονισμούς της ΕΕ για την Κοινή Αγροτική Πολιτική, κυρίως τον Κανονισμό (ΕΕ) αριθ. 1306/2013.",
        score=5,
        justification="Η απάντηση είναι πλήρως βασισμένη στο περιεχόμενο του κειμένου και παρέχει ακριβείς πληροφορίες για τους νόμους και κανονισμούς που αναφέρονται στο έγγραφο.",
        grading_context={
            "context": "Το νομικό πλαίσιο λειτουργίας του ΟΠΕΚΕΠΕ προσδιορίζεται από πολλαπλά νομοθετικά κείμενα τόσο σε εθνικό όσο και σε ευρωπαϊκό επίπεδο. Η βασική νομοθετική πράξη παραμένει ο Νόμος 2520/1997 για τη 'Σύσταση Οργανισμού Πληρωμών και Ελέγχου Κοινοτικών Ενισχύσεων Προσανατολισμού και Εγγυήσεων και άλλες διατάξεις', ο οποίος έχει τροποποιηθεί σημαντικά με τον Νόμο 3874/2010. Ο Νόμος 3874/2010 αναδιοργάνωσε τον ΟΠΕΚΕΠΕ ως Νομικό Πρόσωπο Δημοσίου Δικαίου... Ο βασικός κανονισμός που καθορίζει τις αρχές λειτουργίας των οργανισμών πληρωμών είναι ο Κανονισμός (ΕΕ) αριθ. 1306/2013 του Ευρωπαϊκού Κοινοβουλίου και του Συμβουλίου."
        },
    ),
    # Bad example - Low faithfulness (score 1)
    EvaluationExample(
        input="Ποιες είναι οι κύριες αρμοδιότητες του ΟΠΕΚΕΠΕ;",
        output="Ο ΟΠΕΚΕΠΕ είναι υπεύθυνος για τη διαχείριση των επιδοτήσεων των κτηνοτρόφων, την παρακολούθηση των εξαγωγών γεωργικών προϊόντων και τη ρύθμιση των τιμών των αγροτικών προϊόντων στις διεθνείς αγορές. Επίσης, διαχειρίζεται τα δάνεια προς τους αγρότες μέσω της Αγροτικής Τράπεζας.",
        score=1,
        justification="Η απάντηση περιέχει εσφαλμένες ή μη υποστηριζόμενες από το κείμενο πληροφορίες. Ο ΟΠΕΚΕΠΕ δεν είναι υπεύθυνος για τη ρύθμιση τιμών ή τη διαχείριση δανείων μέσω της Αγροτικής Τράπεζας.",
        grading_context={
            "context": "Κύρια αρμοδιότητα του οργανισμού είναι η διαχείριση του Ευρωπαϊκού Γεωργικού Ταμείου Εγγυήσεων (ΕΓΤΕ) και του Ευρωπαϊκού Γεωργικού Ταμείου Αγροτικής Ανάπτυξης (ΕΓΤΑΑ)... ο ΟΠΕΚΕΠΕ είναι υπεύθυνος για την καταβολή των άμεσων ενισχύσεων προς τους αγρότες, οι οποίες περιλαμβάνουν τη βασική ενίσχυση, τις πράσινες πληρωμές, τις πληρωμές για νέους αγρότες και τις συνδεδεμένες ενισχύσεις. Παράλληλα, διαχειρίζεται τα μέτρα αγροτικής ανάπτυξης..."
        },
    ),
    # Medium faithfulness example (score 3)
    EvaluationExample(
        input="Τι είναι το Ολοκληρωμένο Σύστημα Διαχείρισης και Ελέγχου;",
        output="Το Ολοκληρωμένο Σύστημα Διαχείρισης και Ελέγχου (ΟΣΔΕ) είναι ένα σύστημα που χρησιμοποιεί ο ΟΠΕΚΕΠΕ για τη διαχείριση των ενισχύσεων. Περιλαμβάνει το Σύστημα Αναγνώρισης Αγροτεμαχίων και διάφορους μηχανισμούς ελέγχου. Επίσης, το σύστημα χρησιμοποιεί τεχνητή νοημοσύνη για την ανίχνευση απάτης.",
        score=3,
        justification="Η απάντηση περιέχει ακριβείς πληροφορίες για το ΟΣΔΕ αλλά προσθέτει μη υποστηριζόμενη πληροφορία για την τεχνητή νοημοσύνη που δεν αναφέρεται στο κείμενο.",
        grading_context={
            "context": "Ο ΟΠΕΚΕΠΕ λειτουργεί βάσει του Ολοκληρωμένου Συστήματος Διαχείρισης και Ελέγχου (ΟΣΔΕ), το οποίο αποτελεί απαίτηση της ευρωπαϊκής νομοθεσίας για όλους τους οργανισμούς πληρωμών. Το ΟΣΔΕ περιλαμβάνει μια σειρά από διαδικασίες και μηχανισμούς που στοχεύουν στη διασφάλιση της αξιοπιστίας και της αποτελεσματικότητας του συστήματος ενισχύσεων. Βασικό στοιχείο του ΟΣΔΕ είναι το Σύστημα Αναγνώρισης Αγροτεμαχίων (ΣΑΑ)..."
        },
    ),
    # Another bad example (score 2)
    EvaluationExample(
        input="Πότε καταβάλλονται οι άμεσες ενισχύσεις;",
        output="Οι άμεσες ενισχύσεις καταβάλλονται κατά κανόνα από τον Δεκέμβριο κάθε έτους. Ωστόσο, σε ειδικές περιπτώσεις μπορεί να καταβληθούν και νωρίτερα, τον Οκτώβριο, για νέους αγρότες κάτω των 30 ετών.",
        score=2,
        justification="Η πρώτη πληροφορία είναι ακριβής, αλλά η δεύτερη για τις πρόωρες πληρωμές σε νέους αγρότες κάτω των 30 δεν υποστηρίζεται από το κείμενο.",
        grading_context={
            "context": "Η καταβολή των ενισχύσεων γίνεται σύμφωνα με το χρονοδιάγραμμα που προβλέπει η ευρωπαϊκή νομοθεσία. Οι άμεσες ενισχύσεις καταβάλλονται κατά κανόνα από τον Δεκέμβριο κάθε έτους, ενώ οι πληρωμές για τα μέτρα αγροτικής ανάπτυξης ακολουθούν διαφορετικό χρονοδιάγραμμα ανάλογα με το είδος του μέτρου."
        },
    ),
]

# Configure faithfulness metric for local Ollama model
# faithfulness_metric = faithfulness(
#     model="ollama://gemma3:1b",  # Using your local Ollama Gemma model
#     examples=faithfulness_examples,
# )


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
    mlflow.langchain.autolog()
    # mlflow.autolog()
    # mlflow.enable_system_metrics_logging()
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
            # model_info = mlflow.langchain.log_model(
            #     llm,
            #     name="langchain_model",
            #     prompts=["prompts:/summarization-prompt/2"],
            # )

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
            # model_info = mlflow.langchain.log_model(
            #     chain,
            #     name="langchain_model",
            #     prompts=["prompts:/summarization-prompt/2"],
            # )

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
            # for metric_name, metric_func in text_metrics.items():
            #     try:
            #         # Calculate metric score
            #         score = _calculate_metric_score(metric_func, eval_data)
            #         print(f"{metric_name} score: {score}")
            #         results[metric_name] = score

            #     except Exception as e:
            #         # logger.error("Failed to calculate %s: %s", metric_name, str(e))
            #         results[metric_name] = None

            mlflow.log_param("response", response)

            # question_answering_metrics = {
            #     "rouge1": mlflow.metrics.rouge1(),
            #     "rouge2": mlflow.metrics.rouge2(),
            #     "rougeL": mlflow.metrics.rougeL(),
            #     "rougeLsum": mlflow.metrics.rougeLsum(),
            #     "toxicity": mlflow.metrics.toxicity(),
            #     "latency": mlflow.metrics.latency(),
            # }
            # for metric_name, metric_func in question_answering_metrics.items():
            #     try:
            #         # Calculate metric score
            #         score = _calculate_metric_score(metric_func, eval_data)
            #         print(f"{metric_name} score: {score}")
            #         results[metric_name] = score

            #     except Exception as e:
            #         # logger.error("Failed to calculate %s: %s", metric_name, str(e))
            #         results[metric_name] = None
            # judge_llm = ChatOllama(model=LLM_AS_JUDGE_MODEL)
            # model_info = mlflow.pyfunc.log_model(
            #     artifact_path="ollama_judge",
            #     python_model=judge_llm,
            # )
            eval_data = pd.DataFrame(
                {
                    "predictions": [response],
                    "targets": [reference_text],
                    "inputs": [input],  # Add the original question
                    "context": [retriever],  # Add the retrieved documents/context
                }
            )

            judge_model = OllamaJudgeModel(LLM_AS_JUDGE_MODEL)
            model_info = mlflow.pyfunc.log_model(
                name="ollama_judge",
                python_model=judge_model,
            )
            # pip_requirements=[
            #     "requests",
            #     "pandas"
            # ]
            # )

            print(f"Logged Ollama judge model: {model_info.model_uri}")
            # return model_info.model_uri
            loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
            llm_as_judge_metrics = {
                # "retrieval_precision": answer_correctness(),
                # "answer_correctness": answer_correctness(
                #     model=LLM_AS_JUDGE_MODEL,
                #     proxy_url=LLM_AS_JUDGE_URI,
                # ),
                # "faithfulness": faithfulness(
                #     model=LLM_AS_JUDGE_MODEL,
                #     proxy_url=LLM_AS_JUDGE_URI,
                # ),
                # "response_relevancy": relevance(
                #     model=LLM_AS_JUDGE_MODEL,
                #     proxy_url=LLM_AS_JUDGE_URI,
                # ),
                "faithfulness": faithfulness(
                    # model=loaded_model,  # Using your local Ollama Gemma model
                    # model=f"ollama://{LLM_AS_JUDGE_MODEL}",  # Using your local Ollama Gemma model
                    # model=model_info.model_uri,  # Using your local Ollama Gemma model
                    model=f"endpoints://{model_info.model_uri}",  # Using your local Ollama Gemma model
                    examples=faithfulness_examples,
                    # proxy_url=LLM_AS_JUDGE_URI,
                )
            }
            for metric_name, metric_func in llm_as_judge_metrics.items():
                try:
                    # Calculate metric score
                    score = _calculate_metric_score(
                        metric_func,
                        eval_data,
                        model_uri=model_info.model_uri,
                    )
                    print(f"{metric_name} score: {score}")
                    results[metric_name] = score

                except Exception as e:
                    print(f"Failed to calculate {metric_name}: {str(e)}")
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


def _calculate_metric_score(
    metric_func, eval_data: pd.DataFrame, model_uri=None
) -> float:
    """
    Calculate the score for a specific metric.

    Args:
        metric_func: MLflow metric function
        eval_data (pd.DataFrame): Evaluation data with predictions and targets

    Returns:
        float: Calculated metric score
    """
    # eval_results = None
    # if model:
    #     eval_results = mlflow.evaluate(
    #         model=model,
    #         data=eval_data,
    #         targets="targets",
    #         predictions="predictions",
    #         extra_metrics=[metric_func],
    #         evaluator_config={
    #             "col_mapping": {"predictions": "predictions", "targets": "targets"}
    #         },
    #     )
    # else:
    #     # Use MLflow's evaluate function to calculate the metric
    #     eval_results = mlflow.evaluate(
    #         data=eval_data,
    #         targets="targets",
    #         predictions="predictions",
    #         extra_metrics=[metric_func],
    #         evaluator_config={
    #             "col_mapping": {"predictions": "predictions", "targets": "targets"}
    #         },
    #     )
    eval_results = mlflow.evaluate(
        model_uri,
        data=eval_data,
        targets="targets",
        predictions="predictions",
        extra_metrics=[metric_func],
        evaluator_config={
            "col_mapping": {
                "predictions": "predictions",
                "targets": "targets",
                "inputs": "inputs",
                "context": "context",
            }
        },
    )
    # print(f"eval_results: {eval_results}")
    print(
        f"See per-data evaluation results below: \n{eval_results.tables['eval_results_table']}"
    )
    mlflow.log_table(
        eval_results.tables["eval_results_table"], "evaluation_results.json"
    )
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
