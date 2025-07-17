import mlflow
from mlflow.pyfunc import PythonModel


class OllamaJudgeModel(PythonModel):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def load_context(self, context):
        from langchain_community.chat_models import ChatOllama

        self.model = ChatOllama(model=self.model_name)

    def predict(self, context, model_input):
        if isinstance(model_input, str):
            return self.model.invoke(model_input)
        elif hasattr(model_input, "iloc"):  # DataFrame
            return [self.model.invoke(row) for row in model_input.iloc[:, 0]]
        return self.model.invoke(model_input)
