import requests
import ollama

from typing import Any, Dict, List, Optional, ClassVar

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.embeddings import Embeddings


class CustomLLM(LLM):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        response = self._get_response_from_server(prompt)
        return response

    @staticmethod
    def _get_response_from_server(prompt):
        api_url = "http://10.54.11.120:11436/api/generate"
        data = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(api_url, json=data)
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise requests.exceptions.HTTPError(f"{response.status_code}, {response.text}")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"


class CustomOllamaEmbeddings(Embeddings):
    def __init__(self, api_url, embedding_model):
        self.api_url = api_url
        self.embedding_model = embedding_model
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.get_response_from_server(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.get_response_from_server(text)[0]

    def get_response_from_server(self, inp: str | list[str]):
        data = {
            "model": self.embedding_model,
            "input": inp,
        }

        response = requests.post(self.api_url, json=data)
        if response.status_code == 200:
            return response.json()['embeddings']
        else:
            raise requests.exceptions.HTTPError(f"{response.status_code}, {response.text}")
        
