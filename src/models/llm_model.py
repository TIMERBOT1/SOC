from langchain_ollama import ChatOllama
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from src.models.constants import MODEL_NAME
# import torch


def get_embeddings():
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True, "batch_size": 4}
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, show_progress=True
    )
    return embeddings
