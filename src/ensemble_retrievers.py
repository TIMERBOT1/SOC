from src.retrievers.load_retrievers import save_retriever, load_retriever
from langchain.retrievers import EnsembleRetriever
import yaml


def essemble_retrievers(first_retriever, second_retriever, weights):
    retriever = EnsembleRetriever(
        retrievers=[first_retriever, second_retriever], weights=weights
        )
    
    return retriever

      