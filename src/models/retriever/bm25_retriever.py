from langchain.retrievers import BM25Retriever, EnsembleRetriever
import pickle


class BMRetriever:
    def __init__(self):
        pass

    @staticmethod
    def create_retriever(split_text):
        bm25_retriever = BM25Retriever.from_texts(
            split_text, metadatas=[{"source": 1}] * len(split_text)
        )
        bm25_retriever.k = 2
        return bm25_retriever

    @staticmethod
    def save_retriever(path, retriever):
        with open(path, "wb") as f:
            pickle.dump(retriever, f)

    @staticmethod
    def load_retriever(path):
        with open(path, "rb") as f:
            retriever = pickle.load(f)
        return retriever
