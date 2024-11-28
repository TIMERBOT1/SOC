from langchain_community.retrievers import BM25Retriever
from src.retrievers.retriever import Retriever


class BMRetriever(Retriever):

    def create_retriever(self):
        split_text = self.dataset.split_text()
        bm25_retriever = BM25Retriever.from_texts(
            split_text, metadatas=[{"source": 1}] * len(split_text)
        )
        bm25_retriever.k = 2
        return bm25_retriever

