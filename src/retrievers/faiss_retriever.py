from langchain_community.vectorstores import FAISS
from src.retrievers.retriever import Retriever


class FaissRetriever(Retriever):

    def create_retriever(self):
        split_text = self.dataset.split_text()
        faiss_vectorstore = FAISS.from_texts(
            split_text, self.embeddings, metadatas=[{"source": 2}] * len(split_text)
        )
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
        return faiss_retriever

