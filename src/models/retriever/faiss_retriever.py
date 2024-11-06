from langchain_community.vectorstores import FAISS
from src.models.constants import MODEL_NAME


class FaissRetriever:
    def __init__(self):
        pass

    @staticmethod
    def create_retriever(split_text, embeddings):
        faiss_vectorstore = FAISS.from_texts(
            split_text, embeddings, metadatas=[{"source": 2}] * len(texts)
        )
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
        return faiss_retriever

    @staticmethod
    def save_retriever(path, retriever):
        retriever.save(path)

    @staticmethod
    def load_retriever(path):
        retriever = SentenceTransformer(MODEL_NAME)
        retriever.load(path)
        return retriever
