from langchain_community.vectorstores import FAISS


def create_faiss_retriever(split_text, embeddings):
    faiss_vectorstore = FAISS.from_texts(
        split_text, embeddings, metadatas=[{"source": 2}] * len(split_text)
    )
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
    return faiss_retriever
