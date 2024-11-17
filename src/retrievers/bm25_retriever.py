from langchain.retrievers import BM25Retriever


def create_bm25_retriever(split_text):
    bm25_retriever = BM25Retriever.from_texts(
        split_text, metadatas=[{"source": 1}] * len(split_text)
    )
    bm25_retriever.k = 2
    return bm25_retriever
