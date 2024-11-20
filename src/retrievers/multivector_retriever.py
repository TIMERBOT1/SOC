from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore


def init_retriever(vectorstore):
    store = InMemoryByteStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs={"k": 1}
    )
    return retriever


def create_multivector_retriever(embeddings, sub_text_docs, split_text_ids, split_text):
    vectorstore = Chroma(
        collection_name='collection', embedding_function=embeddings
    )
    retriever = init_retriever(vectorstore)

    retriever.vectorstore.add_documents(sub_text_docs)
    retriever.docstore.mset(list(zip(split_text_ids, split_text)))
    return retriever

