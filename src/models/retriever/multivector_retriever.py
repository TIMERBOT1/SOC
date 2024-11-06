import uuid
import faiss
import pickle
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from src.features.splliter import text_into_smaller_docs, table_into_docs


class MLRetriever:
    def __init__(self):
        pass

    @staticmethod
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

    @staticmethod
    def create_retriever(split_text, embedding_function, splitter, collection):
        vectorstore = Chroma(
            collection_name=collection, embedding_function=embedding_function
        )
        retriever = MLRetriever.init_retriever(vectorstore)

        split_text_ids = [str(uuid.uuid4()) for _ in split_text]
        sub_text_docs = splitter(split_text, split_text_ids)

        retriever.vectorstore.add_documents(sub_text_docs)
        retriever.docstore.mset(list(zip(split_text_ids, text_docs)))
        return retriever

    @staticmethod
    def create_retriever_from_text(split_text, embedding_function, collection="documents"):
        return MLRetriever.create_retriever(split_text,
                                            embedding_function,
                                            text_into_smaller_docs,
                                            collection)

    @staticmethod
    def create_retriever_from_table(split_text, embedding_function, collection="documents"):
        return MLRetriever.create_retriever(split_text,
                                            embedding_function,
                                            table_into_docs,
                                            collection)

    @staticmethod
    def save_retriever(file_name, path, retriever):
        retriever_path = f'{path}/{file_name}.pkl'
        with open(retriever_path, 'wb') as f:
            pickle.dump(retriever, f)

        doc_store_path = f'{path}/{file_name}_docstore.pkl'
        with open(doc_store_path, 'wb') as f:
            pickle.dump(retriever.docstore, f)

        index_path = f'{path}/{file_name}_index.faith'
        faiss.write_index(retriever.index, index_path)

    @staticmethod
    def load_retriever(file_name, path):
        retriever_path = f'{path}/{file_name}.pkl'
        with open(retriever_path, 'rb') as f:
            retriever = pickle.load(f)

        doc_store_path = f'{path}/{file_name}_docstore.pkl'
        with open(doc_store_path, 'rb') as f:
            retriever.docstore = pickle.load(f)

        index_path = f'{path}/{file_name}_index.faith'
        retriever.index = faiss.read_index(index_path)

        return retriever

    @staticmethod
    def ensemble_retriever(retriever, table_retriever):
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, table_retriever], weights=[0.5, 0.5]
        )
        return ensemble_retriever
