from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from retrievers.retriever import Retriever


class MVRetriever(Retriever):

    def create_empty_retriever(self, vectorstore):
        store = InMemoryByteStore()
        id_key = "doc_id"
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
            search_kwargs={"k": 1}
        )
        return retriever

    def create_retriever(self):
        # text_list = [x.page_content for x in self.dataset.split_text()[2]]
        # print(type(self.embeddings.embed_documents(text_list)))
        split_docs, split_docs_ids, sub_text_docs = self.dataset.split_text()
        vectorstore = Chroma(
        collection_name='collection', embedding_function=self.embeddings
        )
        retriever = self.create_empty_retriever(vectorstore)

        retriever.vectorstore.add_documents(sub_text_docs)
        retriever.docstore.mset(list(zip(split_docs_ids, split_docs)))
        return retriever
