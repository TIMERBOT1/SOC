from langchain_chroma import Chroma
from retrievers.retriever import Retriever
import pickle


class MVRetriever(Retriever):
    def create_retriever(self):
        split_docs, split_docs_ids, sub_text_docs = self.dataset.split_text()
        vectorstore = Chroma(
        collection_name='collection', 
        embedding_function=self.embeddings,
        persist_directory=self.persist_directory + 'vectorstore/'
        )
        vectorstore.add_documents(sub_text_docs)

        docstore = list(zip(split_docs_ids, split_docs))
        with open(f"{self.persist_directory}docstore.pkl", 'wb') as f:
            pickle.dump(docstore, f)
