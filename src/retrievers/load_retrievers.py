from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
import pickle


def load_retriever(filename, config_name):
    with initialize_config_dir(version_base=None, config_dir="/home/tirgashev/SOC/src/config"):
        cfg = compose(config_name=config_name)
    vectorstore = Chroma(
        persist_directory=f"{filename}vectorstore.pkl",
        embedding_function=instantiate(cfg['embeddings']),
        )
    
    store = InMemoryByteStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs={"k": 1}
    )

    with open(f"{filename}docstore.pkl", "rb") as input_file:
        docstore = pickle.load(input_file)
    retriever.docstore.mset(docstore)
    return retriever
