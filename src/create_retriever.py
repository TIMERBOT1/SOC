import hydra
import json
from omegaconf import DictConfig
from src.retrievers.multivector_retriever import create_multivector_retriever
from src.retrievers.bm25_retriever import create_bm25_retriever
from src.retrievers.faiss_retriever import create_faiss_retriever
from src.retrievers.save_load_retrievers import save_retriever
from hydra.utils import instantiate
from datetime import datetime
import importlib


@hydra.main(version_base=None, config_path='config', config_name='config')
def create_retriever(cfg: DictConfig):
    embedding_model = instantiate(cfg['embeddings'])

    with open(cfg['dataset']['path'], 'r') as fr:
        text = json.load(fr)
    params = dict((k, v) for d in cfg['dataset']['params'] for k, v in d.items())
    full_params = {'text': text} | params
    module_name, function_name = cfg['dataset']['func'].rsplit('.', 1)
    module = importlib.import_module(module_name)
    func = getattr(module, function_name)
    splited_text = func(**full_params)

    if cfg['retriever_name'] == 'multivector':
        retriever = create_multivector_retriever(embedding_model, splited_text[2], splited_text[1], splited_text[0])
    if cfg['retriever_name'] == 'bm25':
        retriever = create_bm25_retriever([x.page_content for x in splited_text])
    if cfg['retriever_name'] == 'faiss':
        retriever = create_faiss_retriever([x.page_content for x in splited_text], embedding_model)

    time = datetime.now().strftime("%Y.%m.%d._%H.%M.%S")
    save_retriever(retriever, f'data/processed/retrievers/{cfg["retriever_name"]}_{time}.pkl')


if __name__ == "__main__":
    create_retriever() 