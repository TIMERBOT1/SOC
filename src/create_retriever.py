import hydra
import json
from omegaconf import DictConfig
from src.retrievers.multivector_retriever import create_multivector_retriever
from src.retrievers.bm25_retriever import create_bm25_retriever
from src.retrievers.faiss_retriever import create_faiss_retriever
from src.retrievers.save_load_retrievers import save_retriever
from datetime import datetime


@hydra.main(version_base=None, config_path='config', config_name='config')
def create_retriever(cfg: DictConfig):
    embedding_model = isinstance(cfg['embeddings'])

    with open(cfg['dataset']['path'], 'r') as fr:
        text = json.load(fr)
    full_params = {'text': text} | cfg['dataset']['params']
    splited_text = cfg['dataset']['func'](full_params)

    if cfg['retriever_name'] == 'multivector':
        retriever = create_multivector_retriever(embedding_model, splited_text[2], splited_text[1], splited_text[0])
    if cfg['retriever_name'] == 'bm25':
        retriever = create_bm25_retriever(splited_text)
    if cfg['retriever_name'] == 'faiss':
        retriever = create_faiss_retriever(splited_text, embedding_model)

    save_retriever(retriever, datetime.now())
