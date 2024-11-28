from hydra.utils import instantiate
from omegaconf import OmegaConf


class Retriever:
    
    def __init__(self, embeddings, dataset):
        print('init')
        self.embeddings = embeddings
        self.dataset = dataset

    def create_retriever(self):
        pass