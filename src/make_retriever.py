from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import importlib 
from retrievers.save_load_retrievers import save_retriever


@hydra.main(version_base=None, config_path="config")
def make_retriever(cfg: DictConfig):
    dataset = instantiate(cfg['dataset'])
    embeddings = instantiate(cfg['embeddings'])

    class_path = cfg.class_path
    module_path, class_name = class_path.rsplit('.', 1)  # разделяет путь модуля и имя класса
    module = importlib.import_module(module_path)  # импортирует модуль
    cls = getattr(module, class_name)

    instance = cls(embeddings, dataset)
    retriever = instance.create_retriever()

    save_retriever(retriever, cfg.file_name)


if __name__ == "__main__":
    make_retriever()
