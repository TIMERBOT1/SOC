from hydra import compose, initialize
from omegaconf import DictConfig
from hydra.utils import instantiate
import importlib 


def make_retriever(cfg: DictConfig):
    dataset = instantiate(cfg['dataset'])
    embeddings = instantiate(cfg['embeddings'])

    class_path = cfg.class_path
    module_path, class_name = class_path.rsplit('.', 1)  # разделяет путь модуля и имя класса
    module = importlib.import_module(module_path)  # импортирует модуль
    cls = getattr(module, class_name)

    instance = cls(embeddings, dataset)

    retriever = instance.create_retriever()

    return retriever


if __name__ == "__main__":
    with initialize(config_path='config', version_base=None):
        cfg = compose(config_name='config')  # Название вашего основного YAML файла
    retriever = make_retriever(cfg)
