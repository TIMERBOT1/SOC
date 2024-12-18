import cmd
from src.make_retriever import make_retriever
from hydra import compose, initialize
from langchain.retrievers import EnsembleRetriever
from src.models.rag_chain import built_rag_chain
from langchain_core.load import dumpd


class ConsoleApp(cmd.Cmd):
    intro = 'Консольное приложение для создания ретриверов и rag-цепочек. Введите help для списка команд.'

    def __init__(self):
        super().__init__()
        self.retriever = None

    def do_get_ret(self, path):
        config_path, config_name = path.rsplit('.', 1)
        with initialize(config_path=config_path, version_base=None):
            cfg = compose(config_name=config_name)
        
        retriever = make_retriever(cfg)
        self.retriever = make_retriever(cfg)
        return retriever

    def do_ensemble(self, path1, path2, weight1 = 0.5, weight2 = 0.5):
        retriever1 = self.do_get_ret(path1)
        retriever2 = self.do_get_ret(path2)
        self.retriever = EnsembleRetriever(
            retrievers=[retriever1, retriever2], weights=[weight1, weight2]
)

    def do_chain(self, template, model):
        chain = built_rag_chain(template, self.retriever, model)
        
        string_representation = dumpd(chain, pretty=True)

        with open("chain.json", "w") as file:
            file.write(string_representation)

    def do_exit(self, arg):
        print('Выход из приложения...')
        return True        

    def do_help(self, arg):
        print("Доступные команды:")
        print("  get_ret <path>   Создать ретривер.")
        print("  ensemble <path1> <path2> <weight1=0.5> <weight2=0.5>  Создать 2 ретривера и объединить их в один")
        print("  do_chain <template> <model>   Создать rag-цепочку с использованием последнего ретриера и сохранить ее.")
        print("  exit           Выход из приложения.")

if __name__ == '__main__':
    ConsoleApp().cmdloop()