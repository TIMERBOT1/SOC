from retrievers.save_load_retrievers import save_retriever, load_retriever
from langchain.retrievers import EnsembleRetriever
import yaml


def essemble_retrievers():
    first_retriever = load_retriever('data/interim/first_retriever.pkl')
    second_retriever = load_retriever('data/interim/second_retriever.pkl')

    params = yaml.safe_load(open("params.yaml"))["essemble_retrievers"]
    weight1 = params['weight1']
    weight2 = params['weight2']

    retriever = EnsembleRetriever(
        retrievers=[first_retriever, second_retriever], weights=[weight1, weight2]
        )
    save_retriever(retriever, 'data/processed/retriever.pkl')


if __name__ == "__main__":
   essemble_retrievers()

      