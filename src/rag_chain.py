from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from retrievers.save_load_retrievers import load_retriever
import pickle
import yaml


def built_rag_chain():
    params = yaml.safe_load(open("params.yaml"))["buitl_rag_chain"]
    template = params['template']
    prompt = ChatPromptTemplate.from_template(template)
    model = params['model']
    retriever = load_retriever('data/processed/retriever.pkl')

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    with open('data/processed/rag_chain.pkl', 'wb') as file:
            pickle.dump(chain, file)


if __name__ == "__main__":
   built_rag_chain()