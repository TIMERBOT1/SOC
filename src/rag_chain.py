from langchain_core.runnables import RunnablePassthrough
from retrievers.load_retrievers import load_retriever
from langchain.retrievers import EnsembleRetriever
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import yaml


def built_rag_chain():
    params = yaml.safe_load(open("params.yaml"))["ensemble_retrievers"]
    first_retriever = load_retriever('data/interim/first_retriever_', 'ret1_config')
    second_retriever = load_retriever('data/interim/second_retriever_', 'ret2_config')
    essembled_retriever = EnsembleRetriever(
        retrievers=[first_retriever, second_retriever], weights=params['weights']
        )
    
    params = yaml.safe_load(open("params.yaml"))["rag_chain_setup"]
    template = params['template']
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOllama(
        model=params['model'],
        temperature=0,
    )

    chain = (
        {"context": essembled_retriever, 
            "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain



