import uuid
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)


def text_into_docs(text, chunk_size):
    split_text = '|'.join(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, separators=['|', ])
    split_docs = splitter.create_documents([split_text])
    return split_docs


def table_into_docs(text, model_name, prompt_text, max_concurrency):
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOllama(
        model=model_name,
        temperature=0,
    )
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    table_ids = [str(uuid.uuid4()) for _ in text]
    table_summaries = summarize_chain.batch(text, {"max_concurrency": max_concurrency})
    summary_tables = [
        Document(page_content=s, metadata={'id_key': table_ids[i]})
        for i, s in enumerate(table_summaries)
    ]
    return (text, table_ids, summary_tables)


def text_into_smaller_docs(text, chunk_size, child_chunk_size):
    sub_text_docs = []
    split_docs = text_into_docs(text, chunk_size)
    split_docs_ids = [str(uuid.uuid4()) for _ in split_docs]
    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size)

    for i, doc in enumerate(split_docs):
        _id = split_docs_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata['id_key'] = _id
        sub_text_docs.extend(_sub_docs)
    return (split_docs, split_docs_ids, sub_text_docs)
