import os
import json
from langchain_core.documents import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


# def split_into_docs(initial_path, split_file_name):
#     with open(initial_path, 'r') as file:
#         file_text = json.load(file)
#
#     split_text = '|'.join(file_text)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=2000, separators=['|', ])
#     split_docs = splitter.create_documents([split_text])
#
#     output_path = f'SOC/src/data/interim/{split_file_name}'
#     with open(output_path, 'w') as split_file:
#         json.dump(split_docs, split_file)
#
#     return split_docs


def text_into_docs(text):
    split_text = '|'.join(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, separators=['|', ])
    split_docs = splitter.create_documents([split_text])
    return split_docs


def table_into_docs(tables, table_ids):
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    summary_tables = [
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(table_summaries)
    ]
    return summary_tables


def text_into_smaller_docs(split_docs, split_docs_ids):
    sub_text_docs = []
    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    for i, doc in enumerate(split_docs):
        _id = split_docs_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_text_docs.extend(_sub_docs)
    return sub_text_docs
