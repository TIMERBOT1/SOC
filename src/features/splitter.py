import uuid
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)


class TextIntoDocs:

    def __init__(self, path, chunk_size):
        with open(path, "r") as text:
            self.text = text.read()
        self.chunk_size = chunk_size

    def split_text(self):
        split_text = '|'.join(self.text)
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, separators=['|', ])
        split_docs = splitter.create_documents([split_text])
        return split_docs


class TableIntoDocs:

    def __init__(self, path, model_name, prompt_text, max_concurrency):
        with open(path, "r") as text:
            self.text = text.read()
        self.model_name = model_name
        self.prompt_text = prompt_text
        self.max_concurrency = max_concurrency


    def split_text(self):
        prompt = ChatPromptTemplate.from_template(self.prompt_text)
        model = ChatOllama(
            model=self.model_name,
            temperature=0,
        )
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        table_ids = [str(uuid.uuid4()) for _ in self.text]
        table_summaries = summarize_chain.batch(self.text, {"max_concurrency": self.max_concurrency})
        summary_tables = [
            Document(page_content=s, metadata={'id_key': table_ids[i]})
            for i, s in enumerate(table_summaries)
        ]
        return (self.text, table_ids, summary_tables)

class TextIntoSmallerDocs:

    def __init__(self, path, chunk_size, child_chunk_size):
        with open(path, "r") as text:
            self.text = text.read()
        self.chunk_size = chunk_size
        self.child_chunk_size = child_chunk_size
        self.splitter = TextIntoDocs(path, chunk_size)

    def split_text(self):
        sub_text_docs = []
        split_docs = self.splitter.split_text()

        split_docs_ids = [str(uuid.uuid4()) for _ in split_docs]
        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.child_chunk_size)

        for i, doc in enumerate(split_docs):
            _id = split_docs_ids[i]
            _sub_docs = child_text_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata['id_key'] = _id
            sub_text_docs.extend(_sub_docs)
        return split_docs, split_docs_ids, sub_text_docs
