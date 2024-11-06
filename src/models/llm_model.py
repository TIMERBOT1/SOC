from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

import subprocess
import uuid
import json
import getpass
import os
import re
import warnings
import torch


class LLMmodel:
    model = ChatOllama(
        model="llama3.1",
        temperature=0,
    )
    model_name = "deepvk/USER-bge-m3"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True, "batch_size": 4}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, show_progress=True
    )