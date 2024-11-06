from langchain_core.prompts import ChatPromptTemplate


def built_rag_chain():
    template = """Answer the question based only on the following context, which can include text and tables, there is a table in LaTeX format and a table caption in plain text format or text is a string
    :
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            {"context": ensemble_retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    return chain
