#!/usr/bin/env python
# coding: utf-8

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import json
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# Load modelos (Embeddings e LLM)

embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model='gpt-3.5-turbo', max_tokens=200)

def load_data():
    base_path = os.environ.get('LAMBDA_TASK_ROOT', '.')
    pdf_link = os.path.join(base_path, "visao-estereo-rev.pdf")
    
    # Verifica se o arquivo existe antes de tentar carregar (bom para debug)
    if not os.path.exists(pdf_link):
        raise FileNotFoundError(f"O PDF não foi encontrado em: {pdf_link}")

    loader = PyPDFLoader(pdf_link, extract_images=False)
    pages = loader.load_and_split()

    # Separar em Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Diminuir o chunk ajuda na memória do Lambda
        chunk_overlap=100,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(pages)

    # Salvar no Vector DB - Chroma
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings_model
    )

    return vectordb.as_retriever(search_kwargs={"k":3})

def get_relevant_docs(question):
    retriever = load_data()
    context = retriever.invoke(question)
    return context


def ask(question, llm):
    TEMPLATE = """
    Você é um especialista em visão computacional e visão estéreo. Responda a pergunta abaixo utilizando o contexto informado.

    Contexto: {context}

    Pergunta: {question}
    """

    prompt = PromptTemplate(template=TEMPLATE, input_variables=['context', 'question'])
    sequence = RunnableSequence(prompt | llm)
    context = get_relevant_docs(question)

    response =sequence.invoke({'context':context, 'question':question})

    return response

def lambda_handler(event, context):
    # query = event.get('question')
    body = json.loads(event.get('body', {}))
    query = body.get('question')
    response = ask(query, llm).content
    return{
        "statusCode":200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "message":"Tarefa Concluída",
            "details": response
        })
    }



