from langchain_core.vector_store import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PDFPlumberloader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdfs_directory ='./pdfs/'
embeddings = OllamaEmbeddings(model='deepseek-r1:14b')
vector_store= InMemoryVectorStore(embeddings)

model = OllamaLLM(model='deepseek-r1:14b')

import streamlit as st

template = """ 
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file):
    loader = PDFPlumberloader(file_path)
    documents = loader.load()
    return documents

def split_text(file):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index= True
    )

    return text_splitter.split_documents(documents)

def index_docs(file):
    vector_store.add_documents(documents)

def retreive_docs(file):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question":question, "context":context})

uploaded_file= st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_documents = retreive_docs(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)
