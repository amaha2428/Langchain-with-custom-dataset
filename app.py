from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import streamlit as st

# load_dotenv()

# api_key = os.getenv('GOOGLE_API_KEY')
api_key = st.secrets['GOOGLE_API_KEY']

llm = GooglePalm(google_api_key=api_key, temperature=0)

embedding = HuggingFaceEmbeddings()

vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='FUPRE_DATA.csv', source_column='prompt', encoding='iso-8859-1')
    data = loader.load()

    vector_db = FAISS.from_documents(documents=data, embedding=embedding)
    vector_db.save_local(vectordb_file_path)

def qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embedding)

    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
        chain_type="stuff",
        retriever= retriever,
        input_key= "query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain



if __name__ == '__main__':
    chain = qa_chain()
    print(chain('WHo is the HOD of computer science'))
    # pass
  

