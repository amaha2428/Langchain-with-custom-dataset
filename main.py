import streamlit as st
from app import qa_chain

st.title("LangChain Test 🌱")

question = st.text_input("Question: ")

if question:
    chain = qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])