import streamlit as st
from app import qa_chain

st.title("LangChain Test 🌱")
st.write('Ensure to add question mark (?) at the end oa every question')
question = st.text_input("Question: ")

if question:
    chain = qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])