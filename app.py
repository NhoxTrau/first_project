
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from langchain.globals import set_llm_cache, get_llm_cache
from langchain.schema import HumanMessage

# some code using set_llm_cache and get_llm_cache


def init():
    load_dotenv()

@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text.encode('utf-8').decode('utf-8')

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,  # Giảm kích thước chunk
        chunk_overlap=50,  # Giảm overlap
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False,legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
    )
    return conversation_chain


def handle_user_input(user_question):
    try:
        response = st.session_state.conversation({"question": user_question})
        
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {str(e)}")


def main():
    init()
    st.set_page_config(page_title="Chat với tôi", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat với tôi :books:")
    user_question = st.text_input("Hỏi một câu về tài liệu của bạn:")

    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}", "Xin chào, robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Xin chào! Tôi có thể giúp gì cho bạn?"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Tài liệu của bạn")
        pdf_docs = st.file_uploader("Tải lên các file PDF và nhấn 'Xử lý'", accept_multiple_files=True)
        if st.button("Xử lý"):
            with st.spinner("Đang xử lý..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Xử lý hoàn tất!")

if __name__ == '__main__':
    main()