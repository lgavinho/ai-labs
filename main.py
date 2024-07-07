import streamlit as st
import time
from langchain_community.vectorstores import FAISS

from ai_generator import AIGenerator
from settings import DALLE_MODEL_VERSION, LLM_MODEL
from utils import extract_from_html_page, extract_from_pdf
from vector_db import VectorDatabase


PDF_FILE_PATH_SOURCE = "2024-MidiacodeTextRepository.pdf"
PAGE_URL_SOURCE = "https://ptbr.midiacode.com/2022/02/22/perguntas-frequentes/"
VERSION = '0.0.5'

ai = AIGenerator()

def streamed_response(question: str, my_vectorstore: FAISS):    
    response = ai.create_text_response(question, my_vectorstore)
    for word in response.split():
        yield word + " "
        time.sleep(0.5)
        
        
def create_vector_database():    
    print("Extracting text from PDF: ", PDF_FILE_PATH_SOURCE)
    text_chunks_from_pdf = extract_from_pdf(PDF_FILE_PATH_SOURCE)
    print("Extracting text from HTML: ", PAGE_URL_SOURCE)
    text_chunks_from_html = extract_from_html_page(url=PAGE_URL_SOURCE)
    text_chunks = text_chunks_from_pdf + text_chunks_from_html
    print("Creating vectorstore...")
    db = VectorDatabase()
    my_vectorstore = db.create_vectorstore(text_chunks)
    st.session_state.midiacode_vectorstore = my_vectorstore
    print("Vectorstore created.")

        

def main():
    """
    Main function to run the Midiacode Chatbot.
    """

    st.title(f"Midiacode Chatbot")
    st.write(f"Powered by Midiacode AI Labs. Version {VERSION}")
    st.caption(f"Models: {LLM_MODEL}, {DALLE_MODEL_VERSION}")

    if "midiacode_vectorstore" not in st.session_state:
        with st.spinner("Carregando base de conhecimento do Midiacode..."):
            create_vector_database()

    my_vectorstore = st.session_state.midiacode_vectorstore

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    generate_image = st.toggle("AI Image Creator", False)

    with st.chat_message("assistant"):
        st.write("How can I help you?")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Write here..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            if generate_image:                                
                image_url = ai.create_image(prompt, size="256x256")
                print("Image URL: ", image_url)
                st.image(image_url, use_column_width=True)
                st.caption(f"Cost estimate: {ai.last_price_usage:.4f} USD")
            else:
                answer = ai.create_text_response(prompt, my_vectorstore)
                response = st.markdown(answer)                
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})
                st.caption(f"Cost estimate: {ai.last_price_usage:.4f} USD")


st.set_page_config(
    layout="centered", page_title="Midiacode Chatbot", page_icon=":robot:")


if __name__ == "__main__":
    main()
