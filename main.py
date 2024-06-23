import streamlit as st
import re
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
import time
import random

from prompt_template import get_prompt

# Arquivo PDF com o conteúdo do Midiacode
file_path = "2024-MidiacodeTextRepository.pdf"
VERSION = '0.0.3'


def split_paragraphs(rawText):
    """
    Splits the raw text into paragraphs.

    Args:
        rawText (str): The raw text to be split.

    Returns:
        list: A list of paragraphs.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(rawText)


def clean_text(text):
    # Remove unnecessary line breaks and join words into proper sentences
    cleaned_text = re.sub(r'\s*\n\s*', ' ', text)
    return cleaned_text


def extract_from_pdf(filepath: str):
    """
    Extracts text from a PDF file.

    Args:
        filepath (str): The path to the PDF file.

    Returns:
        list: A list of text chunks extracted from the PDF.
    """
    text_chunks = []
    with open(filepath, 'rb') as f:
        reader = PdfReader(f)
        print("Number of pages: ", len(reader.pages))
        for i, page in enumerate(reader.pages):
            print(f"Extracting page {i+1}...")
            raw = page.extract_text()
            cleaned = clean_text(raw)
            chunks = split_paragraphs(cleaned)
            text_chunks += chunks
    return text_chunks


def create_vectorstore(text_chunks: str):
    """
    Creates a vector store from the given text chunks.

    Args:
        text_chunks (list): A list of text chunks.

    Returns:
        FAISS: The created vector store.
    """
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_texts(text_chunks, embeddings)
    return store


def retrieve_context(query: str, db: FAISS):
    """
    Retrieves context from the vector store based on the given query.

    Args:
        query (str): The query string.
        db (FAISS): The vector store.

    Returns:
        list: A list of similar responses.
    """
    similar_response = db.similarity_search(query, k=30)
    return similar_response


def generate_response(question: str, my_vectorstore: FAISS):
    """
    Generates a response to the given question.

    Args:
        question (str): The question string.

    Returns:
        str: The generated response.
    """

    print("Creating LLM chain...")
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    prompt = get_prompt()
    chain = prompt | llm

    print("Retrieving context for question: ", question)
    custom_content = retrieve_context(question, my_vectorstore)

    print("Invoking chain...")
    inputs = {
        "question": question,
        "custom_content": custom_content
    }
    response = chain.invoke(inputs)

    print("---- Response:")
    print(response.content)
    return response.content


def streamed_response(question: str, my_vectorstore: FAISS):
    response = generate_response(question, my_vectorstore)
    if random.choice(['yes', 'no']) == 'yes':
        footer_message = "Se preferir, pode acessar nosso site [midiacode.com](https://midiacode.com/) e também solicitar um chat com nossa equipe."
        response += "\n\n" + footer_message
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def main():
    """
    Main function to run the Midiacode Chatbot.
    """

    st.title(f"Midiacode Chatbot")
    st.write(f"Powered by Midiacode AI Labs. Versão {VERSION}")

    if "my_vectorstore" not in st.session_state:
        with st.spinner("Carregando PDF..."):
            print("Extracting text from PDF: ", file_path)
            text_chunks = extract_from_pdf(file_path)
            print("Creating vectorstore...")
            my_vectorstore = create_vectorstore(text_chunks)
            st.session_state.my_vectorstore = my_vectorstore
            print("Vectorstore created.")

    my_vectorstore = st.session_state.my_vectorstore

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.chat_message("assistant"):
        st.write("Como posso te ajudar?")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Escreva sua pergunta aqui..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            answer = streamed_response(prompt, my_vectorstore)
            response = st.write_stream(answer)
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response})


st.set_page_config(
    layout="centered", page_title="Midiacode Chatbot", page_icon=":robot:")


if __name__ == "__main__":
    main()
