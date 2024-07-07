from typing import Tuple
import requests
from openai import OpenAI
import streamlit as st
import re
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
import time
import random

from prompt_template import get_prompt


PDF_FILE_PATH_SOURCE = "2024-MidiacodeTextRepository.pdf"
PAGE_URL_SOURCE = "https://ptbr.midiacode.com/2022/02/22/perguntas-frequentes/"
LLM_MODEL = "gpt-3.5-turbo-0125"
VERSION = '0.0.5'

DALLE_MODEL_VERSION = "dall-e-2"
OPEN_AI_DALLE_PRICE_PER_IMAGE_256X256 = 0.016

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


def extract_from_html_page(url: str):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            texts = soup.get_text(separator='\n')
            cleaned = clean_text(texts)
            chunks = split_paragraphs(cleaned)
            return chunks
        else:
            print(
                f"Failed to retrieve HTML: Status Code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
    return None


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
    similar_response = db.similarity_search(query, k=20)
    return similar_response


def create_text_response(question: str, my_vectorstore: FAISS) -> str:
    """
    Generates a response to the given question.

    Args:
        question (str): The question string.

    Returns:
        str: The generated response.
    """

    print("Creating LLM chain...")
    llm = ChatOpenAI(temperature=0, model=LLM_MODEL)
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
    
    answer = response.content
    if random.choice(['yes', 'no']) == 'yes':
        footer_message = "Se preferir, pode acessar nosso site [midiacode.com](https://midiacode.com/) e também solicitar um chat com nossa equipe."
        answer += "\n\n" + footer_message

    print(answer)
    
    # getting usage of tokens
    total_tokens = 0
    response_metadata = response.response_metadata
    if response_metadata:
        token_usage = response_metadata.get('token_usage')
        print("Tokens usage: ", token_usage)
        if token_usage:
            total_tokens = token_usage.get('total_tokens')
    
    return answer


def streamed_response(question: str, my_vectorstore: FAISS):
    response, total_tokens = create_text_response(question, my_vectorstore)
    for word in response.split():
        yield word + " "
        time.sleep(0.5)


def create_image(prompt: str, size="1024x1792", quality="standard"):
    print("Generating image...")
    client = OpenAI()

    response = client.images.generate(
        model=DALLE_MODEL_VERSION,
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
    )
    print(response)
    image_url = response.data[0].url
    return image_url


def download_image(url, save_path):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open a file for writing in binary mode
            with open(save_path, 'wb') as f:
                # Write the contents of the response (image content) to the file
                f.write(response.content)
            print(f"Image downloaded successfully and saved at: {save_path}")
        else:
            print(
                f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    """
    Main function to run the Midiacode Chatbot.
    """

    st.title(f"Midiacode Chatbot")
    st.write(f"Powered by Midiacode AI Labs. Version {VERSION}")

    if "my_vectorstore" not in st.session_state:
        with st.spinner("Carregando base de conhecimento..."):
            print("Extracting text from PDF: ", PDF_FILE_PATH_SOURCE)
            text_chunks_from_pdf = extract_from_pdf(PDF_FILE_PATH_SOURCE)
            print("Extracting text from HTML: ", PAGE_URL_SOURCE)
            text_chunks_from_html = extract_from_html_page(url=PAGE_URL_SOURCE)
            text_chunks = text_chunks_from_pdf + text_chunks_from_html
            print("Creating vectorstore...")
            my_vectorstore = create_vectorstore(text_chunks)
            st.session_state.my_vectorstore = my_vectorstore
            print("Vectorstore created.")

    my_vectorstore = st.session_state.my_vectorstore

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
                image_url = create_image(prompt, size="256x256")
                print("Image URL: ", image_url)
                st.image(image_url, use_column_width=True)
            else:
                answer = create_text_response(prompt, my_vectorstore)
                response = st.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})


st.set_page_config(
    layout="centered", page_title="Midiacode Chatbot", page_icon=":robot:")


if __name__ == "__main__":
    main()
