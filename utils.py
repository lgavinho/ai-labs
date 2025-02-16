import streamlit as st
import requests
import re
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from langchain_text_splitters import CharacterTextSplitter
from streamlit.logger import get_logger
import settings
import tempfile
import os

logger = get_logger(__name__)


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
    text_chunks = []
    with open(filepath, 'rb') as f:
        reader = PdfReader(f)
        logger.info("Number of pages: %d", len(reader.pages))
        for i, page in enumerate(reader.pages):
            logger.info("Extracting page %d...", i+1)
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
            logger.error("Failed to retrieve HTML: Status Code %d", response.status_code)
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching URL: %s", e)
    return None


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
            logger.info("Image downloaded successfully and saved at: %s", save_path)
        else:
            logger.error("Failed to download image. Status code: %d", response.status_code)
    except Exception as e:
        logger.error("An error occurred: %s", e)

def add_sidebar():
    with st.sidebar:    
        st.header("Sobre")
        st.write(f"Versão {settings.VERSION}")    
        st.caption(f"Modelos: {settings.LLM_MODEL}, {settings.DALLE_MODEL_VERSION}, {settings.EMBEDDING_MODEL_VERSION}")
        st.caption(f":moneybag: Custo da sessão: {st.session_state.total_cost:.6f} USD")
        st.write("© Midiacode Lda")

def download_pdf(url):
    """
    Downloads a PDF file from URL and saves it to a temporary location.
    
    Args:
        url (str): The URL of the PDF file to download
        
    Returns:
        str: Full path to the saved PDF file
    """
    try:
        # Send GET request to download the PDF
        response = requests.get(url)
        
        # Check if request was successful
        if response.status_code == 200:
            # Extract filename from URL
            filename = url.split('/')[-1]
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'
                
            # Create full path using temporary directory
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, filename)
            
            # Save the PDF file
            with open(file_path, 'wb') as f:
                f.write(response.content)
                
            logger.info("PDF downloaded successfully and saved at: %s", file_path)
            return file_path
        else:
            logger.error("Failed to download PDF. Status code: %d", response.status_code)
            return None
            
    except Exception as e:
        logger.error("An error occurred while downloading PDF: %s", e)
        return None
