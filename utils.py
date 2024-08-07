import requests
import re
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from langchain_text_splitters import CharacterTextSplitter


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
