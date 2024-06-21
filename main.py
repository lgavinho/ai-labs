import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

from prompt_template import get_prompt

# Arquivo PDF com o conteúdo do Midiacode
file_path = "2024-MidiacodeTextRepository.pdf"
VERSION = '0.0.2'


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
        for page in reader.pages:
            raw = page.extract_text()
            chunks = split_paragraphs(raw)
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


def generate_response(question: str):
    """
    Generates a response to the given question.

    Args:
        question (str): The question string.

    Returns:
        str: The generated response.
    """
    print("Extracting text from PDF: ", file_path)
    text_chunks = extract_from_pdf(file_path)

    print("Creating vectorstore...")
    my_vectorstore = create_vectorstore(text_chunks)

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


def main():
    """
    Main function to run the Midiacode Chatbot.
    """
    st.title(f"Midiacode Chatbot v{VERSION}")
    st.header("Como posso te ajudar?")
    question = st.text_area("Pergunta")
    clicked = st.button("Enviar")

    if question and clicked:
        st.write("Gerando resposta...")
        result = generate_response(question)
        st.info(result)


st.set_page_config(
    layout="centered", page_title="Midiacode Chatbot", page_icon=":robot:")


st.markdown(
    """
    <style>
    /* Estilizando área de texto */
    .stTextArea textarea {
        color: black; /* Define a cor do texto para azul */
        /* Outros estilos de CSS opcionais, como font-size, font-family, etc. */
    }
    </style>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()
