import json
import os
import sqlite3
import numpy as np
import faiss
from requests_aws4auth import AWS4Auth
from opensearchpy.helpers import bulk
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import tiktoken
import settings
from utils import extract_from_html_page, extract_from_pdf

class VectorDatabase:
    
    price_usage = 0
    total_tokens = 0
    
    def __init__(self, model_name=''):
        if model_name == '':
            model_name = settings.EMBEDDING_MODEL_VERSION
            
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name) 
        self.embeddings = OpenAIEmbeddings()        
                
    def calculate_tokens(self, text: str) -> int:
        """
        Calculate the number of tokens for the given text using the tokenizer.

        Args:
            text (str): The text to tokenize.

        Returns:
            int: The number of tokens.
        """
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def calculate_cost(self, total_tokens: int) -> float:
        return total_tokens * settings.OPEN_AI_EMBEDDING_PRICE_PER_TOKEN
    
    def create_vectorstore(self, doc_uuid: str, text_chunks: str):
        self.total_tokens = 0
        self.price_usage = 0
                
        # if not found create new vector        
        faiss_index = FAISS.from_texts(text_chunks, self.embeddings)
        self.total_tokens = sum(self.calculate_tokens(chunk)
                                for chunk in text_chunks)
        self.price_usage = self.calculate_cost(self.total_tokens)                        
        self.save_faiss_vectors(faiss_index=faiss_index, doc_uuid=doc_uuid) 
        return faiss_index
    
    def get_or_create_vectorstore(self, doc_uuid: str):
        self.total_tokens = 0
        self.price_usage = 0
        faiss_index = self.get_faiss_by_id(doc_uuid)
        # use the existing one from OpenSearch
        if faiss_index is not None:            
            print(f'Vector for document {doc_uuid} already exists.')            
            return faiss_index
        
        # create a new one
        text_chunks = self.create_text_chunks_knowledge_base()
        print("Creating vectorstore...")
        faiss_index = self.create_vectorstore(
            doc_uuid=doc_uuid, text_chunks=text_chunks)
        return faiss_index
                    
    def create_text_chunks_knowledge_base(self):        
        # merge two sources
        print("Extracting text from PDF: ", settings.PDF_FILE_PATH_SOURCE)
        text_chunks_from_pdf = extract_from_pdf(settings.PDF_FILE_PATH_SOURCE)
        print("Extracting text from HTML: ", settings.PAGE_URL_SOURCE)
        text_chunks_from_html = extract_from_html_page(url=settings.PAGE_URL_SOURCE)
        text_chunks = text_chunks_from_pdf + text_chunks_from_html
        # texts = [settings.PDF_FILE_PATH_SOURCE, settings.PAGE_URL_SOURCE]
        # metadata_list = [{'text': text} for text in texts]        
        return text_chunks

    def get_file_name_faiss_index(self, doc_uuid):
        return f"{doc_uuid}_index.faiss"
        
    def save_faiss_vectors(self, faiss_index, doc_uuid):
        faiss_file = self.get_file_name_faiss_index(doc_uuid)
        faiss_index.save_local(faiss_file)        
        print('Vectors successfully saved in local path.', faiss_file)
        
    def get_faiss_by_id(self, doc_uuid):
        faiss_file = self.get_file_name_faiss_index(doc_uuid)
        if not os.path.exists(faiss_file):
            print(f"Vectorstore not found: {faiss_file}")
            return None
        
        print("Loading vectorstore... ", faiss_file)
        new_db = FAISS.load_local(
            faiss_file, self.embeddings,
            allow_dangerous_deserialization=True)
        return new_db
        
    
