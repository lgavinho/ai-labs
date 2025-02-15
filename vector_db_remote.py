import json
import os
import sqlite3
import numpy as np
import faiss
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from requests_aws4auth import AWS4Auth
from opensearchpy.helpers import bulk
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pinecone import Pinecone, ServerlessSpec
import tiktoken
import settings
from utils import extract_from_html_page, extract_from_pdf
from streamlit.logger import get_logger


logger = get_logger(__name__)


class VectorRemoteDatabase:
    
    price_usage = 0
    total_tokens = 0
    
    def __init__(self, model_name=''):
        if model_name == '':
            model_name = settings.EMBEDDING_MODEL_VERSION
            
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name) 
        self.embeddings = OpenAIEmbeddings(model=model_name)  # Specify the model here
        
        self.pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.INDEX_NAME

    def create_index_if_not_exist(self): 
        try:
            index_data = self.pinecone.describe_index(self.index_name)
            logger.info(index_data)     
            if index_data and index_data['status']['ready']:
                logger.info("Index already exists.")
                return True
        except Exception as e:
            logger.error(f"Error: {e}")
            
        logger.info(f"Index {self.index_name} does not exist. Creating...")
        self.pinecone.create_index(
            name=self.index_name,
            dimension=settings.EMBEDDING_MODEL_DIMENSION, 
            metric="cosine", # better for semantic search
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
        logger.info("Index created.") 
        return True       
            
                        
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
                
        # when not found create new vector        
        faiss_index = FAISS.from_texts(text_chunks, self.embeddings)
        self.total_tokens = sum(self.calculate_tokens(chunk)
                                for chunk in text_chunks)
        self.price_usage = self.calculate_cost(self.total_tokens)                                
        vector_index = self.save_faiss_vectors(faiss_index=faiss_index, doc_uuid=doc_uuid) 
        return vector_index
    
    def get_or_create_vectorstore(self, doc_uuid: str):
        self.total_tokens = 0
        self.price_usage = 0        
        vector_index = self.get_pinecone_index()
                
        logger.info("Getting vectorstore for namespace: %s", doc_uuid)
        if self.namespace_exists(namespace=doc_uuid):
            logger.info("Namespace %s already exists.", doc_uuid)
            return vector_index

        text_chunks = self.create_text_chunks_knowledge_base()
        logger.info("Creating vectorstore...")
        vector_index = self.create_vectorstore(
            doc_uuid=doc_uuid, text_chunks=text_chunks)
        return vector_index
                    
    def create_text_chunks_knowledge_base(self):        
        # merge two sources
        logger.info("Extracting text from PDF: %s", settings.PDF_FILE_PATH_SOURCE)
        text_chunks_from_pdf = extract_from_pdf(settings.PDF_FILE_PATH_SOURCE)
        logger.info("Extracting text from HTML: %s", settings.PAGE_URL_SOURCE)
        text_chunks_from_html = extract_from_html_page(url=settings.PAGE_URL_SOURCE)
        text_chunks = text_chunks_from_pdf + text_chunks_from_html
        # texts = [settings.PDF_FILE_PATH_SOURCE, settings.PAGE_URL_SOURCE]
        # metadata_list = [{'text': text} for text in texts]     
        logger.info("Text chunks: %d", len(text_chunks))   
        return text_chunks

    def get_pinecone_index(self):
        self.create_index_if_not_exist()
        while not self.pinecone.describe_index(self.index_name).status['ready']:
            logger.info('Index not ready. Waiting...')
            time.sleep(1)
        return self.pinecone.Index(self.index_name)
        
    def save_faiss_vectors(self, faiss_index, doc_uuid):
        logger.info('Saving vectors to pinecone...')        
        dimension = len(faiss_index.index.reconstruct(0))
        logger.info('Vector dimension: %d', dimension)
        pinecone_index = self.get_pinecone_index()

        # Preparar dados para upsert
        vectors_to_upsert = []
        for i in range(faiss_index.index.ntotal):
            logger.info('Vector %d of %d', i, faiss_index.index.ntotal)
            # Extrair vetor
            vector = faiss_index.index.reconstruct(i).tolist()
            
            # Extrair documento correspondente
            doc_id = faiss_index.index_to_docstore_id[i]
            doc = faiss_index.docstore.search(doc_id)
            metadata = {"text": doc.page_content}
            
            # Usar doc_id como ID no Pinecone (ou gere um único)
            logger.info('Appending vector %d of %d with id %s', i, faiss_index.index.ntotal, doc_id)
            vectors_to_upsert.append((str(doc_id), vector, metadata))
        
        # Enviar em lotes (ex: 100 vetores por lote)
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            logger.info('Batch %d of %d', i, len(vectors_to_upsert))
            pinecone_index.upsert(vectors=vectors_to_upsert[i:i+batch_size], namespace=doc_uuid)
                
        logger.info('Vectors successfully saved in pinecone.')
        return pinecone_index

    def namespace_exists(self, namespace: str) -> bool:        
        index = self.get_pinecone_index()
        stats = index.describe_index_stats()
        logger.info(stats)
        return namespace in stats['namespaces']
