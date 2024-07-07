from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import tiktoken

from settings import EMBEDDING_MODEL_VERSION, OPEN_AI_EMBEDDING_PRICE_PER_TOKEN

class VectorDatabase:
    
    price_usage = 0
    total_tokens = 0
    
    def __init__(self, model_name=''):
        if model_name == '':
            model_name = EMBEDDING_MODEL_VERSION
            
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        
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
        return total_tokens * OPEN_AI_EMBEDDING_PRICE_PER_TOKEN
    
    def create_vectorstore(self, text_chunks: str):
        """
        Creates a vector store from the given text chunks.

        Args:
            text_chunks (list): A list of text chunks.

        Returns:
            FAISS: The created vector store.
        """
        embeddings = OpenAIEmbeddings()
        store = FAISS.from_texts(text_chunks, embeddings)
                
        self.total_tokens = sum(self.calculate_tokens(chunk)
                                for chunk in text_chunks)
        self.price_usage = self.calculate_cost(self.total_tokens)
        return store
