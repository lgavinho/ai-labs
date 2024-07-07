from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class VectorDatabase:
    
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
        return store
