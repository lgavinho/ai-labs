import random
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from prompt_template import get_prompt
import settings
from streamlit.logger import get_logger


logger = get_logger(__name__)

class AIGenerator:
    
    last_price_usage = 0

    def __init__(self):
        self.pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL_VERSION)
    
    def retrieve_context(self, query: str, db: FAISS):
        logger.info("Retrieving context for question: %s", query)
        similar_response = db.similarity_search(query, k=20)
        return similar_response

    def retrieve_context_from_remote(self, query: str, db_index, source_id: str):  
        query_embedding = self.embeddings.embed_query(query)         
        results = db_index.query(
            vector=query_embedding,
            namespace=source_id,
            top_k=20,
            include_values=False,
            include_metadata=True            
        )        
        if not results.matches:
            logger.warning("No matches found in vector database!!!")
            return None
            
        context_text = "\n".join([match.metadata['text'] for match in results.matches])        
        return context_text

    
    def create_text_response(self, question: str, my_vectorstore: FAISS) -> str:
        """
        Generates a response to the given question.

        Args:
            question (str): The question string.

        Returns:
            str: The generated response.
        """

        logger.info("Creating LLM chain...")
        llm = ChatOpenAI(temperature=0, model=settings.LLM_MODEL)
        prompt = get_prompt()
        chain = prompt | llm

        logger.info("Retrieving context for question: %s", question)
        custom_content = self.retrieve_context(question, my_vectorstore)

        logger.info("Invoking chain...")
        inputs = {
            "question": question,
            "custom_content": custom_content
        }
        response = chain.invoke(inputs)

        answer = response.content
        if random.choice(['yes', 'no']) == 'yes':
            footer_message = "Se preferir, pode acessar nosso site [midiacode.com](https://midiacode.com/) e também solicitar um chat com nossa equipe."
            answer += "\n\n" + footer_message

        logger.info("Generated answer: %s", answer)

        # getting usage of tokens       
        self.last_price_usage = 0 
        response_metadata = response.response_metadata
        if response_metadata:
            token_usage = response_metadata.get('token_usage')
            logger.info("Tokens usage: %s", token_usage)
            if token_usage:                                
                input_price = token_usage.get('prompt_tokens', 0) * settings.OPEN_AI_GPT_PRICE_PER_INPUT_TOKEN
                out_price = token_usage.get(
                    'completion_tokens', 0) * settings.OPEN_AI_GPT_PRICE_PER_OUTPUT_TOKEN
                self.last_price_usage = input_price + out_price
        return answer

    def create_text_response_with_remote_db(self, question: str, my_vectorstore, source_id: str) -> str:
        # TODO use doc id to retrieve context from different names
        logger.info("Creating LLM chain...")
        llm = ChatOpenAI(temperature=0, model=settings.LLM_MODEL)
        prompt = get_prompt()
        chain = prompt | llm

        logger.info("Retrieving context for question: %s", question)
        custom_content = self.retrieve_context_from_remote(question, my_vectorstore, source_id)

        logger.info("Invoking chain...")
        inputs = {
            "question": question,
            "custom_content": custom_content
        }
        response = chain.invoke(inputs)

        answer = response.content
        if random.choice(['yes', 'no']) == 'yes':
            footer_message = "Se preferir, pode acessar nosso site [midiacode.com](https://midiacode.com/) e também solicitar um chat com nossa equipe."
            answer += "\n\n" + footer_message

        logger.info("Generated answer: %s", answer)

        # getting usage of tokens       
        self.last_price_usage = 0 
        response_metadata = response.response_metadata
        if response_metadata:
            token_usage = response_metadata.get('token_usage')
            logger.info("Tokens usage: %s", token_usage)
            if token_usage:                                
                input_price = token_usage.get('prompt_tokens', 0) * settings.OPEN_AI_GPT_PRICE_PER_INPUT_TOKEN
                out_price = token_usage.get(
                    'completion_tokens', 0) * settings.OPEN_AI_GPT_PRICE_PER_OUTPUT_TOKEN
                self.last_price_usage = input_price + out_price

        return answer
    

    def create_image(self, prompt: str, size="1024x1792", quality="standard"):
        logger.info("Generating image...")
        client = OpenAI()

        response = client.images.generate(
            model=settings.DALLE_MODEL_VERSION,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )
        logger.info("Image generation response: %s", response)
        image_url = response.data[0].url
        
        self.last_price_usage = settings.OPEN_AI_DALLE_PRICE_PER_IMAGE_256X256
        
        return image_url
