import random
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from prompt_template import get_prompt
from settings import DALLE_MODEL_VERSION, LLM_MODEL, OPEN_AI_DALLE_PRICE_PER_IMAGE_256X256, OPEN_AI_GPT_PRICE_PER_INPUT_TOKEN, OPEN_AI_GPT_PRICE_PER_OUTPUT_TOKEN


class AIGenerator:
    
    last_price_usage = 0
    

    def retrieve_context(self, query: str, db: FAISS):
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
    
    def create_text_response(self, question: str, my_vectorstore: FAISS) -> str:
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
        custom_content = self.retrieve_context(question, my_vectorstore)

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
        self.last_price_usage = 0 
        response_metadata = response.response_metadata
        if response_metadata:
            token_usage = response_metadata.get('token_usage')
            print("Tokens usage: ", token_usage)
            if token_usage:                                
                input_price = token_usage.get('prompt_tokens', 0) * OPEN_AI_GPT_PRICE_PER_INPUT_TOKEN
                out_price = token_usage.get(
                    'completion_tokens', 0) * OPEN_AI_GPT_PRICE_PER_OUTPUT_TOKEN
                self.last_price_usage = input_price + out_price
                
        return answer

    def create_image(self, prompt: str, size="1024x1792", quality="standard"):
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
        
        self.last_price_usage = OPEN_AI_DALLE_PRICE_PER_IMAGE_256X256
        
        return image_url
