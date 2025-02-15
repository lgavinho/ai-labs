import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from ai_generator import AIGenerator
from prompt_template import prompt_template
import settings
from vector_db import VectorDatabase
from vector_db_remote import VectorRemoteDatabase
from streamlit.logger import get_logger

logger = get_logger(__name__)

ai = AIGenerator()
db = VectorRemoteDatabase()

# def streamed_response(question: str, my_vectorstore: FAISS):    
#     response = ai.create_text_response(question, my_vectorstore)
#     for word in response.split():
#         yield word + " "
#         time.sleep(0.5)

        
def main():
    """
    Main function to run the Midiacode Chat.
    """
    
    st.title(f"Midiacode Chat")
    st.logo(settings.MIDIACODE_LOGO_URL, link="https://midiacode.com/")
    st.write(f"Desenvolvido por Midiacode AI Labs. Versão {settings.VERSION}")
    st.caption(
        f"Modelos: {settings.LLM_MODEL}, {settings.DALLE_MODEL_VERSION}, {settings.EMBEDDING_MODEL_VERSION}")
    
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0

    if "midiacode_vectorstore" not in st.session_state:
        logger.info("Carregando base de conhecimento...")
        with st.spinner("Carregando base de conhecimento Midiacode..."):
            vector_index = db.get_or_create_vectorstore(settings.SOURCE_UUID)
            st.session_state.midiacode_vectorstore = vector_index
            logger.info("Base de conhecimento criada.")
            st.caption(
                f":money_with_wings: Custo estimado: {db.price_usage:.6f} USD para esta base de conhecimento.")
            st.session_state.total_cost += db.price_usage

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    generate_image = st.toggle("Criador de Imagens IA", False)
    
    st.sidebar.header("Template do Prompt")
    st.sidebar.write(prompt_template)

    assistant_avatar = "robot_dog.png" 
    user_avatar = "anonimous.png"

    # Display chat messages from history on app rerun
    new_history = True
    for message in st.session_state.messages:        
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            new_history = False

    if new_history:    
        logger.info("Nenhum histórico de chat encontrado.")
        with st.chat_message("assistant", avatar=assistant_avatar):
            st.markdown("Como posso ajudar hoje?")

    if prompt := st.chat_input("Escreva aqui..."):
        # Display user message in chat message container
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar=assistant_avatar):
            if generate_image:                                
                image_url = ai.create_image(prompt, size="1024x1024")
                logger.info("URL da Imagem: %s", image_url)
                st.image(image_url, use_container_width=True)
                st.caption(
                    f":money_with_wings: Custo estimado para esta interação: {ai.last_price_usage:.6f} USD")
                st.session_state.total_cost += ai.last_price_usage
            else:
                # Show thinking message
                # Show thinking message with animation
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown(settings.THINKING_ANIMATION, unsafe_allow_html=True)
                
                # Generate response
                answer = ai.create_text_response_with_remote_db(
                    prompt, st.session_state.midiacode_vectorstore, source_id=settings.SOURCE_UUID)                
                
                # Replace thinking message with actual response
                thinking_placeholder.empty()
                response = st.markdown(answer)                
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})
            st.caption(
                f":money_with_wings: Custo estimado para esta interação: {ai.last_price_usage:.6f} USD")
            st.session_state.total_cost += ai.last_price_usage
        st.caption(
            f":moneybag: Custo total estimado nesta sessão: {st.session_state.total_cost:.6f} USD")

st.set_page_config(
    layout="centered", 
    page_title="Midiacode Chat", 
    page_icon=":robot:",
    initial_sidebar_state="collapsed"  # Add this parameter
)


if __name__ == "__main__":
    main()
