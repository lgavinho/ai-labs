import streamlit as st
from utils import add_sidebar
import json
import re
from langchain.prompts import PromptTemplate
from contentspot import ContentSpotService
from streamlit.logger import get_logger
import settings
from ai_generator import AIGenerator
from vector_db_remote import VectorRemoteDatabase
from prompt_template import get_prompt, prompt_template_generic

logger = get_logger(__name__)

# Example: https://1mc.co/zh9gTW

def get_content_data(code: str) -> dict:
    service = ContentSpotService()
    content = service.get_content(code)
    if content:
        # logger.info(json.dumps(content, indent=4))
        return content
    return None

def new_chat():
    url = st.text_input("Entre uma URL 1mc.co (ex: https://1mc.co/140uKUqP)")
    if st.button("Chat", key="chat_button"):
        if url:
            pattern = r'(?:https?:\/\/)?1mc\.co\/(\w+)'
            match = re.match(pattern, url)            
            if match:
                extracted_code = match.group(1)
                st.query_params["code"] = extracted_code
                st.rerun()
            else:
                st.error("Por favor, insira uma URL válida no formato: https://1mc.co/code")    


# Get the code parameter from URL
params = st.query_params
short_code = params.get("code", "")

st.title("QR Code Chat")
add_sidebar()

ai_prompt = PromptTemplate(
    input_variables=["question", "custom_content", "content_title"],
    template=prompt_template_generic
)
ai = AIGenerator(template_prompt=ai_prompt, chat_type="qrcode")
db = VectorRemoteDatabase()

if short_code:    
    with st.spinner('Carregando conteúdo...'):
        content_data = get_content_data(short_code)

    if not content_data:
        st.error(f"😱 Não foi possível carregar o conteúdo com o código {short_code}.")
        new_chat()

    if content_data:
        content_type = content_data.get("content_type_slug")
        content_title = content_data.get("title")
        if content_type not in ["pdf"]:
            st.error(f"😢 Esse código {short_code} é do tipo de conteúdo {content_type} que ainda não é suportado.")
            new_chat()
            
        if content_type == "pdf":
            source_url = content_data.get("source_url")
            with st.expander(f"{content_title} ({content_type})"):
                cover_url = content_data.get("cover_url")
                if cover_url:
                    st.image(cover_url, width=200)                
                qrcode_url = content_data.get("qrcode_url")
                if qrcode_url:
                    st.image(qrcode_url, width=100)
                st.write(content_data.get("short_link"))  
            # Chat             
            vector_store_session_id = f"{short_code}_vectorstore"
            if vector_store_session_id not in st.session_state:
                logger.info("Carregando base de conhecimento...")
                with st.spinner("Carregando base de conhecimento..."):     
                    message_placeholder = st.empty()
                    message_placeholder.write("O primeiro acesso ao conteúdo pode levar alguns minutos. Aguarde...")                                
                    vector_index = db.get_or_create_vectorstore(doc_uuid=short_code, source_url=source_url)
                    st.session_state[vector_store_session_id] = vector_index
                    message_placeholder.empty()  # Remove a mensagem após concluir
                logger.info("Base de conhecimento criada.")
                st.caption(f"Base de conhecimento carregada.")
                st.caption(
                    f":money_with_wings: Custo estimado: {db.price_usage:.6f} USD para esta base de conhecimento.")
                st.session_state.total_cost += db.price_usage
            
            # TODO Criar a estrutura de conversação com o chatbot
            # Initialize chat history for this session code
            history_message_id = f"{short_code}_messages"
            if history_message_id not in st.session_state:
                st.session_state[history_message_id] = []
            assistant_avatar = "icone_midiacode.png"
            user_avatar = "user.png"
            # Display chat messages from history on app rerun
            new_history = True
            for message in st.session_state[history_message_id]:        
                avatar = assistant_avatar if message["role"] == "assistant" else user_avatar
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])
                    new_history = False

            if new_history:    
                logger.info("Nenhum histórico de chat encontrado.")
                with st.chat_message("assistant", avatar=assistant_avatar):
                    st.markdown("**Olá! Bem-vindo(a) ao QR Code Chat!** 🤖✨")
                    st.markdown((
                        "Sou um assistente virtual alimentado por inteligência artificial, "
                        f"criado para oferecer informações rápidas exclusivamente sobre o conteúdo {content_title}. "
                        "Quer entender melhor os conceitos principais, tirar dúvidas sobre tópicos específicos ou "
                        "explorar como aplicar esse conhecimento em seus projetos?"
                    ))
                    st.markdown("Como posso ajudar? 😊")

            if prompt := st.chat_input("Pergunte aqui..."):
                # Display user message in chat message container
                with st.chat_message("user", avatar=user_avatar):
                    st.markdown(prompt)
                # Add user message to chat history
                st.session_state[history_message_id].append({"role": "user", "content": prompt})

                with st.chat_message("assistant", avatar=assistant_avatar):
                    # Show thinking message with animation
                    thinking_placeholder = st.empty()
                    thinking_placeholder.markdown(settings.THINKING_ANIMATION, unsafe_allow_html=True)
                    
                    # Generate response
                    answer = ai.create_text_response_with_remote_db(
                        prompt, st.session_state[vector_store_session_id], source_id=short_code,
                        add_midiacode_ads=False,
                        content_title=content_title)
                    
                    # Replace thinking message with actual response
                    thinking_placeholder.empty()
                    response = st.markdown(answer)                
                    # Add assistant response to chat history
                    st.session_state[history_message_id].append(
                        {"role": "assistant", "content": answer})
                    st.caption(
                        f":money_with_wings: Custo estimado para esta interação: {ai.last_price_usage:.6f} USD")
                    st.session_state.total_cost += ai.last_price_usage
                st.caption(
                    f":moneybag: Custo total estimado nesta sessão: {st.session_state.total_cost:.6f} USD")                                           
else:
    st.write("Nenhum código fornecido na URL. Use a URL no formato: `/qr?code=seu-codigo` (normalmente seu-codigo está no link `https://1mc.co/<seu-codigo>`) ou")        
    new_chat()



