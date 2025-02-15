import streamlit as st
from utils import add_sidebar
import json
import re
from contentspot import ContentSpotService
from streamlit.logger import get_logger
import settings

logger = get_logger(__name__)


def get_content_data(code: str) -> dict:
    service = ContentSpotService()
    content = service.get_content(code)
    if content:
        logger.info(json.dumps(content, indent=4))
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
code = params.get("code", "")

st.title("QR Code Chat")
add_sidebar()

if code:    
    with st.spinner('Carregando conteúdo...'):
        content_data = get_content_data(code)

    if not content_data:
        st.error(f"😱 Não foi possível carregar o conteúdo com o código {code}.")
        new_chat()

    if content_data:
        content_type = content_data.get("content_type_slug")
        content_title = content_data.get("title")
        if content_type not in ["pdf"]:
            st.error(f"😢 Esse código {code} é do tipo de conteúdo {content_type} que ainda não é suportado.")
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
            # TODO carregar base conhecimento
            # TODO se nao existir a base de conhecimento, fazer o RAG e salvar no vector store
            # TODO verificar se é necessário refazer a base de conhecimento porque foi atualizada
            # TODO Criar a estrutura de conversação com o chatbot
                                    
else:
    st.write("Nenhum código fornecido na URL. Use a URL no formato: `/qr?code=seu-codigo` (normalmente seu-codigo está no link `https://1mc.co/<seu-codigo>`) ou")        
    new_chat()



