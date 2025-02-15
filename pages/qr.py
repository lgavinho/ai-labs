import streamlit as st
from utils import add_sidebar


# Get the code parameter from URL
params = st.query_params
code = params.get("code", "")

st.title("Midiacode QR Code Chat")
add_sidebar()

if code:
    st.write(f"Código recebido: {code}")
else:
    st.error("Nenhum código fornecido na URL")
    st.write("Use a URL no formato: `/qr?code=seu-codigo`. Normalmente seu-codigo está no link `https://1mc.co/<seu-codigo>`")
