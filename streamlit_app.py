import streamlit as st
import settings

st.set_page_config(
    layout="centered", 
    page_title="Midiacode Chat", 
    page_icon=":robot:",
    initial_sidebar_state="collapsed"  # Add this parameter
)

st.logo(settings.MIDIACODE_LOGO_URL, link="https://midiacode.com/")

home_page = st.Page("pages/home.py", title="Midiacode Chat", icon=":material/robot:")
qrcode_page = st.Page("pages/qr.py", title="QR Code Chat", icon=":material/qr_code:")
pg = st.navigation([
    home_page,
    qrcode_page
])

pg.run()