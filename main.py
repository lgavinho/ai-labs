import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from ai_generator import AIGenerator
from prompt_template import prompt_template
from settings import DALLE_MODEL_VERSION, EMBEDDING_MODEL_VERSION, LLM_MODEL, MIDIACODE_LOGO_URL, SOURCE_UUID
from vector_db import VectorDatabase
from vector_db_remote import VectorRemoteDatabase


VERSION = '0.0.11'

ai = AIGenerator()
db = VectorRemoteDatabase()

# def streamed_response(question: str, my_vectorstore: FAISS):    
#     response = ai.create_text_response(question, my_vectorstore)
#     for word in response.split():
#         yield word + " "
#         time.sleep(0.5)

        
def main():
    """
    Main function to run the Midiacode Chatbot.
    """
    
    st.title(f"Midiacode Chatbot")
    st.logo(MIDIACODE_LOGO_URL, link="https://midiacode.com/")
    st.write(f"Powered by Midiacode AI Labs. Version {VERSION}")
    st.caption(
        f"Models: {LLM_MODEL}, {DALLE_MODEL_VERSION}, {EMBEDDING_MODEL_VERSION}")
    
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0

    if "midiacode_vectorstore" not in st.session_state:
        print("Loading vectorstore...")
        with st.spinner("Loading Midiacode knowledge base..."):
            vector_index = db.get_or_create_vectorstore(SOURCE_UUID)
            st.session_state.midiacode_vectorstore = vector_index
            print("Vectorstore created.")
            st.caption(
                f":money_with_wings: Cost estimate: {db.price_usage:.6f} USD for this knowledge base.")
            st.session_state.total_cost += db.price_usage

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    generate_image = st.toggle("AI Image Creator", False)
    
    st.sidebar.header("Prompt Template")
    st.sidebar.write(prompt_template)

    # Display chat messages from history on app rerun
    new_history = True
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            new_history = False

    if new_history:    
        print("No chat history found.")
        with st.chat_message("assistant"):
            st.markdown("How can I help you?")

    # React to user input
    if prompt := st.chat_input("Write here..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            if generate_image:                                
                image_url = ai.create_image(prompt, size="1024x1024")
                print("Image URL: ", image_url)
                st.image(image_url, use_column_width=True)
                st.caption(
                    f":money_with_wings: Cost estimate for this interaction: {ai.last_price_usage:.6f} USD")
                st.session_state.total_cost += ai.last_price_usage
            else:
                # for Pinecone VectorRemoteDatabase use ai.create_text_response_with_remote_db
                # for local VectorDatabase use ai.create_text_response()
                answer = ai.create_text_response_with_remote_db(
                    prompt, st.session_state.midiacode_vectorstore, source_id=SOURCE_UUID)                
                response = st.markdown(answer)                
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})
                st.caption(
                    f":money_with_wings: Cost estimate for this interaction: {ai.last_price_usage:.6f} USD")
                st.session_state.total_cost += ai.last_price_usage
            st.caption(
                f":moneybag: Total cost estimate in this session: {st.session_state.total_cost:.6f} USD")


st.set_page_config(
    layout="centered", page_title="Midiacode Chatbot", page_icon=":robot:")


if __name__ == "__main__":
    main()
