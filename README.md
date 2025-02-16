# AI Labs

This repository contains the code for the AI Labs project.
Use: Open AI model.

The `main.py` file contains two functions: `generate_response` and `main`. The `generate_response` function generates a chatbot response by creating an LLM chain, initializing a `ChatOpenAI` instance, retrieving a prompt, and invoking the chain with the user's question and retrieved context. The `main` function serves as the program's entry point, setting up a Streamlit interface. It collects user input, calls `generate_response` upon clicking "Enviar" (Send), and displays the generated response. In essence, `generate_response` processes the chatbot logic, while `main` handles the user interface.

This project uses PDF files as context, including:
- Midiacode company documentation
- Any PDF content published on Midiacode Studio platform (through QR Code Chat feature)

### QR Code Chat Feature
The application now includes a QR Code Chat feature that allows users to interact with PDF content published on Midiacode Studio. Users can:
- Access content through Midiacode's short URLs (1mc.co)
- Chat with AI about the PDF content
- Get instant responses based on the document's context

All document embeddings are stored in Pinecone vector database for efficient retrieval and persistent storage, enabling fast and accurate responses even with large documents.

## Tech Stack

This project uses the following technologies and packages:

### Core Technologies
- Python 3.11.4
- Poetry (Package Management)
- Streamlit (Web Interface)

### AI/ML Packages
- OpenAI API
- LangChain

### Storage & Vector Databases
- Pinecone
- FAISS

## Local Setup

To set up the project locally, follow these steps:

1. Create a virtual environment using `pyenv`:

   ```shell
   pyenv virtualenv 3.11.4 ai-labs
   ```

2. Set the local Python version to the newly created virtual environment:

   ```shell
   pyenv local ai-labs
   ```

3. Initialize the project with Poetry:

   ```shell
   poetry init
   ```

4. Activate the Poetry shell:

   ```shell
   poetry shell
   ```

5. Install the project dependencies:

   ```shell
   poetry install
   ```

## Run

To run the project, follow these steps:

1. Set the OpenAI API key as an environment variable:

   ```shell
   export OPENAI_API_KEY={add your api key}
   export AWS_ACCESS_KEY={add AWS Access Key}
   export AWS_SECRET_KEY={add AWS Secret Key}
   export PINECONE_API_KEY={add Pinecone API Key}
   ```

2. Run the main script using Streamlit:

   ```shell
   streamlit run main.py
   ```

## References

Here are some helpful references related to this project:

- [How to Build a PDF Chatbot with LangChain and Faiss](https://kevincoder.co.za/how-to-build-a-pdf-chatbot-with-langchain-and-faiss)
- [YouTube: AI Labs Demo](https://youtu.be/rOjusRRO1EI?si=KFhcJ4FH4eCxdCGG&t=741)
- [LLM Chains using Runnables](https://medium.com/@manoj-gupta/llm-chains-using-runnables-df500d2b7490)
- [YouTube: AI Labs Tutorial](https://www.youtube.com/watch?v=moJRxxEddzU)
- [Streamlit: Build a basic LLM chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)
