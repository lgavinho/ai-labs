import os
VERSION = '0.0.11'

CONTENT_SPOT_BASE_URL = "https://contentspot.midiacode.pt"
MIDIACODE_LOGO_URL = "https://static.midiacode.com/logos/logo-midiacode-main-h.png"
LLM_MODEL = "gpt-4o-mini-2024-07-18"
DALLE_MODEL_VERSION = "dall-e-3"
EMBEDDING_MODEL_VERSION = "text-embedding-3-large"
EMBEDDING_MODEL_DIMENSION = 3072
OPEN_AI_EMBEDDING_PRICE_PER_TOKEN = 0.13/1000000
OPEN_AI_DALLE_PRICE_PER_IMAGE_256X256 = 0.040
OPEN_AI_GPT_PRICE_PER_INPUT_TOKEN = 0.150/1000000
OPEN_AI_GPT_PRICE_PER_OUTPUT_TOKEN = 0.600/1000000
# always change the UUID when you change or update the source document
SOURCE_UUID = "ccc27e35-c964-4259-bc93-11e74cf60b02"
PDF_FILE_PATH_SOURCE = "2024-MidiacodeTextRepository.pdf"
PAGE_URL_SOURCE = "https://ptbr.midiacode.com/2022/02/22/perguntas-frequentes/"

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = "ailabs1"


THINKING_ANIMATION = """
<div style='display: flex; align-items: center; gap: 10px;'>
    <div class='typing-dot'></div>
    <div class='typing-dot'></div>
    <div class='typing-dot'></div>
</div>
<style>
    .typing-dot {
        width: 8px;
        height: 8px;
        background-color: #ffffff;
        border-radius: 50%;
        animation: typing 1s infinite ease-in-out;
    }
    .typing-dot:nth-child(1) { animation-delay: 0.2s; }
    .typing-dot:nth-child(2) { animation-delay: 0.4s; }
    .typing-dot:nth-child(3) { animation-delay: 0.6s; }
    @keyframes typing {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
</style>
"""

