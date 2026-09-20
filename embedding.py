import os
from dotenv import load_dotenv
from langchain.embeddings import init_embeddings

load_dotenv(override=True)

embedding_model = init_embeddings(
    model="openai:BAAI/bge-m3",
    api_key=os.getenv("EMBED_API_KEY"),
    base_url=os.getenv("EMBED_BASE_URL"),
)