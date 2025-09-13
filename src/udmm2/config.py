# config.py - إعدادات مركزية للوكيل
LLM_PROVIDER = "gemini"   # "openai" | "llamacpp" | "echo" | "gemini"
OPENAI_MODEL = "gpt-4o-mini"
LLAMACPP_SERVER = "http://127.0.0.1:8080/completion"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
FAISS_INDEX_PATH = "data/faiss.index"
FAISS_META_PATH = "data/faiss_meta.json"
ENV_ALLOW_LEARN = True
API_HOST = "0.0.0.0"
API_PORT = 8000
