# llamacpp_impl.py
import requests
from .interface import LLMInterface

class LlamaCPPModel(LLMInterface):
    def __init__(self, server_url: str):
        self.server_url = server_url

    def chat(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        try:
            payload = {"prompt": prompt, "temperature": temperature, "n_predict": max_tokens}
            r = requests.post(self.server_url, json=payload, timeout=30)
            r.raise_for_status()
            j = r.json()
            # خذ المحتوى حسب صيغة خادمك
            return j.get("content") or j.get("text") or str(j)
        except Exception as e:
            return f"[LLAMA_CPP_ERROR] {e}"
