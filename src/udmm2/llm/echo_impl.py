# echo_impl.py
from .interface import LLMInterface

class EchoModel(LLMInterface):
    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        # مفيدة للتجربة بدون LLM خارجي
        head = prompt.replace("\n", " ")[:800]
        return f"[ECHO] {head}"
