from .interface import LLMInterface

class EchoModel(LLMInterface):
    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        return f"[ECHO_MODE] {prompt}"
