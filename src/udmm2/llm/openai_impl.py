# openai_impl.py
import os
from .interface import LLMInterface
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

class OpenAIModel(LLMInterface):
    def __init__(self, model_name):
        self.model_name = model_name
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        if not self.client:
            return "[OPENAI_KEY_NOT_SET] " + prompt[:300]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[OPENAI_ERROR] {e}"
