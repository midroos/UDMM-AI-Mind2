import os
from openai import OpenAI
from .interface import LLMInterface

class OpenAIModel(LLMInterface):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[OPENAI_ERROR] {e}"
