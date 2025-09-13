# interface.py
from abc import ABC, abstractmethod

class LLMInterface(ABC):
    @abstractmethod
    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        ...
