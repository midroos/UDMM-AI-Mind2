import os
import requests
from .interface import LLMInterface

class GeminiModel(LLMInterface):
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        if not self.api_key:
            return f"[GEMINI_API_KEY_NOT_SET] {prompt[:300]}"

        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        try:
            resp = requests.post(self.endpoint, headers=headers, params=params, json=data, timeout=30)
            resp.raise_for_status()
            out = resp.json()
            if "candidates" in out and out["candidates"]:
                candidate = out["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"] and candidate["content"]["parts"]:
                    return candidate["content"]["parts"][0]["text"]
            return f"[GEMINI_UNEXPECTED_RESPONSE] {str(out)}"
        except requests.exceptions.RequestException as e:
            return f"[GEMINI_REQUEST_ERROR] {e}"
        except Exception as e:
            return f"[GEMINI_ERROR] {e}"
