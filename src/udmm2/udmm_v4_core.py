"""
udmm_v4_core.py
Integrated UDMM v4 core with AttractorDynamics, IntentManager, MemoryModule,
and a Gemini-specific LLM wrapper.
"""

import os
import json
import time
import math
import uuid
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import requests
from scipy.special import expit as sigmoid

# Optional import for sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
    SENTE = True
except ImportError:
    SENTE = False

# --- MemoryModule ---
class MemoryModule:
    def __init__(self, tau_alg: float = 0.8, kappa: float = 0.6, alpha: float = 1.0, beta: float = 0.5):
        self.schemas: List[Tuple[np.ndarray, float]] = []
        self.tau_alg, self.kappa, self.alpha, self.beta = tau_alg, kappa, alpha, beta

    def evaluate(self, schema: np.ndarray, global_intention: float) -> float:
        if not self.schemas: return float(np.mean(schema))
        existing_schemas = np.vstack([s for s, _ in self.schemas])
        mean_schema = existing_schemas.mean(axis=0)
        denom = (np.linalg.norm(mean_schema) * np.linalg.norm(schema) + 1e-9)
        return (float(np.dot(mean_schema, schema) / denom) + 1.0) / 2.0

    def step(self, theta: np.ndarray, q_post: np.ndarray, p_prior: np.ndarray, error: np.ndarray, global_intention: float, lr: float = 0.1) -> Tuple[np.ndarray, float]:
        p, q = np.asarray(p_prior, dtype=float) + 1e-10, np.asarray(q_post, dtype=float) + 1e-10
        IT = float(np.sum(p * np.log(p / q)))
        if IT <= self.tau_alg:
            theta -= lr * error
        else:
            A = sigmoid(self.alpha * (IT - self.tau_alg) + self.beta * float(np.linalg.norm(error)))
            if A > self.kappa:
                e_norm = error / (np.linalg.norm(error) + 1e-9)
                new_schema = np.concatenate([e_norm, [global_intention]])
                if self.evaluate(new_schema, global_intention) > self.kappa:
                    self.schemas.append((new_schema, 1.0))
        return theta, IT

# --- AttractorDynamics ---
class AttractorDynamics:
    def __init__(self, dimensionality: int = 5, alpha: float = 0.1, beta: float = 0.3, seed: Optional[int] = None):
        if seed is not None: np.random.seed(seed)
        self.dimensionality = dimensionality
        self.alpha, self.beta = alpha, beta
        self.attractor_state = np.random.dirichlet(np.ones(dimensionality))
        self.labels = ["Ego", "Social", "Symbolic", "Physical", "Cultural"][:dimensionality]
        self.history: List[np.ndarray] = [self.attractor_state.copy()]
        self.dynamics_matrix = np.random.randn(dimensionality, dimensionality) * 0.05
        np.fill_diagonal(self.dynamics_matrix, 1.0)
        self.coupling_matrix = np.random.randn(dimensionality, dimensionality) * 0.05
        np.fill_diagonal(self.coupling_matrix, 0.0)

    def update(self, external_influence: np.ndarray, internal_tension: float) -> np.ndarray:
        influence = np.resize(np.asarray(external_influence, dtype=float), self.dimensionality)
        delta = self.alpha * (self.beta * influence + self.dynamics_matrix @ self.attractor_state + self.coupling_matrix @ self.attractor_state - internal_tension * self.attractor_state)
        new_state = np.clip(self.attractor_state + delta, 0.0, 1.0)
        self.attractor_state = new_state / (new_state.sum() or 1.0)
        self.history.append(self.attractor_state.copy())
        return self.attractor_state

    def get_state(self) -> np.ndarray:
        return self.attractor_state

# --- IntentManager ---
class IntentManager:
    def __init__(self, dimensionality: int = 5):
        self.dim = dimensionality
        self.intent_hierarchy = {k: np.random.dirichlet(np.ones(self.dim)) for k in ["structural", "self", "symbolic"]}
        self.history = {k: [v.copy()] for k, v in self.intent_hierarchy.items()}

    def update(self, attractor_state: np.ndarray, external_influence: np.ndarray, internal_tension: float, lr: float = 0.08):
        for level, factor in {"structural": 1.0, "self": 1.5, "symbolic": 2.0}.items():
            delta = lr * ((attractor_state * factor) + (np.asarray(external_influence) * 0.5) - (internal_tension * self.intent_hierarchy[level]))
            new_intent = np.clip(self.intent_hierarchy[level] + delta, 0.0, 1.0)
            self.intent_hierarchy[level] = new_intent / (new_intent.sum() or 1.0)
            self.history[level].append(new_intent.copy())

    def get_hierarchy(self) -> Dict[str, np.ndarray]:
        return self.intent_hierarchy

# --- PromptBuilder ---
class PromptBuilder:
    def __init__(self, attractor: AttractorDynamics, intent: IntentManager):
        self.attractor, self.intent = attractor, intent
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') if SENTE else None
        self.memory_texts: List[str] = []

    def build_prompt(self, user_input: str, memory_context: List[str] = []) -> str:
        context = {
            "attractor_state": dict(zip(self.attractor.labels, self.attractor.get_state().tolist())),
            "intent_hierarchy": {k: v.tolist() for k, v in self.intent.get_hierarchy().items()},
            "memory_context": memory_context[:5]
        }
        return (
            "You are an AI agent implementing the Unified Dynamic Model of Mind (UDMM).\n"
            "Respond in Arabic when possible and reflect your internal state.\n\n"
            f"INTERNAL_STATE:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
            f"USER_INPUT:\n{user_input}\n\n"
            "Please answer concisely."
        )

    def add_memory_text(self, text: str):
        self.memory_texts.insert(0, text)
        self.memory_texts = self.memory_texts[:5000]

    def retrieve_memory_texts(self, query: str, top_k: int = 3) -> List[str]:
        if not self.embedder or not self.memory_texts: return []
        q_emb = self.embedder.encode(query, convert_to_numpy=True)
        embs = self.embedder.encode(self.memory_texts, convert_to_numpy=True)
        sims = util.cos_sim(q_emb, embs).cpu().numpy().flatten()
        return [self.memory_texts[i] for i in np.argsort(-sims)[:top_k]]

# --- Gemini LLM Wrapper ---
class GeminiLLMWrapper:
    def __init__(self, api_key: Optional[str], model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    def chat(self, prompt: str, temperature: float = 0.3) -> str:
        if not self.api_key:
            return "[ERROR: Gemini API Key not provided]"

        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 800}
        }
        try:
            resp = requests.post(self.endpoint, headers=headers, params=params, json=data, timeout=20)
            resp.raise_for_status()
            out = resp.json()
            if "candidates" in out and out["candidates"]:
                candidate = out["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"] and candidate["content"]["parts"]:
                    return candidate["content"]["parts"][0]["text"]
            return f"[GEMINI UNEXPECTED RESPONSE] {str(out)[:500]}"
        except requests.exceptions.RequestException as e:
            return f"[GEMINI REQUEST ERROR] {e}"

# --- UDMMCore ---
class UDMMCore:
    def __init__(self, dimensionality: int = 5, memory_params: dict = None, gemini_api_key: Optional[str] = None):
        self.attractor = AttractorDynamics(dimensionality=dimensionality)
        self.intent = IntentManager(dimensionality=dimensionality)
        self.prompt_builder = PromptBuilder(self.attractor, self.intent)
        self.memory = MemoryModule(**(memory_params or {}))
        self.theta = np.zeros(dimensionality + 1)
        self.time_step, self.internal_tension, self.system_health = 0, 0.0, 1.0
        self.history: List[Dict[str, Any]] = []
        self.llm = GeminiLLMWrapper(api_key=gemini_api_key)

    def _estimate_q_post_and_prior(self, user_input: str) -> Tuple[np.ndarray, np.ndarray]:
        if self.prompt_builder.embedder:
            e = self.prompt_builder.embedder.encode(user_input, convert_to_numpy=True)
            d = min(len(e), len(self.theta))
            q_post = np.abs(e[:d]) / (np.sum(np.abs(e[:d])) + 1e-9)
            p_prior = np.abs(self.theta[:d] + 1e-6) / (np.sum(np.abs(self.theta[:d] + 1e-6)) + 1e-9)
            if len(q_post) < len(self.theta):
                q_post = np.pad(q_post, (0, len(self.theta) - len(q_post)), 'constant', constant_values=1e-6)
                p_prior = np.pad(p_prior, (0, len(self.theta) - len(p_prior)), 'constant', constant_values=1e-6)
            return q_post, p_prior
        L = len(self.theta)
        q_post = np.ones(L) * (len(user_input.split()) + 1)
        return q_post / q_post.sum(), (np.abs(self.theta) + 1.0) / (np.abs(self.theta) + 1.0).sum()

    def process_input(self, user_input: str) -> Dict[str, Any]:
        q_post, p_prior = self._estimate_q_post_and_prior(user_input)
        error = q_post - p_prior
        g_intent = float(self.intent.get_hierarchy()["symbolic"].mean())
        _, self.internal_tension = self.memory.step(self.theta, q_post, p_prior, error, g_intent, lr=0.0)

        ext_influence = np.random.dirichlet(np.ones(self.attractor.dimensionality))
        new_state = self.attractor.update(ext_influence, self.internal_tension)
        self.intent.update(new_state, ext_influence, self.internal_tension)

        mem_texts = self.prompt_builder.retrieve_memory_texts(user_input)
        prompt = self.prompt_builder.build_prompt(user_input, memory_context=mem_texts)
        llm_resp = self.llm.chat(prompt)

        self.prompt_builder.add_memory_text(f"Q:{user_input} A:{llm_resp}")
        self.theta, _ = self.memory.step(self.theta, q_post, p_prior, error, g_intent, lr=0.05)
        self.system_health = max(0.1, 1.0 - self.internal_tension * 0.5)

        entry = {"ts": time.time(), "step": self.time_step, "input": user_input, "response": llm_resp, "internal_tension": self.internal_tension, "attractor_state": self.attractor.get_state().tolist(), "intent": {k: v.tolist() for k, v in self.intent.get_hierarchy().items()}}
        self.history.append(entry)
        self.time_step += 1
        return entry

# --- Quick Test ---
if __name__ == "__main__":
    print("Starting UDMMCore Gemini-only demo (local) ...")
    api_key_from_env = os.environ.get("GEMINI_API_KEY")
    core = UDMMCore(dimensionality=5, gemini_api_key=api_key_from_env)

    if not api_key_from_env:
        print("\nWARNING: GEMINI_API_KEY environment variable not found. LLM calls will fail.")
        print("You can set it with: export GEMINI_API_KEY='your_key_here'\n")

    samples = ["ما هو النموذج الديناميكي الموحد للعقل؟", "اشرح لي حالة الجاذب الرمزي."]
    for q in samples:
        out = core.process_input(q)
        print(f"> Q: {q}")
        print(f"  -> Resp: {out.get('response', 'N/A')[:200]}...")
        print(f"  -> IT: {out.get('internal_tension', 0.0):.4f}")
        print("-" * 60)

    print("Demo finished.")
