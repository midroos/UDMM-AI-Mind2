"""
udmm_v4_core.py
Integrated UDMM v4 core with AttractorDynamics, IntentManager, MemoryModule, PromptBuilder and LLM wrapper.
Put this file under src/udmm2/ or the project root and import UDMMCore.
"""

import os
import json
import time
import math
import uuid
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# Optional imports: sentence-transformers (for embeddings), openai
try:
    from sentence_transformers import SentenceTransformer, util
    SENTE = True
except Exception:
    SENTE = False

try:
    import openai
    OPENAI_PY_AVAILABLE = True
except Exception:
    OPENAI_PY_AVAILABLE = False

# -------------------------
# MemoryModule (from Appendix)
# -------------------------
from scipy.special import expit as sigmoid

def KL_div(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float) + 1e-10
    q = np.asarray(q, dtype=float) + 1e-10
    return float(np.sum(p * np.log(p / q)))

def gradient(error: np.ndarray) -> np.ndarray:
    # Placeholder gradient (identity). Replace with learned gradient if available.
    return np.asarray(error, dtype=float)

def deltaF(error: np.ndarray) -> float:
    return float(np.linalg.norm(error))

def propose_schema(error: np.ndarray, global_intention: float) -> np.ndarray:
    # Create a simple schema vector: concat normalized error + scalar intent
    e = np.asarray(error, dtype=float)
    if e.ndim == 0: e = np.array([e])
    norm = e / (np.linalg.norm(e) + 1e-9)
    return np.concatenate([norm, np.array([global_intention])])

class MemoryModule:
    def __init__(self, tau_alg: float = 0.8, kappa: float = 0.6, alpha: float = 1.0, beta: float = 0.5):
        self.schemas: List[Tuple[np.ndarray, float]] = []  # (vector, weight)
        self.tau_alg = tau_alg
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta

    def evaluate(self, schema: np.ndarray, global_intention: float) -> float:
        # Simple evaluation: cosine similarity to mean existing schema or norm
        if not self.schemas:
            return float(np.mean(schema))  # naive
        existing = np.vstack([s for s, _ in self.schemas])
        mean_schema = existing.mean(axis=0)
        # cosine similarity
        denom = (np.linalg.norm(mean_schema) * np.linalg.norm(schema) + 1e-9)
        cos = float(np.dot(mean_schema, schema) / denom)
        return (cos + 1.0) / 2.0  # scale 0..1

    def step(self,
             theta: np.ndarray,
             q_post: np.ndarray,
             p_prior: np.ndarray,
             error: np.ndarray,
             global_intention: float,
             lr: float = 0.1) -> Tuple[np.ndarray, float]:
        """
        Core memory update step:
         - If IT <= tau_alg: do local param update (theta)
         - Else: compute restructuring signal A, possibly add new schema
        Returns updated theta and computed IT
        """
        IT = KL_div(q_post, p_prior)
        if IT <= self.tau_alg:
            # local update: theta -= lr * grad
            theta = theta - lr * gradient(error)
        else:
            A = sigmoid(self.alpha * (IT - self.tau_alg) + self.beta * deltaF(error))
            if A > self.kappa:
                new_schema = propose_schema(error, global_intention)
                score = self.evaluate(new_schema, global_intention)
                if score > self.kappa:
                    self.schemas.append((new_schema, 1.0))
        return theta, float(IT)

    # Utilities
    def recall_similar(self, query_vec: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return top-k similar schemas (cosine)."""
        if not self.schemas:
            return []
        embeddings = np.vstack([s for s, _ in self.schemas])
        sims = (embeddings @ query_vec) / ((np.linalg.norm(embeddings, axis=1)+1e-9) * (np.linalg.norm(query_vec)+1e-9))
        idxs = list(np.argsort(-sims)[:top_k])
        return [{"schema": self.schemas[i][0].tolist(), "weight": self.schemas[i][1], "score": float(sims[i])} for i in idxs]

    def save(self, path: str):
        data = {"schemas": [[s.tolist(), float(w)] for s, w in self.schemas], "tau_alg": self.tau_alg}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        if not os.path.exists(path): return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.schemas = [(np.array(s), float(w)) for s, w in data.get("schemas", [])]

# -------------------------
# AttractorDynamics
# -------------------------
class AttractorDynamics:
    def __init__(self, dimensionality: int = 5, alpha: float = 0.1, beta: float = 0.3, seed: Optional[int] = None):
        if seed is not None: np.random.seed(seed)
        self.dimensionality = dimensionality
        self.alpha = alpha
        self.beta = beta
        # Start with Dirichlet (sums to 1)
        self.attractor_state = np.random.dirichlet(np.ones(dimensionality))
        self.labels = ["Ego", "Social", "Symbolic", "Physical", "Cultural"][:dimensionality]
        self.history: List[np.ndarray] = [self.attractor_state.copy()]
        self.dynamics_matrix = np.random.randn(dimensionality, dimensionality) * 0.05
        np.fill_diagonal(self.dynamics_matrix, 1.0)
        self.coupling_matrix = np.random.randn(dimensionality, dimensionality) * 0.05
        np.fill_diagonal(self.coupling_matrix, 0.0)

    def update(self, external_influence: np.ndarray, internal_tension: float) -> np.ndarray:
        external_influence = np.asarray(external_influence, dtype=float)
        if external_influence.size != self.dimensionality:
            # map/resize
            external_influence = np.resize(external_influence, self.dimensionality)
        influence_effect = self.beta * external_influence
        dynamic_effect = self.dynamics_matrix @ self.attractor_state
        coupling_effect = self.coupling_matrix @ self.attractor_state
        tension_effect = internal_tension * self.attractor_state
        delta = self.alpha * (influence_effect + dynamic_effect + coupling_effect - tension_effect)
        new_state = self.attractor_state + delta
        new_state = np.clip(new_state, 0.0, 1.0)
        if new_state.sum() == 0:
            new_state = np.ones(self.dimensionality) / self.dimensionality
        else:
            new_state = new_state / new_state.sum()
        self.attractor_state = new_state
        self.history.append(new_state.copy())
        return new_state

    def get_state(self) -> np.ndarray:
        return self.attractor_state

    def compute_influence_score(self, context_vector: np.ndarray) -> float:
        return float(np.dot(self.attractor_state, np.asarray(context_vector, dtype=float)[:self.dimensionality]))

# -------------------------
# IntentManager
# -------------------------
class IntentManager:
    def __init__(self, dimensionality: int = 5):
        self.dim = dimensionality
        self.intent_hierarchy = {
            "structural": np.random.dirichlet(np.ones(self.dim)),
            "self": np.random.dirichlet(np.ones(self.dim)),
            "symbolic": np.random.dirichlet(np.ones(self.dim))
        }
        self.history = {k: [v.copy()] for k, v in self.intent_hierarchy.items()}

    def update(self, attractor_state: np.ndarray, external_influence: np.ndarray,
               internal_tension: float, lr: float = 0.08):
        for level in ["structural", "self", "symbolic"]:
            influence_factor = 1.0 if level == "structural" else (1.5 if level == "self" else 2.0)
            delta = lr * ((attractor_state * influence_factor) + (np.asarray(external_influence) * 0.5) - (internal_tension * self.intent_hierarchy[level]))
            new_intent = self.intent_hierarchy[level] + delta
            new_intent = np.clip(new_intent, 0.0, 1.0)
            if new_intent.sum() == 0: new_intent = np.ones(self.dim) / self.dim
            else: new_intent = new_intent / new_intent.sum()
            self.intent_hierarchy[level] = new_intent
            self.history[level].append(new_intent.copy())

    def get_hierarchy(self) -> Dict[str, np.ndarray]:
        return self.intent_hierarchy

# -------------------------
# PromptBuilder (basic)
# -------------------------
class PromptBuilder:
    def __init__(self, attractor: AttractorDynamics, intent: IntentManager):
        self.attractor = attractor
        self.intent = intent
        # optional sentence-transformer
        if SENTE:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self.embedder = None
        else:
            self.embedder = None
        self.memory_texts: List[str] = []

    def build_prompt(self, user_input: str, memory_context: List[str] = []) -> str:
        context = {
            "attractor_state": dict(zip(self.attractor.labels, self.attractor.get_state().tolist())),
            "intent_hierarchy": {k: v.tolist() for k, v in self.intent.get_hierarchy().items()},
            "memory_context": memory_context[:5]
        }
        prompt = (
            "You are an AI agent implementing the Unified Dynamic Model of Mind (UDMM).\n"
            "Respond in Arabic when possible and reflect the internal attractor state and intent hierarchy.\n\n"
            f"INTERNAL_STATE:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
            f"USER_INPUT:\n{user_input}\n\n"
            "Please answer concisely and if you are uncertain ask for clarification."
        )
        return prompt

    def add_memory_text(self, text: str):
        self.memory_texts.append(text)
        if len(self.memory_texts) > 5000:
            self.memory_texts = self.memory_texts[-5000:]

    def retrieve_memory_texts(self, query: str, top_k: int = 3) -> List[str]:
        # If embedder present, perform semantic retrieval
        if self.embedder:
            q_emb = self.embedder.encode(query, convert_to_numpy=True)
            if not self.memory_texts:
                return []
            embs = self.embedder.encode(self.memory_texts, convert_to_numpy=True)
            sims = util.cos_sim(q_emb, embs).cpu().numpy().flatten()
            idxs = np.argsort(-sims)[:top_k]
            return [self.memory_texts[i] for i in idxs]
        else:
            # fallback: simple substring match
            res = [t for t in self.memory_texts if query in t][:top_k]
            return res

# -------------------------
# Simple LLM wrapper (OpenAI or echo)
# -------------------------
class LLMWrapper:
    def __init__(self, provider: str = "openai", model_name: str = "gpt-4o-mini"):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        if self.provider == "openai" and OPENAI_PY_AVAILABLE:
            openai.api_key = self.api_key

    def chat(self, prompt: str, temperature: float = 0.2) -> str:
        if self.provider == "openai" and OPENAI_PY_AVAILABLE and self.api_key:
            try:
                # best-effort basic call for OpenAI older/newer SDK compatibility
                resp = openai.ChatCompletion.create(model=self.model_name, messages=[{"role":"user","content":prompt}], temperature=temperature, max_tokens=512)
                # try multiple extraction patterns
                if resp and "choices" in resp and len(resp["choices"])>0:
                    text = resp["choices"][0].get("message", {}).get("content") or resp["choices"][0].get("text")
                    return text or "[ERROR: empty response]"
                return "[ERROR: unexpected openai response]"
            except Exception as e:
                return f"[OpenAI ERROR] {e}"
        # fallback (echo)
        snippet = prompt if len(prompt) < 800 else prompt[:800] + "..."
        return f"[ECHO] {snippet}"

# -------------------------
# UDMMCore - integrates everything
# -------------------------
class UDMMCore:
    def __init__(self,
                 dimensionality: int = 5,
                 memory_params: dict = None,
                 llm_provider: str = "echo",
                 llm_model: str = "gpt-4o-mini"):
        self.attractor = AttractorDynamics(dimensionality=dimensionality)
        self.intent = IntentManager(dimensionality=dimensionality)
        self.prompt_builder = PromptBuilder(self.attractor, self.intent)
        self.memory = MemoryModule(**(memory_params or {}))
        # theta: internal params for memory (vector)
        self.theta = np.zeros(dimensionality + 1)
        self.time_step = 0
        self.history: List[Dict[str, Any]] = []
        self.internal_tension = 0.0
        self.system_health = 1.0
        self.llm = LLMWrapper(provider=llm_provider, model_name=llm_model)

    def _estimate_q_post_and_prior(self, user_input: str) -> Tuple[np.ndarray, np.ndarray]:
        # Placeholder: q_post: distribution derived from new input embedding; p_prior: prior predicted distribution
        # We'll make small vectors so KL makes sense.
        # If embedder available, use it
        if self.prompt_builder.embedder:
            e = self.prompt_builder.embedder.encode(user_input, convert_to_numpy=True)
            # compress to dim
            d = min(len(e), len(self.theta))
            q_post = np.abs(e[:d])
            p_prior = np.abs(self.theta[:d] + 1e-6)
            # normalize
            q_post = q_post / (q_post.sum() + 1e-9)
            p_prior = p_prior / (p_prior.sum() + 1e-9)
            # pad to same size
            if len(q_post) < len(self.theta):
                q_post = np.pad(q_post, (0, len(self.theta)-len(q_post)), 'constant', constant_values=1e-6)
                p_prior = np.pad(p_prior, (0, len(self.theta)-len(p_prior)), 'constant', constant_values=1e-6)
            return q_post, p_prior
        # fallback: simple soft distributions from word counts
        words = user_input.split()
        L = len(self.theta)
        q_post = np.ones(L) * (len(words) + 1)
        p_prior = np.abs(self.theta) + 1.0
        q_post = q_post / q_post.sum()
        p_prior = p_prior / p_prior.sum()
        return q_post, p_prior

    def _compute_error_signal(self, q_post: np.ndarray, p_prior: np.ndarray) -> np.ndarray:
        # error = posterior - prior (vector)
        e = q_post - p_prior
        return e

    def process_input(self, user_input: str) -> Dict[str, Any]:
        # 1) estimate distributions and error
        q_post, p_prior = self._estimate_q_post_and_prior(user_input)
        error = self._compute_error_signal(q_post, p_prior)

        # 2) compute informational tension via MemoryModule logic (KL)
        _, IT = self.memory.step(theta=self.theta, q_post=q_post, p_prior=p_prior, error=error, global_intention=float(self.intent.get_hierarchy()["structural"].mean()), lr=0.0)
        self.internal_tension = IT

        # 3) external influence vector (from user input semantics or random)
        ext_influence = np.abs(np.random.dirichlet(np.ones(self.attractor.dimensionality)))  # placeholder
        # optionally compute influence using prompt builder/embedding
        influence_score = self.attractor.compute_influence_score(ext_influence)

        # 4) update attractors & intents
        new_state = self.attractor.update(ext_influence, self.internal_tension)
        self.intent.update(new_state, ext_influence, self.internal_tension)

        # 5) retrieve memory texts (RAG)
        mem_texts = self.prompt_builder.retrieve_memory_texts(user_input)

        # 6) build prompt & call LLM
        prompt = self.prompt_builder.build_prompt(user_input, memory_context=mem_texts)
        llm_resp = self.llm.chat(prompt)

        # 7) store memory text and update theta via memory.step (with lr)
        self.prompt_builder.add_memory_text(f"Q:{user_input} A:{llm_resp}")
        self.theta, it2 = self.memory.step(theta=self.theta, q_post=q_post, p_prior=p_prior, error=error, global_intention=float(self.intent.get_hierarchy()["structural"].mean()), lr=0.05)
        # it2 should be similar to IT, but updated

        # 8) update body/system health (simple heuristic)
        self.system_health = max(0.1, min(1.0, 1.0 - self.internal_tension * 0.5))

        # 9) save history entry
        entry = {
            "ts": time.time(),
            "step": self.time_step,
            "input": user_input,
            "response": llm_resp,
            "internal_tension": float(self.internal_tension),
            "attractor_state": self.attractor.get_state().tolist(),
            "intent": {k: v.tolist() for k, v in self.intent.get_hierarchy().items()}
        }
        self.history.append(entry)
        self.time_step += 1

        return {
            "response": llm_resp,
            "internal_tension": float(self.internal_tension),
            "attractor_state": self.attractor.get_state().tolist(),
            "intent": {k: v.tolist() for k, v in self.intent.get_hierarchy().items()},
            "system_health": self.system_health
        }

    # Utility methods for persistence
    def save_state(self, path: str):
        data = {
            "theta": self.theta.tolist(),
            "history": self.history,
            "attractor_history": [h.tolist() for h in self.attractor.history]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_state(self, path: str):
        if not os.path.exists(path): return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.theta = np.asarray(data.get("theta", self.theta))
        self.history = data.get("history", self.history)

# -------------------------
# Quick test harness when running file directly
# -------------------------
if __name__ == "__main__":
    print("Starting UDMMCore quick demo (local) ...")
    core = UDMMCore(dimensionality=5, llm_provider=os.environ.get("LLM_PROVIDER","echo"), llm_model=os.environ.get("LLM_MODEL","gpt-4o-mini"))
    samples = [
        "ما هو UDMM؟",
        "اشرح لي نظرية النسبية ببساطة",
        "كيف أتحقق من صحة فكرة علمية؟",
        "هل الجاذب الافتراضي يغير قراراتي؟"
    ]
    for q in samples:
        out = core.process_input(q)
        print(f"> Q: {q}")
        print(f"  -> resp: {out['response'][:400]}")
        print(f"  -> IT: {out['internal_tension']:.4f} state: {out['attractor_state']}")
        print("-"*60)
    core.save_state("udmm_v4_state.json")
    print("Demo finished. State saved to udmm_v4_state.json")
