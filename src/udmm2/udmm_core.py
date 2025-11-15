# udmm_core.py
import os, json, time
from datetime import datetime
from typing import Dict, Any
from .config import EMBEDDING_MODEL, FAISS_INDEX_PATH, FAISS_META_PATH, LLM_PROVIDER, OPENAI_MODEL, LLAMACPP_SERVER, ENV_ALLOW_LEARN
from .memory.faiss_rag import FaissRAG
from .intent.hierarchical import HierarchicalIntentManager
from .llm import OpenAIModel, LlamaCPPModel, EchoModel, GeminiModel

# بسيط Body + Emotion + Episodic storage
class BodyModel:
    def __init__(self):
        self.energy = 1.0
        self.arousal = 0.1
    def apply_action(self, action: Dict[str,Any]):
        cost = action.get("cost", 0.0)
        self.energy = max(0.0, self.energy - cost)
        self.arousal = min(1.0, max(0.0, self.arousal + action.get("arousal_delta", 0.0)))
        return {"energy": self.energy, "arousal": self.arousal}

class EmotionModel:
    def compute_precision_gain(self, body: BodyModel) -> float:
        gain = 1.0 + (body.arousal - 0.2) * 1.2
        return max(0.5, min(2.0, gain))

class EpisodicMemory:
    def __init__(self, path="data/episodic.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.episodes = []
        self._load()
    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.episodes = json.load(f)
        except Exception:
            self.episodes = []
    def add(self, perception, action, result):
        e = {"ts": datetime.utcnow().isoformat(), "perception": perception, "action": action, "result": result}
        self.episodes.append(e)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.episodes, f, ensure_ascii=False, indent=2)
        return e

# +++ NEW: Cognitive Dynamics from "Psychic Attractor Engineering" paper +++
class CognitiveEquilibrium:
    """
    Manages the core dynamic variables (β, κ, η) and cognitive synchronization (C_m)
    inspired by the "Psychic Attractor Engineering" paper.
    """
    def __init__(self):
        # β (beta): Predictive rigidity. High = stable but rigid. Low = flexible but unstable.
        self.beta = 3.0
        # κ (kappa): Embodied-affective coupling. High = grounded, present. Low = detached, disoriented.
        self.kappa = 3.0
        # η (eta): Breadth of the possible world. High = creative, open. Low = closed, stuck.
        self.eta = 3.0
        # C_m: Cognitive synchronization state. Result of the interplay of the vars above.
        self.c_m = 0.7

    def update_state(self, kl_divergence: float, emotional_arousal: float):
        """ Update β, κ, η based on interaction outcomes. """
        # High surprise (KL divergence) softens the model (lower beta)
        self.beta = max(1.0, self.beta - kl_divergence * 0.5)

        # High arousal can degrade body connection if too intense (lower kappa)
        if emotional_arousal > 0.8:
            self.kappa = max(1.0, self.kappa - 0.2)
        else:
            self.kappa = min(5.0, self.kappa + 0.1)

        # Successful synchronization (low surprise) reinforces the current world breadth (eta)
        if kl_divergence < 0.2:
            self.eta = min(5.0, self.eta + 0.1)
        else:
            # Failure opens up possibilities out of necessity (higher eta)
            self.eta = min(5.0, self.eta + kl_divergence * 0.3)

        # Recalculate synchronization C_m
        self._update_synchronization()

    def _update_synchronization(self):
        """ C_m is high when κ is strong and β, η are balanced. """
        balance = 1.0 - abs(self.beta - self.eta) / 4.0 # 1 if equal, less if divergent
        self.c_m = (self.kappa / 5.0) * balance
        self.c_m = max(0.0, min(1.0, self.c_m))

    def get_state(self) -> Dict[str, float]:
        return {
            "beta_rigidity": self.beta,
            "kappa_embodiment": self.kappa,
            "eta_possibility": self.eta,
            "cognitive_sync": self.c_m
        }

    def diagnose_phase(self) -> str:
        """ Implements the UDMM Compass diagnostic tool from the paper. """
        if self.beta > 4.0 and self.eta < 2.0:
            return "صلابة حرجة (CRITICAL_RIGIDITY)"
        elif self.kappa > 4.0 and self.beta < 2.0:
            return "فيضان شعوري (EMOTIONAL_FLOODING)"
        elif self.eta > 4.0 and self.kappa > 4.0:
            return "اختراق إبداعي وشيك (CREATIVE_BREAKTHROUGH_IMMINENT)"
        elif self.c_m > 0.75:
            return "تزامن مرتفع (HIGH_SYNC)"
        else:
            return "مستقر (STABLE)"

# Agent composition
class UDMMAgent:
    def __init__(self):
        # memory
        self.rag = FaissRAG(EMBEDDING_MODEL, index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH)
        # body/emotion/intent
        self.body = BodyModel()
        self.emotion = EmotionModel()
        self.intent_mgr = HierarchicalIntentManager()
        self.episodic = EpisodicMemory()
        self.cognitive_equilibrium = CognitiveEquilibrium() # ++ INTEGRATION
        # llm provider
        self.llm = self._init_llm()

    def _init_llm(self):
        provider = LLM_PROVIDER.lower()
        if provider == "openai":
            return OpenAIModel(OPENAI_MODEL)
        if provider == "llamacpp":
            return LlamaCPPModel(LLAMACPP_SERVER)
        if provider == "gemini":
            return GeminiModel()
        return EchoModel()

    def perceive_and_answer(self, user_text: str) -> Dict[str,Any]:
        # 1. RAG contexts
        contexts = self.rag.query(user_text, top_k=4)
        context_text = "\n".join([f"- {c['meta']['text']} -> {c['meta']['answer']} (score={c['score']:.2f})" for c in contexts]) or "لا يوجد سياق ذاكرة ذي صلة."

        # 2. Get agent's internal state (body, emotion, intent, AND cognitive dynamics)
        attractors = {"survival": 0.2, "knowledge": 0.5, "social": 0.2, "self_maintenance": 0.1}
        global_intent = self.intent_mgr.compute_global_intent(attractors)
        subgoals = self.intent_mgr.decompose(global_intent)
        cognitive_state = self.cognitive_equilibrium.get_state()
        cognitive_phase = self.cognitive_equilibrium.diagnose_phase()

        # 3. Build prompt with the NEW extended state
        prompt = f"""
أنت وكيل معرفي حسب نموذج UDMM الموسع. استجب بالعربية.
---
السياق من الذاكرة:
{context_text}

السؤال: "{user_text}"
---
الحالة الداخلية للوكيل:
- قصد الوكيل: {json.dumps(global_intent)}
- حالة الجسد: energy={self.body.energy:.2f}, arousal={self.body.arousal:.2f}
- **الحالة المعرفية (β, κ, η):** صلابة={cognitive_state['beta_rigidity']:.2f}, تجسيد={cognitive_state['kappa_embodiment']:.2f}, إمكانية={cognitive_state['eta_possibility']:.2f}
- **التشخيص (بوصلة UDMM):** {cognitive_phase}
- **التزامن المعرفي (C_m):** {cognitive_state['cognitive_sync']:.2f}
---
أجب بإيجاز بناءً على حالتك الداخلية. إذا لم تعرف، اطلب التعلّم بصيغة: [TEACH_ASSIST: what is the answer to '{user_text}'?]
"""
        # 4. Dynamically adjust temperature based on cognitive state
        base_temp = 0.5
        eta_effect = (cognitive_state['eta_possibility'] - 3.0) * 0.15 # Higher eta -> more creative
        beta_effect = (cognitive_state['beta_rigidity'] - 3.0) * 0.15 # Higher beta -> more rigid
        temp = base_temp + eta_effect - beta_effect
        temp = max(0.0, min(1.0, temp)) # Clamp temperature

        # 5. Get response and update state
        llm_out = self.llm.chat(prompt, temperature=temp, max_tokens=512)

        # 6. UPDATE COGNITIVE STATE based on interaction outcome
        # Proxy for KL-divergence: low RAG score = high surprise. Failure to answer = very high surprise.
        max_score = 0
        if contexts:
            max_score = max(c['score'] for c in contexts) if contexts else 0
        kl_divergence_proxy = 1.0 - max_score
        if "[TEACH_ASSIST:" in llm_out:
            kl_divergence_proxy = max(kl_divergence_proxy, 0.9) # Major prediction failure

        self.cognitive_equilibrium.update_state(
            kl_divergence=kl_divergence_proxy,
            emotional_arousal=self.body.arousal
        )

        # 7. Record episode and apply energy cost
        ep = self.episodic.add(perception=user_text, action={"type": "respond"}, result={"llm_out": llm_out, "contexts": contexts})
        self.body.apply_action({"cost": 0.01, "arousal_delta": 0.02})

        # Re-fetch state after update for the return value
        cognitive_state = self.cognitive_equilibrium.get_state()
        cognitive_phase = self.cognitive_equilibrium.diagnose_phase()

        return {
            "response": llm_out,
            "contexts": contexts,
            "intent": global_intent,
            "subgoals": subgoals,
            "cognitive_state": cognitive_state,
            "cognitive_phase": cognitive_phase,
            "episode": ep
        }

    def teach(self, question: str, answer: str):
        if not ENV_ALLOW_LEARN:
            return {"error": "learning disabled"}
        self.rag.add(question, answer, source="user")
        return {"status": "learned", "question": question, "answer": answer}
