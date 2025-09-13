# udmm_core.py
import os, json, time
from datetime import datetime
from typing import Dict, Any
from .config import EMBEDDING_MODEL, FAISS_INDEX_PATH, FAISS_META_PATH, LLM_PROVIDER, OPENAI_MODEL, LLAMACPP_SERVER, ENV_ALLOW_LEARN
from .memory.faiss_rag import FaissRAG
from .intent.hierarchical import HierarchicalIntentManager
from .llm import OpenAIModel, LlamaCPPModel, EchoModel

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
        # llm provider
        self.llm = self._init_llm()

    def _init_llm(self):
        provider = LLM_PROVIDER.lower()
        if provider == "openai":
            return OpenAIModel(OPENAI_MODEL)
        if provider == "llamacpp":
            return LlamaCPPModel(LLAMACPP_SERVER)
        return EchoModel()

    def perceive_and_answer(self, user_text: str) -> Dict[str,Any]:
        # 1. RAG contexts
        contexts = self.rag.query(user_text, top_k=4)
        context_text = "\n".join([f"- {c['meta']['text']} -> {c['meta']['answer']} (score={c['score']:.2f})" for c in contexts]) or "لا يوجد سياق ذاكرة ذي صلة."
        # 2. intent & emotion
        attractors = {"survival": 0.2, "knowledge": 0.5, "social": 0.2, "self_maintenance": 0.1}
        global_intent = self.intent_mgr.compute_global_intent(attractors)
        subgoals = self.intent_mgr.decompose(global_intent)
        precision_gain = self.emotion.compute_precision_gain(self.body)
        # 3. build prompt (Arabic)
        prompt = f"""
أنت وكيل معرفي حسب نموذج UDMM. استجب بالعربية.
السياق من الذاكرة:
{context_text}

السؤال: "{user_text}"

قصد الوكيل: {json.dumps(global_intent)}
أهداف فرعية: {json.dumps(subgoals)}
حالة الجسد: energy={self.body.energy:.2f}, arousal={self.body.arousal:.2f}
دقة عاطفية: {precision_gain:.2f}

أجب بإيجاز، واذا لم تعرف فاطلب التعلّم بصيغة: [TEACH_ASSIST: what is the answer to '{user_text}'?]
"""
        # temperature inversely proportional to precision
        temp = max(0.0, 0.8 - (precision_gain - 1.0) * 0.4)
        llm_out = self.llm.chat(prompt, temperature=temp, max_tokens=512)
        # record episode
        ep = self.episodic.add(perception=user_text, action={"type": "respond"}, result={"llm_out": llm_out, "contexts": contexts})
        # small energy cost
        self.body.apply_action({"cost": 0.01, "arousal_delta": 0.02})
        return {"response": llm_out, "contexts": contexts, "intent": global_intent, "subgoals": subgoals, "precision_gain": precision_gain, "episode": ep}

    def teach(self, question: str, answer: str):
        if not ENV_ALLOW_LEARN:
            return {"error": "learning disabled"}
        self.rag.add(question, answer, source="user")
        return {"status": "learned", "question": question, "answer": answer}
