# udmm_core.py
import os, json, time
from datetime import datetime
from typing import Dict, Any, List
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

class CognitiveEquilibrium:
    def __init__(self):
        self.C_m = 0.5  # التوازن المعرفي
        self.tau = 0.3  # التوتر الهدفي
        self.gamma = 0.3  # التوتر الواقعي

    def update_equilibrium(self, goal_alignment, reality_tension):
        """تحديث C_m بناءً على τ و γ"""
        self.tau = goal_alignment
        self.gamma = reality_tension

        if self.tau + self.gamma > 0:
            self.C_m = self.tau / (self.tau + self.gamma)
        else:
            self.C_m = 0.5

        return self.C_m

    def get_affective_state(self):
        """الحالة العاطفية بناءً على C_m"""
        if self.C_m < 0.3:
            return "قلق", "🫤"
        elif self.C_m > 0.7:
            return "متحمس", "🚀"
        else:
            return "متوازن", "😊"

class EmbodiedBodyModel(BodyModel):
    def __init__(self):
        super().__init__()
        self.physical_state = {
            "posture": "واقف",  # وضعية الجسم
            "movement": "هادئ",  # نوع الحركة
            "expression": "محايد"  # التعبير الوجهي
        }

    def apply_embodied_action(self, action: Dict[str, Any]):
        # تحديث الحالة الجسدية بناءً على الفعل
        result = self.apply_action(action)

        # تأثير التوازن المعرفي على التجسيد
        C_m = action.get("C_m", 0.5)

        if C_m < 0.3:
            self.physical_state = {"posture": "منقبض", "movement": "متوتر", "expression": "قلق"}
        elif C_m > 0.7:
            self.physical_state = {"posture": "منفتح", "movement": "سريع", "expression": "متحمس"}
        else:
            self.physical_state = {"posture": "مستقيم", "movement": "انسجام", "expression": "هادئ"}

        return {**result, **self.physical_state}

class EmbodimentSystem:
    def __init__(self):
        self.embodiment_modes = {
            "text": TextEmbodiment(),
            "avatar": AvatarEmbodiment(),
            "hardware": HardwareEmbodiment(),
            "reverse": ReverseEmbodiment()
        }
        self.current_mode = "text"

    def generate_embodied_response(self, text_response, cognitive_state, physical_state):
        """توليد استجابة مجسدة بناءً على الحالة"""
        embodiment = self.embodiment_modes[self.current_mode]

        return embodiment.express(
            text=text_response,
            cognitive_state=cognitive_state,
            physical_state=physical_state
        )

class TextEmbodiment:
    def express(self, text, cognitive_state, physical_state):
        """التجسيد النصي - إضافة مؤشرات عاطفية"""
        state, emoji = cognitive_state.get_affective_state()

        embodied_text = f"{emoji} *{physical_state['movement']}* {text}"
        embodied_text += f"\n[الحالة: {state} - C_m: {cognitive_state.C_m:.2f}]"

        return embodied_text

class AvatarEmbodiment:
    def express(self, text, cognitive_state, physical_state):
        """التجسيد بالافتار - إرجاع أوامر حركة"""
        return {
            "text": text,
            "animation": physical_state["movement"],
            "expression": physical_state["expression"],
            "voice_tone": "قلق" if cognitive_state.C_m < 0.3 else "متحمس" if cognitive_state.C_m > 0.7 else "طبيعي"
        }

class HardwareEmbodiment:
    def express(self, text, cognitive_state, physical_state):
        return {"error": "not implemented"}

class ReverseEmbodiment:
    def express(self, text, cognitive_state, physical_state):
        return {"error": "not implemented"}

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
        if provider == "gemini":
            return GeminiModel()
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

class EmbodiedUDMMAgent(UDMMAgent):
    def __init__(self):
        super().__init__()
        self.cognitive_eq = CognitiveEquilibrium()
        self.embodied_body = EmbodiedBodyModel()
        self.embodiment_system = EmbodimentSystem()

    def perceive_and_act(self, user_input: str, input_type: str = "text") -> Dict[str, Any]:
        # 1. المعالجة الأساسية
        basic_response = self.perceive_and_answer(user_input)

        # 2. حساب التوازن المعرفي
        goal_alignment = self._calculate_goal_alignment(user_input, basic_response["intent"])
        reality_tension = self._calculate_reality_tension(user_input, basic_response["contexts"])

        C_m = self.cognitive_eq.update_equilibrium(goal_alignment, reality_tension)

        # 3. تطبيق التجسيد الجسدي
        physical_action = {
            "cost": 0.01,
            "arousal_delta": 0.02,
            "C_m": C_m
        }
        body_state = self.embodied_body.apply_embodied_action(physical_action)

        # 4. توليد الاستجابة المجسدة
        embodied_response = self.embodiment_system.generate_embodied_response(
            text_response=basic_response["response"],
            cognitive_state=self.cognitive_eq,
            physical_state=body_state
        )

        return {
            **basic_response,
            "embodied_response": embodied_response,
            "cognitive_equilibrium": {
                "C_m": C_m,
                "tau": self.cognitive_eq.tau,
                "gamma": self.cognitive_eq.gamma
            },
            "physical_state": body_state
        }

    def _calculate_goal_alignment(self, user_input: str, intent: Dict) -> float:
        """حساب τ - التوافق مع الأهداف الداخلية"""
        # تحليل مدى توافق المدخلات مع أهداف الوكيل
        goal_keywords = ["تعلم", "معرفة", "تطوير", "نمو", "فهم"]
        alignment_score = 0.1

        for keyword in goal_keywords:
            if keyword in user_input:
                alignment_score += 0.2

        return min(alignment_score, 1.0)

    def _calculate_reality_tension(self, user_input: str, contexts: List) -> float:
        """حساب γ - التوتر مع الواقع"""
        # تحمد مدى توافق النتائج مع التوقعات
        tension_score = 0.1

        if not contexts or contexts[0]["score"] < 0.7:
            tension_score += 0.3  # توتر عند عدم وجود إجابات جيدة

        anxiety_words = ["مشكلة", "خطأ", "صعب", "لا أعرف"]
        for word in anxiety_words:
            if word in user_input:
                tension_score += 0.2

        return min(tension_score, 1.0)

# اختبار الوكيل المجسد
if __name__ == '__main__':
    agent = EmbodiedUDMMAgent()

    # تفاعل مع تجسيد كامل
    response = agent.perceive_and_act("أشعر بالقلق من المستقبل")
    print(response["embodied_response"])

    # تغيير نمط التجسيد
    agent.embodiment_system.current_mode = "avatar"
    avatar_response = agent.perceive_and_act("أريد تعلم شيء جديد")
    print(avatar_response)
