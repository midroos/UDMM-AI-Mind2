# udmm_core.py
import os, json, time
from datetime import datetime
from typing import Dict, Any, List
from .config import EMBEDDING_MODEL, FAISS_INDEX_PATH, FAISS_META_PATH, LLM_PROVIDER, OPENAI_MODEL, LLAMACPP_SERVER, ENV_ALLOW_LEARN
from .memory.faiss_rag import FaissRAG
from .intent.hierarchical import HierarchicalIntentManager
from .llm import OpenAIModel, LlamaCPPModel, EchoModel, GeminiModel

# udmm_core.py - الإضافات الجديدة
class CognitiveEquilibrium:
    """
    يدمج مفهوم C_m من نظرية UDMM لحساب التوازن المعرفي
    بين التوتر الهدفي (τ) والتوتر الواقعي (γ)
    """

    def __init__(self):
        self.C_m = 0.5  # التوازن المعرفي الأساسي
        self.tau = 0.3  # التوتر الهدفي (الأهداف الداخلية)
        self.gamma = 0.3  # التوتر الواقعي (التكيف مع البيئة)
        self.history = []

    def update_equilibrium(self, goal_alignment: float, reality_tension: float) -> float:
        """
        تحديث التوازن المعرفي بناءً على القوى المتعارضة

        Args:
            goal_alignment (float): محاذاة الأهداف (0-1)
            reality_tension (float): توتر الواقع (0-1)

        Returns:
            float: قيمة C_m المحدثة
        """
        self.tau = goal_alignment
        self.gamma = reality_tension

        # حساب C_m مع تجنب القسمة على الصفر
        if self.tau + self.gamma > 0:
            self.C_m = self.tau / (self.tau + self.gamma)
        else:
            self.C_m = 0.5

        # حفظ التاريخ للمراقبة
        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "C_m": self.C_m,
            "tau": self.tau,
            "gamma": self.gamma
        })

        return self.C_m

    def get_affective_state(self) -> tuple[str, str]:
        """الحالة العاطفية بناءً على C_m"""
        if self.C_m < 0.3:
            return "قلق", "🫤"
        elif self.C_m > 0.7:
            return "متحمس", "🚀"
        else:
            return "متوازن", "😊"

    def get_equilibrium_insights(self) -> Dict[str, Any]:
        """تحليل متعمق للتوازن المعرفي"""
        state, emoji = self.get_affective_state()

        return {
            "cognitive_equilibrium": self.C_m,
            "goal_tension": self.tau,
            "reality_tension": self.gamma,
            "affective_state": state,
            "emoji": emoji,
            "stability": "عالية" if 0.4 <= self.C_m <= 0.6 else "متوسطة" if 0.3 <= self.C_m <= 0.7 else "منخفضة"
        }

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

class EmbodiedBodyModel(BodyModel):
    """
    يمتد BodyModel الأساسي ليشمل الحالات الجسدية للتجسيد
    """

    def __init__(self):
        super().__init__()
        self.physical_state = {
            "posture": "واقف",
            "movement": "هادئ",
            "expression": "محايد",
            "gesture": "لا شيء"
        }
        self.physical_history = []

    def apply_embodied_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        تطبيق فعل مع تحديث الحالة الجسدية بناءً على التوازن المعرفي

        Args:
            action: الفعل المطلوب مع معاملات التجسيد

        Returns:
            الحالة الجسدية المحدثة
        """
        # تطبيق الفعل الأساسي على الجسد
        body_result = self.apply_action(action)

        # تحديث الحالة الجسدية بناءً على C_m
        C_m = action.get("C_m", 0.5)
        self._update_physical_embodiment(C_m)

        # دمج النتائج
        embodied_result = {**body_result, **self.physical_state}

        # حفظ التاريخ
        self.physical_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            **embodied_result
        })

        return embodied_result

    def _update_physical_embodiment(self, C_m: float):
        """تحديث التجسيد الجسدي بناءً على التوازن المعرفي"""
        if C_m < 0.3:
            # حالة قلق - توتر جسدي
            self.physical_state = {
                "posture": "منقبض",
                "movement": "متوتر",
                "expression": "قلق",
                "gesture": "فرك اليدين"
            }
        elif C_m > 0.7:
            # حالة حماس - انفتاح جسدي
            self.physical_state = {
                "posture": "منفتح",
                "movement": "سريع",
                "expression": "متحمس",
                "gesture": "إيماءات واسعة"
            }
        else:
            # حالة توازن - انسجام جسدي
            self.physical_state = {
                "posture": "مستقيم",
                "movement": "انسجام",
                "expression": "هادئ",
                "gesture": "إيماءات طبيعية"
            }

class EmbodimentSystem:
    """
    نظام مرن للتجسيد يدعم وسائط متعددة للتفاعل
    """

    def __init__(self):
        self.embodiment_modes = {
            "text": TextEmbodiment(),
            "avatar": AvatarEmbodiment(),
            "hardware": HardwareEmbodiment(),
            "reverse": ReverseEmbodiment()
        }
        self.current_mode = "text"
        self.embodiment_history = []

    def set_embodiment_mode(self, mode: str):
        """تغيير نمط التجسيد"""
        if mode in self.embodiment_modes:
            self.current_mode = mode
            return f"✅ تم تغيير نمط التجسيد إلى: {mode}"
        else:
            return f"❌ نمط التجسيد غير مدعوم: {mode}"

    def generate_embodied_response(self,
                                text_response: str,
                                cognitive_equilibrium: CognitiveEquilibrium,
                                physical_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        توليد استجابة مجسدة بناءً على الحالة المعرفية والجسدية

        Args:
            text_response: الرد النصي الأساسي
            cognitive_equilibrium: حالة التوازن المعرفي
            physical_state: الحالة الجسدية الحالية

        Returns:
            الاستجابة المجسدة الكاملة
        """
        embodiment = self.embodiment_modes[self.current_mode]

        embodied_output = embodiment.express(
            text=text_response,
            cognitive_equilibrium=cognitive_equilibrium,
            physical_state=physical_state
        )

        # حفظ تاريخ التجسيد
        self.embodiment_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "mode": self.current_mode,
            "output": embodied_output
        })

        return embodied_output

    def get_available_modes(self) -> List[str]:
        """الحصول على أنماط التجسيد المتاحة"""
        return list(self.embodiment_modes.keys())


class TextEmbodiment:
    """التجسيد النصي - إضافة مؤشرات عاطفية وحركية"""

    def express(self, text: str, cognitive_equilibrium: CognitiveEquilibrium, physical_state: Dict[str, Any]) -> Dict[str, Any]:
        equilibrium_insights = cognitive_equilibrium.get_equilibrium_insights()

        embodied_text = f"{equilibrium_insights['emoji']} *{physical_state['movement']}* {text}"
        embodied_text += f"\n[الحالة: {equilibrium_insights['affective_state']} - C_m: {equilibrium_insights['cognitive_equilibrium']:.2f}]"
        embodied_text += f"\n[وضعية: {physical_state['posture']} - تعبير: {physical_state['expression']}]"

        return {
            "type": "text",
            "content": embodied_text,
            "cognitive_state": equilibrium_insights,
            "physical_state": physical_state,
            "timestamp": datetime.utcnow().isoformat()
        }


class AvatarEmbodiment:
    """التجسيد بالافتار - أوامر للحركات والتعبيرات"""

    def express(self, text: str, cognitive_equilibrium: CognitiveEquilibrium, physical_state: Dict[str, Any]) -> Dict[str, Any]:
        equilibrium_insights = cognitive_equilibrium.get_equilibrium_insights()

        return {
            "type": "avatar",
            "text_content": text,
            "animation_commands": {
                "animation_type": physical_state["movement"],
                "facial_expression": physical_state["expression"],
                "posture": physical_state["posture"],
                "gesture": physical_state["gesture"]
            },
            "voice_parameters": {
                "tone": self._get_voice_tone(equilibrium_insights["cognitive_equilibrium"]),
                "speed": self._get_speech_speed(equilibrium_insights["cognitive_equilibrium"]),
                "volume": "طبيعي"
            },
            "cognitive_state": equilibrium_insights
        }

    def _get_voice_tone(self, C_m: float) -> str:
        if C_m < 0.3:
            return "هادئ ومتزن"
        elif C_m > 0.7:
            return "حيوي ومتحمس"
        else:
            return "واضح وطبيعي"

    def _get_speech_speed(self, C_m: float) -> str:
        if C_m < 0.3:
            return "بطيء"
        elif C_m > 0.7:
            return "سريع"
        else:
            return "معتدل"


class HardwareEmbodiment:
    """التجسيد المادي - تفاعل مع الهاردوير"""

    def express(self, text: str, cognitive_equilibrium: CognitiveEquilibrium, physical_state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "hardware",
            "text_content": text,
            "hardware_commands": {
                "led_color": self._get_led_color(cognitive_equilibrium.C_m),
                "motor_movement": physical_state["movement"],
                "display_expression": physical_state["expression"]
            },
            "sensor_readings": {
                "energy_level": physical_state.get("energy", 1.0),
                "arousal_level": physical_state.get("arousal", 0.1)
            }
        }

    def _get_led_color(self, C_m: float) -> str:
        if C_m < 0.3:
            return "أزرق"  # هادئ
        elif C_m > 0.7:
            return "أحمر"  # نشيط
        else:
            return "أخضر"  # متوازن


class ReverseEmbodiment:
    """التجسيد العكسي - تحليل النص إلى حالات مجسدة"""

    def express(self, text: str, cognitive_equilibrium: CognitiveEquilibrium, physical_state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "reverse_embodiment",
            "original_text": text,
            "embodiment_analysis": {
                "detected_emotion": self._analyze_emotion(text),
                "suggested_posture": self._suggest_posture(text),
                "recommended_expression": self._recommend_expression(text),
                "movement_pattern": self._detect_movement_pattern(text)
            },
            "cognitive_interpretation": cognitive_equilibrium.get_equilibrium_insights()
        }

    def _analyze_emotion(self, text: str) -> str:
        emotions = {
            "قلق": ["خائف", "قلق", "متوتر", "خوف"],
            "فرح": ["سعيد", "فرح", "مبتهج", "مسرور"],
            "حزن": ["حزين", "مكتئب", "بائس", "تعيس"]
        }

        for emotion, keywords in emotions.items():
            if any(keyword in text for keyword in keywords):
                return emotion
        return "محايد"

    def _suggest_posture(self, text: str) -> str:
        return "not implemented"

    def _recommend_expression(self, text: str) -> str:
        return "not implemented"

    def _detect_movement_pattern(self, text: str) -> str:
        return "not implemented"

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
    """
    وكيل UDMM مجسد يدمج التوازن المعرفي والتجسيد الجسدي
    لإنتاج ردود ديناميكية تستجيب للحالة الداخلية
    """

    def __init__(self):
        super().__init__()

        # مكونات التجسيد الجديدة
        self.cognitive_equilibrium = CognitiveEquilibrium()
        self.embodied_body = EmbodiedBodyModel()
        self.embodiment_system = EmbodimentSystem()

        # إحصائيات التجسيد
        self.embodiment_stats = {
            "total_interactions": 0,
            "mode_usage": {"text": 0, "avatar": 0, "hardware": 0, "reverse": 0},
            "affective_states": {"قلق": 0, "متوازن": 0, "متحمس": 0}
        }

    def perceive_and_act(self, user_input: str, input_type: str = "text") -> Dict[str, Any]:
        """
        التفاعل الرئيسي مع التجسيد الكامل

        Args:
            user_input: مدخلات المستخدم
            input_type: نوع المدخلات (نص، صوت، إلخ)

        Returns:
            الاستجابة المجسدة الكاملة
        """
        self.embodiment_stats["total_interactions"] += 1

        # 1. المعالجة الأساسية من UDMMAgent
        basic_response = self.perceive_and_answer(user_input)

        # 2. حساب التوازن المعرفي
        goal_alignment = self._calculate_goal_alignment(user_input, basic_response["intent"])
        reality_tension = self._calculate_reality_tension(user_input, basic_response["contexts"])

        C_m = self.cognitive_equilibrium.update_equilibrium(goal_alignment, reality_tension)

        # 3. تحديث التجسيد الجسدي
        physical_action = {
            "cost": 0.01,
            "arousal_delta": 0.02,
            "C_m": C_m
        }
        body_state = self.embodied_body.apply_embodied_action(physical_action)

        # 4. توليد الاستجابة المجسدة
        embodied_response = self.embodiment_system.generate_embodied_response(
            text_response=basic_response["response"],
            cognitive_equilibrium=self.cognitive_equilibrium,
            physical_state=body_state
        )

        # 5. تحديث الإحصائيات
        self._update_embodiment_stats()

        return {
            **basic_response,
            "embodied_response": embodied_response,
            "cognitive_equilibrium": self.cognitive_equilibrium.get_equilibrium_insights(),
            "physical_state": body_state,
            "embodiment_mode": self.embodiment_system.current_mode,
            "interaction_id": self.embodiment_stats["total_interactions"]
        }

    def _calculate_goal_alignment(self, user_input: str, intent: Dict) -> float:
        """حساب τ - التوافق مع الأهداف الداخلية"""
        alignment_score = 0.1  # درجة أساسية

        # كلمات مفتاحية تشير إلى تحقيق الأهداف
        goal_keywords = ["تعلم", "معرفة", "تطوير", "نمو", "فهم", "شرح", "مساعدة"]
        for keyword in goal_keywords:
            if keyword in user_input:
                alignment_score += 0.15

        # تحسين المحاذاة بناءً على النية
        if intent.get("confidence", 0) > 0.7:
            alignment_score += 0.2

        return min(alignment_score, 1.0)

    def _calculate_reality_tension(self, user_input: str, contexts: List) -> float:
        """حساب γ - التوتر مع الواقع"""
        tension_score = 0.1  # درجة أساسية

        # زيادة التوتر عند عدم وجود سياق جيد
        if not contexts or contexts[0]["score"] < 0.7:
            tension_score += 0.3

        # كلمات مفتاحية تشير إلى القلق أو المشاكل
        anxiety_keywords = ["مشكلة", "خطأ", "صعب", "لا أعرف", "قلق", "خوف", "محتار"]
        for keyword in anxiety_keywords:
            if keyword in user_input:
                tension_score += 0.1

        # زيادة التوتر عند وجود أخطاء حديثة
        if self.embodied_body.energy < 0.3:
            tension_score += 0.2

        return min(tension_score, 1.0)

    def _update_embodiment_stats(self):
        """تحديث إحصائيات التجسيد"""
        # تحديث استخدام الأنماط
        current_mode = self.embodiment_system.current_mode
        self.embodiment_stats["mode_usage"][current_mode] += 1

        # تحديث الحالات العاطفية
        affective_state = self.cognitive_equilibrium.get_affective_state()[0]
        self.embodiment_stats["affective_states"][affective_state] += 1

    def set_embodiment_mode(self, mode: str) -> str:
        """تغيير نمط التجسيد"""
        result = self.embodiment_system.set_embodiment_mode(mode)
        return result

    def get_embodiment_report(self) -> Dict[str, Any]:
        """تقرير مفصل عن أداء التجسيد"""
        total = self.embodiment_stats["total_interactions"]

        return {
            "embodiment_performance": {
                "total_interactions": total,
                "mode_distribution": {
                    mode: count for mode, count in self.embodiment_stats["mode_usage"].items()
                },
                "affective_distribution": {
                    state: count for state, count in self.embodiment_stats["affective_states"].items()
                },
                "current_cognitive_equilibrium": self.cognitive_equilibrium.get_equilibrium_insights(),
                "available_modes": self.embodiment_system.get_available_modes()
            }
        }


# مثال استخدام الوكيل المجسد
def demonstrate_embodied_agent():
    # إنشاء الوكيل المجسد
    embodied_agent = EmbodiedUDMMAgent()

    print("🧠 بدء عرض وكيل UDMM المجسد:\n")

    # اختبار تفاعلات مختلفة
    test_inputs = [
        "أحتاج مساعدة في فهم هذا الموضوع",
        "أشعر بالقلق من المستقبل",
        "هذا مثير للاهتمام! أريد تعلم المزيد",
        "لدي مشكلة في فهم الكود"
    ]

    for i, user_input in enumerate(test_inputs, 1):
        print(f"التفاعل {i}:")
        print(f"المستخدم: '{user_input}'")

        response = embodied_agent.perceive_and_act(user_input)
        embodied_output = response["embodied_response"]

        if embodied_output["type"] == "text":
            print(f"الوكيل: {embodied_output['content']}")
        else:
            print(f"الوكيل [نمط {embodied_output['type']}]: {embodied_output}")

        print(f"التوازن المعرفي: C_m = {response['cognitive_equilibrium']['cognitive_equilibrium']:.2f}")
        print("-" * 60)

    # عرض التقرير النهائي
    report = embodied_agent.get_embodiment_report()
    print("\n📊 تقرير أداء التجسيد:")
    print(f"إجمالي التفاعلات: {report['embodiment_performance']['total_interactions']}")
    print(f"توزيع الأنماط: {report['embodiment_performance']['mode_distribution']}")

# تشغيل العرض
if __name__ == "__main__":
    demonstrate_embodied_agent()
