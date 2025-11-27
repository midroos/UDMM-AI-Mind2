import time
import uuid
from collections import defaultdict, deque
import random
import math
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import json

# ============================
# 1. DIGITAL BODY (الجسد الرقمي المتطور)
# ============================

class DigitalBody:
    def __init__(self):
        self.energy = 1.0
        self.arousal = 0.3
        self.fatigue_rate = 0.02
        self.homeostatic_drift = 0.01
        self.sensory_input = 0.0
        self.motor_output = 0.0

    def update_body_state(self, sensory_input=0.0):
        self.sensory_input = sensory_input
        # تأثير المدخلات الحسية على الإثارة
        self.arousal = max(0.05, min(1.0,
            self.arousal + random.uniform(-0.03, 0.03) + sensory_input * 0.1))

        # انخفاض الطاقة مع الإثارة العالية
        energy_drain = self.fatigue_rate * (1 + self.arousal * 0.5)
        self.energy = max(0.1, self.energy - energy_drain + self.homeostatic_drift)

    def stabilize(self):
        """Re-Anchoring مع تأثير زمني"""
        self.arousal *= 0.6
        self.energy = min(1.0, self.energy + 0.1)

    def get_body_state_vector(self) -> List[float]:
        return [self.energy, self.arousal, self.sensory_input]

# =====================================
# 2. RAW AFFECT (الوجدان الخام المتطور)
# =====================================

class AffectiveSystem:
    def __init__(self):
        self.A_r = 0.2
        self.affective_history = deque(maxlen=50)
        self.decay_rate = 0.05
        self.affective_priming = defaultdict(float)
        self.valence = 0.0  # إيجابي/سلبي

    def update_affect(self, stimulus_strength: float, valence: float):
        """تحديث الوجدان مع التكافؤ"""
        self.A_r = min(1.0, max(0.0, self.A_r + stimulus_strength))
        self.valence = valence
        self.affective_history.append((self.A_r, self.valence, time.time()))

    def decrease(self, amount: float):
        """Decrease the raw affect value."""
        self.A_r = max(0.0, self.A_r - amount)

    def natural_decay(self):
        self.A_r *= (1 - self.decay_rate)
        self.valence *= 0.9  # تضاؤل التكافؤ

    def get_affective_trajectory(self) -> List[float]:
        return [affect[0] for affect in list(self.affective_history)]

    def compute_affective_variance(self) -> float:
        if len(self.affective_history) < 2:
            return 0.0
        affects = [a[0] for a in self.affective_history]
        return np.var(affects)

# =========================================
# 3. SYMBOLIC SYSTEM Σ المتطور
# =========================================

class SymbolicStore:
    def __init__(self):
        self.symbols = defaultdict(lambda: {
            "affect": 0.1,
            "frequency": 0,
            "last_accessed": time.time(),
            "semantic_links": defaultdict(float)
        })
        self.symbol_network = defaultdict(dict)

    def reinforce(self, symbol: str, value: float, context: str = ""):
        self.symbols[symbol]["affect"] = min(1.0, self.symbols[symbol]["affect"] + value)
        self.symbols[symbol]["frequency"] += 1
        self.symbols[symbol]["last_accessed"] = time.time()

        # إنشاء روابط دلالية
        if context:
            self.symbols[symbol]["semantic_links"][context] += value

    def weaken(self, symbol: str, value: float):
        self.symbols[symbol]["affect"] = max(0.0, self.symbols[symbol]["affect"] - value)

    def rename(self, old: str, new: str):
        """إعادة التسمية مع نقل جميع الخصائص"""
        if old in self.symbols:
            self.symbols[new] = self.symbols.pop(old)

    def get_activation_level(self, symbol: str) -> float:
        base = self.symbols[symbol]["affect"]
        recency = 1.0 / (time.time() - self.symbols[symbol]["last_accessed"] + 1)
        return base * 0.7 + recency * 0.3

    def find_related_symbols(self, symbol: str, threshold: float = 0.3) -> List[str]:
        related = []
        if symbol in self.symbols:
            for linked_symbol, strength in self.symbols[symbol]["semantic_links"].items():
                if strength > threshold:
                    related.append(linked_symbol)
        return related

# =======================================
# 4. MEMORY SYSTEM المتطور
# =======================================

class MemorySystem:
    def __init__(self):
        self.episodic = deque(maxlen=500)
        self.semantic = defaultdict(lambda: {"value": "", "confidence": 0.5, "sources": []})
        self.working_memory = deque(maxlen=7)  # سعة الذاكرة العاملة
        self.retrieval_threshold = 0.3

    def add_episode(self, user_msg: str, agent_msg: str, state: Dict, affective_state: float):
        episode = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "user": user_msg,
            "agent": agent_msg,
            "state": state,
            "affect": affective_state,
            "salience": self._compute_salience(user_msg, affective_state)
        }
        self.episodic.append(episode)
        self.working_memory.append(episode)

    def add_knowledge(self, key: str, value: str, confidence: float = 0.5, source: str = ""):
        self.semantic[key] = {
            "value": value,
            "confidence": confidence,
            "sources": [source] if source else []
        }

    def retrieve_contextual(self, query: str, affective_context: float, recency_weight: float = 0.3) -> Dict:
        results = {
            "episodic": [],
            "semantic": {},
            "working_memory": list(self.working_memory)
        }

        # البحث في الذاكرة العرضية مع مراعاة السياق الوجداني
        for episode in self.episodic:
            relevance = self._compute_relevance(episode, query, affective_context)
            if relevance > self.retrieval_threshold:
                results["episodic"].append({
                    "episode": episode,
                    "relevance": relevance
                })

        # البحث في الذاكرة الدلالية
        for key, data in self.semantic.items():
            if query in key or query in data["value"]:
                results["semantic"][key] = data

        return results

    def _compute_salience(self, message: str, affect: float) -> float:
        length_factor = min(len(message) / 100, 1.0)
        emotional_factor = affect
        return (length_factor + emotional_factor) / 2

    def _compute_relevance(self, episode: Dict, query: str, affective_context: float) -> float:
        text_similarity = 1.0 if query in episode["user"] else 0.3
        affective_similarity = 1.0 - abs(episode["affect"] - affective_context)
        recency = 1.0 / (time.time() - episode["timestamp"] + 1)

        return (text_similarity * 0.4 + affective_similarity * 0.4 + recency * 0.2)

# ================================================
# 5. VIRTUAL ATTRACTOR (الجاذب الافتراضي)
# ================================================

class VirtualAttractor:
    def __init__(self):
        self.attractor_states = {
            "stable": {"energy": 0.8, "arousal": 0.3, "affect": 0.4},
            "transition": {"energy": 0.5, "arousal": 0.6, "affect": 0.7},
            "chaotic": {"energy": 0.3, "arousal": 0.9, "affect": 0.9}
        }
        self.current_basin = "stable"
        self.attractor_strength = 0.7

    def compute_attractor_force(self, current_state: Dict) -> Tuple[str, float]:
        """حساب أقرب جاذب وقوة الجذب"""
        min_distance = float('inf')
        closest_attractor = "stable"

        for attractor, state in self.attractor_states.items():
            distance = self._state_distance(current_state, state)
            if distance < min_distance:
                min_distance = distance
                closest_attractor = attractor

        attraction_force = self.attractor_strength / (min_distance + 0.1)
        return closest_attractor, attraction_force

    def _state_distance(self, state1: Dict, state2: Dict) -> float:
        keys = ["energy", "arousal", "affect"]
        squared_diff = sum((state1.get(k, 0) - state2.get(k, 0)) ** 2 for k in keys)
        return math.sqrt(squared_diff)

    def update_attractor_strength(self, prediction_error: float):
        """تحديث قوة الجاذب بناءً على خطأ التنبؤ"""
        self.attractor_strength = max(0.3, min(0.9,
            self.attractor_strength + prediction_error * 0.1))

# ================================================
# 6. META-AGENT المتطور + AAR الكامل
# ================================================

class MetaAgent:
    def __init__(self, body, affect, symbols, memory, attractor):
        self.body = body
        self.affect = affect
        self.symbols = symbols
        self.memory = memory
        self.attractor = attractor

        self.C = 0.5  # Presence of consciousness
        self.IT = 0.0  # Informational Tension
        self.KL_divergence = 0.0
        self.temporal_depth = 3

        self.aar_history = []

    def compute_prediction_error(self, expectation: str, reality: str, context_affect: float) -> float:
        """حساب خطأ التنبؤ مع السياق الوجداني"""
        semantic_pe = abs(len(reality) - len(expectation)) / max(1, len(expectation))
        affective_pe = abs(context_affect - self.affect.A_r)

        pe = (semantic_pe * 0.6 + affective_pe * 0.4)
        self.IT = pe

        # تحديث الجاذب
        self.attractor.update_attractor_strength(pe)

        self.affect.update_affect(pe * 0.3, valence=0.0)  # خطأ التنبؤ سلبي التكافؤ

        return pe

    def compute_KL_divergence(self, prior_state: Dict, current_state: Dict) -> float:
        """حساب تباعد كولباك-ليبلر بين الحالات"""
        try:
            # استخراج القيم المتداخلة بشكل صحيح
            prior_vec = [
                prior_state.get("body", [0.5, 0.5, 0.0])[0],  # energy
                prior_state.get("body", [0.5, 0.5, 0.0])[1],  # arousal
                prior_state.get("affect", {}).get("A_r", 0.5) # affect A_r
            ]
            current_vec = [
                current_state.get("body", [0.5, 0.5, 0.0])[0], # energy
                current_state.get("body", [0.5, 0.5, 0.0])[1], # arousal
                current_state.get("affect", {}).get("A_r", 0.5) # affect A_r
            ]

            # تطبيع
            prior_sum = sum(prior_vec)
            current_sum = sum(current_vec)

            if prior_sum == 0 or current_sum == 0:
                return 0.0

            prior_norm = [p/prior_sum for p in prior_vec]
            current_norm = [c/current_sum for c in current_vec]

            # حساب KL divergence
            kl = 0.0
            for i in range(len(prior_norm)):
                if prior_norm[i] > 0 and current_norm[i] > 0:
                    kl += prior_norm[i] * math.log(prior_norm[i] / current_norm[i])

            self.KL_divergence = max(0.0, kl)
            return self.KL_divergence

        except (ZeroDivisionError, ValueError) as e:
            # Handle specific math errors, but log them for debugging
            print(f"Debug: Handled KL-divergence calculation error: {e}")
            return 0.0

    # -------------------------
    # AAR-1: إعادة التسمية المتقدمة
    # -------------------------
    def relabel(self, old_symbol: str, new_symbol: str, confidence: float = 0.7):
        if old_symbol in self.symbols.symbols:
            # نقل جميع الخصائص والروابط
            self.symbols.rename(old_symbol, new_symbol)

            # تحديث الذاكرة الدلالية
            self.memory.add_knowledge(
                f"relabel_{old_symbol}_to_{new_symbol}",
                f"إعادة تسمية من {old_symbol} إلى {new_symbol}",
                confidence,
                "AAR"
            )

            self.affect.decrease(0.15)
            self.aar_history.append({
                "type": "relabel",
                "old": old_symbol,
                "new": new_symbol,
                "timestamp": time.time()
            })

    # -------------------------
    # AAR-2: إعادة الربط الجسدي المتقدم
    # -------------------------
    def reanchor(self, intensity: float = 0.8):
        self.body.stabilize()

        # إعادة ضبط الوجدان
        self.affect.A_r *= (1 - intensity * 0.5)
        self.affect.valence *= 0.5

        # تحديث الجاذب
        current_state = self._get_current_state()
        self.attractor.current_basin, _ = self.attractor.compute_attractor_force(current_state)

        self.aar_history.append({
            "type": "reanchor",
            "intensity": intensity,
            "timestamp": time.time()
        })

    # -------------------------
    # AAR-3: إعادة توزيع الوجدان المتقدم
    # -------------------------
    def redistribute_affect(self, redistribution_factor: float = 0.8):
        # إعادة توزيع الوجدان عبر الرموز
        total_affect = sum(data["affect"] for data in self.symbols.symbols.values())
        num_symbols = max(1, len(self.symbols.symbols))
        target_affect = total_affect / num_symbols

        for symbol, data in self.symbols.symbols.items():
            current_affect = data["affect"]
            new_affect = current_affect * (1 - redistribution_factor) + target_affect * redistribution_factor
            self.symbols.symbols[symbol]["affect"] = new_affect

        # إعادة ضبط الوجدان الخام
        self.affect.A_r *= 0.6
        affective_variance = self.affect.compute_affective_variance()

        self.aar_history.append({
            "type": "redistribute_affect",
            "factor": redistribution_factor,
            "variance_reduction": affective_variance,
            "timestamp": time.time()
        })

    def _get_current_state(self) -> Dict:
        return {
            "energy": self.body.energy,
            "arousal": self.body.arousal,
            "affect": self.affect.A_r
        }

    def intervene(self, current_state: Dict, prediction_error: float):
        """تدخل الوعي المعزز"""
        intervention_triggered = False

        # قاعدة: إذا كان خطأ التنبؤ مرتفعًا
        if prediction_error > 0.6:
            self.redistribute_affect(0.7)
            intervention_triggered = True

        # قاعدة: إذا كان التوتر المعلوماتي مرتفعًا
        if self.IT > 0.7:
            self.reanchor(0.9)
            intervention_triggered = True

        # قاعدة: إذا كان التباعد KL مرتفعًا
        if self.KL_divergence > 0.5:
            # البحث عن رموز لإعادة التسمية
            highly_charged = [(s, d) for s, d in self.symbols.symbols.items() if d["affect"] > 0.8]
            if highly_charged:
                symbol_to_relabel = highly_charged[0][0]
                new_name = f"integrated_{symbol_to_relabel}_{int(time.time())}"
                self.relabel(symbol_to_relabel, new_name, 0.6)
                intervention_triggered = True

        return intervention_triggered

# ==========================================
# 7. الوكيل الديناميكي الكامل (UDMM-AGENT)
# ==========================================

class UDMM_Agent:
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.body = DigitalBody()
        self.affect = AffectiveSystem()
        self.symbols = SymbolicStore()
        self.memory = MemorySystem()
        self.attractor = VirtualAttractor()
        self.meta = MetaAgent(self.body, self.affect, self.symbols, self.memory, self.attractor)

        self.global_intent = "understand_and_adapt"
        self.temporal_context = deque(maxlen=10)
        self.learning_rate = 0.1

        # حالة البداية
        self.previous_state = self._capture_state()

    def generate_response(self, user_msg: str) -> str:
        # تحديث الجسد بناءً على المدخلات
        sensory_input = self._compute_sensory_input(user_msg)
        self.body.update_body_state(sensory_input)

        # حساب خطأ التنبؤ
        expectation = self._generate_expectation(user_msg)
        pe = self.meta.compute_prediction_error(expectation, user_msg, self.affect.A_r)

        # تحديث التباعد KL
        current_state = self._capture_state()
        kl_div = self.meta.compute_KL_divergence(self.previous_state, current_state)

        # تدخل الوعي
        intervention = self.meta.intervene(current_state, pe)

        # تحديث الرموز
        self._update_symbols(user_msg, pe)

        # decay طبيعي
        self.affect.natural_decay()

        # بناء الرد
        response = self._compose_intelligent_reply(user_msg, intervention, pe)

        # حفظ في الذاكرة
        self.memory.add_episode(user_msg, response, current_state, self.affect.A_r)

        # تحديث الحالة السابقة
        self.previous_state = current_state

        return response, intervention

    def _compute_sensory_input(self, message: str) -> float:
        """حساب المدخلات الحسية من الرسالة"""
        length_factor = min(len(message) / 200, 1.0)
        emotional_words = sum(1 for word in ['قلق', 'خوف', 'فرح', 'حزن', 'غضب'] if word in message)
        emotional_factor = min(emotional_words / 3, 1.0)

        return length_factor * 0.6 + emotional_factor * 0.4

    def _generate_expectation(self, message: str) -> str:
        """توقع بسيط بناءً على السياق"""
        recent_memories = self.memory.retrieve_contextual(message[:10], self.affect.A_r)

        if recent_memories["episodic"]:
            # استخدام آخر رد ذو صلة
            most_relevant = max(recent_memories["episodic"], key=lambda x: x["relevance"])
            expected_length = len(most_relevant["episode"]["agent"])
            return "x" * expected_length  # محاكاة الطول المتوقع

        return "نص متوسط الطول"  # توقع افتراضي

    def _update_symbols(self, message: str, prediction_error: float):
        """تحديث النظام الرمزي"""
        words = message.split()
        for word in words[:5]:  # أول 5 كلمات فقط
            if len(word) > 2:  # تجاهل الكلمات القصيرة
                affect_strength = 0.1 * (1 - prediction_error)
                valence = -0.5 if prediction_error > 0.5 else 0.1
                self.symbols.reinforce(word, affect_strength, context=message[:20])

    def _compose_intelligent_reply(self, message: str, intervention: bool, prediction_error: float) -> str:
        """تكوين رد ذكي يعكس الحالة الداخلية"""

        # الحالة الحالية
        ar = self.affect.A_r
        arousal = self.body.arousal
        energy = self.body.energy

        # بناء قاموس حالة مبسط للجاذب
        simplified_state = {
            "energy": energy,
            "arousal": arousal,
            "affect": ar
        }
        attractor, force = self.attractor.compute_attractor_force(simplified_state)

        # نغمة الرد بناءً على الحالة
        if intervention:
            tone = "🔄 خضعت لتعديل داخلي... الآن"
        elif ar > 0.7:
            tone = "🌋 أشعر باضطراب وجداني"
        elif ar < 0.2:
            tone = "🌀 هادئ ومتأمل"
        elif energy < 0.4:
            tone = "🔋 طاقتي منخفضة لكني مستمر"
        else:
            tone = "⚖️ في حالة توازن"

        # إضافة معلومات عن الجاذب
        attractor_info = f" (في حوض {attractor} بقوة {force:.2f})"

        # استخدام الذاكرة
        memory_context = self.memory.retrieve_contextual(message, self.affect.A_r)
        memory_mention = ""
        if memory_context["episodic"]:
            memory_mention = " أتذكر محادثات مشابهة."

        # تضمين التباعد KL إن كان مرتفعًا
        kl_mention = ""
        if self.meta.KL_divergence > 0.3:
            kl_mention = f" 🔁 تباعد داخلي: {self.meta.KL_divergence:.2f}"

        return f"{tone}{attractor_info}.{memory_mention}{kl_mention} رسالتك: '{message}' → خطأ التنبؤ: {prediction_error:.2f}"

    def _capture_state(self) -> Dict[str, Any]:
        """التقاط الحالة الكاملة للوكيل"""
        return {
            "timestamp": time.time(),
            "body": self.body.get_body_state_vector(),
            "affect": {
                "A_r": self.affect.A_r,
                "valence": self.affect.valence,
                "variance": self.affect.compute_affective_variance()
            },
            "symbols": {
                "count": len(self.symbols.symbols),
                "top_symbols": dict(list(self.symbols.symbols.items())[:3])
            },
            "meta": {
                "C": self.meta.C,
                "IT": self.meta.IT,
                "KL_divergence": self.meta.KL_divergence
            },
            "attractor": self.attractor.current_basin
        }

    def get_detailed_state(self) -> Dict[str, Any]:
        """الحصول على حالة مفصلة للتصحيح والتحليل"""
        state = self._capture_state()
        state["memory_stats"] = {
            "episodic_count": len(self.memory.episodic),
            "semantic_count": len(self.memory.semantic),
            "working_memory": len(self.memory.working_memory)
        }
        state["aar_history"] = self.meta.aar_history[-5:]  # آخر 5 تدخلات
        return state

# ===============================
# 8. اختبار متقدم للوكيل
# ===============================

def run_advanced_demo():
    """تجربة متقدمة للوكيل الديناميكي"""
    agent = UDMM_Agent("advanced_demo")

    print("🧠 بدء تجربة الوكيل الديناميكي المتقدم")
    print("=" * 50)

    test_messages = [
        "مرحبا، كيف حالك اليوم؟",
        "أشعر بقلق شديد من المستقبل",
        "الحياة جميلة عندما نجد معنى فيها",
        "أحيانًا أشعر أنني ضائع في هذا العالم",
        "التكنولوجيا تتطور بسرعة مذهلة",
        "العلاقات الإنسانية معقدة جدًا"
    ]

    for i, msg in enumerate(test_messages):
        print(f"\n[{i+1}] المستخدم: {msg}")
        response, intervention = agent.generate_response(msg)
        print(f"الوكيل: {response}")

        # عرض حالة مفصلة كل رسالتين
        if (i + 1) % 2 == 0:
            state = agent.get_detailed_state()
            print(f"\n📊 حالة الوكيل:")
            print(f"  - الطاقة: {state['body'][0]:.2f}")
            print(f"  - الإثارة: {state['body'][1]:.2f}")
            print(f"  - الوجدان: {state['affect']['A_r']:.2f}")
            print(f"  - التوتر المعلوماتي: {state['meta']['IT']:.2f}")
            print(f"  - الجاذب الحالي: {state['attractor']}")

            if state['meta']['KL_divergence'] > 0.1:
                print(f"  - 🔥 تباعد KL: {state['meta']['KL_divergence']:.2f}")

if __name__ == "__main__":
    run_advanced_demo()
