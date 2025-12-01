import time
from collections import deque
from typing import Dict, Any

from .body import DigitalBody
from .affect import AffectiveSystem
from .symbolic import SymbolicStore
from .memory import MemorySystem
from .attractor import VirtualAttractor
from .meta_agent import MetaAgent

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

        self.previous_state = self._capture_state()

    def generate_response(self, user_msg: str):
        sensory_input = self._compute_sensory_input(user_msg)
        self.body.update_body_state(sensory_input)

        expectation = self._generate_expectation(user_msg)
        pe = self.meta.compute_prediction_error(expectation, user_msg, self.affect.A_r)

        current_state = self._capture_state()
        kl_div = self.meta.compute_KL_divergence(self.previous_state, current_state)

        intervention = self.meta.intervene(current_state, pe)

        self._update_symbols(user_msg, pe)

        self.affect.natural_decay()

        response = self._compose_intelligent_reply(user_msg, intervention, pe)

        self.memory.add_episode(user_msg, response, current_state, self.affect.A_r)

        self.previous_state = current_state

        return response, intervention

    def _compute_sensory_input(self, message: str) -> float:
        length_factor = min(len(message) / 200, 1.0)
        emotional_words = sum(1 for word in ['قلق', 'خوف', 'فرح', 'حزن', 'غضب'] if word in message)
        emotional_factor = min(emotional_words / 3, 1.0)

        return length_factor * 0.6 + emotional_factor * 0.4

    def _generate_expectation(self, message: str) -> str:
        recent_memories = self.memory.retrieve_contextual(message[:10], self.affect.A_r)

        if recent_memories["episodic"]:
            most_relevant = max(recent_memories["episodic"], key=lambda x: x["relevance"])
            expected_length = len(most_relevant["episode"]["agent"])
            return "x" * expected_length

        return "نص متوسط الطول"

    def _update_symbols(self, message: str, prediction_error: float):
        words = message.split()
        for word in words[:5]:
            if len(word) > 2:
                affect_strength = 0.1 * (1 - prediction_error)
                self.symbols.reinforce(word, affect_strength, context=message[:20])

    def _compose_intelligent_reply(self, message: str, intervention: bool, prediction_error: float) -> str:
        ar = self.affect.A_r
        arousal = self.body.arousal
        energy = self.body.energy

        simplified_state = {
            "energy": energy,
            "arousal": arousal,
            "affect": ar
        }
        attractor, force = self.attractor.compute_attractor_force(simplified_state)

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

        attractor_info = f" (في حوض {attractor} بقوة {force:.2f})"

        memory_context = self.memory.retrieve_contextual(message, self.affect.A_r)
        memory_mention = ""
        if memory_context["episodic"]:
            memory_mention = " أتذكر محادثات مشابهة."

        kl_mention = ""
        if self.meta.KL_divergence > 0.3:
            kl_mention = f" 🔁 تباعد داخلي: {self.meta.KL_divergence:.2f}"

        return f"{tone}{attractor_info}.{memory_mention}{kl_mention} رسالتك: '{message}' → خطأ التنبؤ: {prediction_error:.2f}"

    def _capture_state(self) -> Dict[str, Any]:
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
        state = self._capture_state()
        state["memory_stats"] = {
            "episodic_count": len(self.memory.episodic),
            "semantic_count": len(self.memory.semantic),
            "working_memory": len(self.memory.working_memory)
        }
        state["aar_history"] = self.meta.aar_history[-5:]
        return state

def run_advanced_demo():
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
