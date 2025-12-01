import time
import json
from collections import deque
from typing import Dict, Any

from .body import DigitalBody
from .affect import AffectiveSystem
from .symbolic import SymbolicStore
from .memory import MemorySystem
from .attractor import VirtualAttractor
from .meta_agent import MetaAgent
from .llm import OpenAIModel, GeminiModel, EchoModel
from .config import LLM_PROVIDER, OPENAI_MODEL

class UDMM_Agent:
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.body = DigitalBody()
        self.affect = AffectiveSystem()
        self.symbols = SymbolicStore()
        self.memory = MemorySystem()
        self.attractor = VirtualAttractor()

        # Initialize LLM
        self.llm = self._init_llm()

        # MetaAgent needs all components, including the LLM if it influences interventions
        self.meta = MetaAgent(self.body, self.affect, self.symbols, self.memory, self.attractor)

        self.global_intent = "understand_and_adapt"
        self.temporal_context = deque(maxlen=10)
        self.learning_rate = 0.1

        self.previous_state = self._capture_state()

    def _init_llm(self):
        provider = LLM_PROVIDER.lower()
        if provider == "openai":
            return OpenAIModel(OPENAI_MODEL)
        if provider == "gemini":
            return GeminiModel()
        # Default to echo model for safety
        return EchoModel()

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

        # This now calls the LLM
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
        simplified_state = self.meta._get_current_state()
        attractor, force = self.attractor.compute_attractor_force(simplified_state)

        # Build the detailed prompt for the LLM
        prompt = f"""
أنت وكيل معرفي متقدم، ومهمتك هي فهم الأسئلة باللغة العربية والرد عليها بدقة ووضوح باللغة العربية الفصحى.
حالتك الداخلية الحالية هي كما يلي:
- **الحالة الجسدية**: طاقة={self.body.energy:.2f}, إثارة={self.body.arousal:.2f}
- **الحالة الوجدانية**: وجدان خام={self.affect.A_r:.2f}, تكافؤ={self.affect.valence:.2f}
- **حوض الجذب**: أنت حاليًا في حوض '{attractor}' بقوة جذب {force:.2f}.
- **خطأ التنبؤ الأخير**: {prediction_error:.2f}
- **التوتر المعلوماتي**: {self.meta.IT:.2f}

---
مثال:
السؤال: "ما هي عاصمة فرنسا؟"
الجواب: "عاصمة فرنسا هي باريس."
---

الآن، بناءً على حالتك الداخلية والسياق، أجب على السؤال التالي: "{message}"
"""

        # Call the LLM with a temperature influenced by arousal/affect
        temp = max(0.1, min(1.0, 0.4 + self.body.arousal * 0.5 - self.affect.A_r * 0.3))

        llm_response = self.llm.chat(prompt, temperature=temp, max_tokens=256)

        return llm_response

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
    # This function is now for simple CLI testing and will use the configured LLM
    print("🧠 بدء تجربة الوكيل الديناميكي المتقدم (CLI)")
    print("=" * 50)
    # Ensure you have OPENAI_API_KEY or GEMINI_API_KEY in your environment
    agent = UDMM_Agent("cli_demo")

    test_messages = [
        "مرحبا، كيف حالك اليوم؟",
        "ما هو شعورك حيال الذكاء الاصطناعي؟",
        "أعطني فكرة إبداعية."
    ]

    for i, msg in enumerate(test_messages):
        print(f"\n[{i+1}] المستخدم: {msg}")
        response, _ = agent.generate_response(msg)
        print(f"الوكيل: {response}")
        state = agent.get_detailed_state()
        print(f"  (الحالة الداخلية: طاقة={state['body'][0]:.2f}, إثارة={state['body'][1]:.2f}, وجدان={state['affect']['A_r']:.2f})")

if __name__ == "__main__":
    run_advanced_demo()
