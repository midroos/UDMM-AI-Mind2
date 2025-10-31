# udmm_core.py
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import deque, defaultdict
import numpy as np
from .config import EMBEDDING_MODEL, FAISS_INDEX_PATH, FAISS_META_PATH, LLM_PROVIDER, OPENAI_MODEL, LLAMACPP_SERVER, ENV_ALLOW_LEARN
from .memory.faiss_rag import FaissRAG
from .intent.hierarchical import HierarchicalIntentManager
from .llm import OpenAIModel, LlamaCPPModel, EchoModel, GeminiModel

class CognitiveEquilibrium:
    """
    فئة لنمذجة التوازن المعرفي للوكيل بناءً على نظرية UDMM
    نماذج القوى المتعارضة بين الأهداف الداخلية والتكيف مع الواقع
    """

    def __init__(self, initial_C_m: float = 0.5):
        self.C_m = initial_C_m  # معامل التوازن المعرفي
        self.tau = 0.3  # التوتر الهدفي (الأهداف الداخلية)
        self.gamma = 0.3  # التوتر الواقعي (التكيف مع البيئة)
        self.equilibrium_history = deque(maxlen=1000)  # سجل التوازن
        self.affective_states_log = deque(maxlen=500)  # سجل الحالات العاطفية

        # تهيئة السجل
        self._record_equilibrium_state()

    def update_equilibrium(self, goal_alignment: float, reality_tension: float,
                         learning_rate: float = 0.1) -> Dict[str, Any]:
        """
        تحديث التوازن المعرفي بناءً على القوى المتعارضة

        Args:
            goal_alignment: محاذاة الأهداف (0-1)
            reality_tension: توتر الواقع (0-1)
            learning_rate: معدل التعلم للتعديلات

        Returns:
            تحليل مفصل للتوازن المعرفي
        """
        # تطبيق التنعيم على التغيرات
        smoothed_tau = (1 - learning_rate) * self.tau + learning_rate * goal_alignment
        smoothed_gamma = (1 - learning_rate) * self.gamma + learning_rate * reality_tension

        self.tau = max(0.0, min(1.0, smoothed_tau))
        self.gamma = max(0.0, min(1.0, smoothed_gamma))

        # حساب C_m مع تجنب القسمة على الصفر
        if self.tau + self.gamma > 0:
            self.C_m = self.tau / (self.tau + self.gamma)
        else:
            self.C_m = 0.5

        # تسجيل الحالة الحالية
        equilibrium_analysis = self._record_equilibrium_state()

        return equilibrium_analysis

    def _record_equilibrium_state(self) -> Dict[str, Any]:
        """تسجيل حالة التوازن الحالية في السجل"""
        affective_state, emoji, color = self._get_detailed_affective_state()

        state_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "C_m": float(self.C_m),
            "tau": float(self.tau),
            "gamma": float(self.gamma),
            "affective_state": affective_state,
            "emoji": emoji,
            "color_code": color,
            "stability_index": self._calculate_stability_index(),
            "trend": self._analyze_equilibrium_trend()
        }

        self.equilibrium_history.append(state_record)
        self.affective_states_log.append(affective_state)

        return state_record

    def _get_detailed_affective_state(self) -> Tuple[str, str, str]:
        """الحصول على حالة عاطفية مفصلة مع رموز وألوان"""
        if self.C_m < 0.2:
            return "قلق حاد", "😰", "#FF6B6B"  # أحمر
        elif self.C_m < 0.35:
            return "قلق معتدل", "😟", "#FFA726"  # برتقالي
        elif self.C_m < 0.45:
            return "قلق طفيف", "🫤", "#FFD54F"  # أصفر
        elif 0.45 <= self.C_m <= 0.55:
            return "توازن مثالي", "😊", "#4CAF50"  # أخضر
        elif self.C_m <= 0.65:
            return "حماس معتدل", "😄", "#66BB6A"  # أخضر فاتح
        elif self.C_m <= 0.8:
            return "حماس عالي", "🚀", "#29B6F6"  # أزرق
        else:
            return "حماس مفرط", "🔥", "#AB47BC"  # بنفسجي

    def _calculate_stability_index(self) -> float:
        """حساب مؤشر الاستقرار بناءً على التغيرات الأخيرة"""
        if len(self.equilibrium_history) < 5:
            return 1.0

        recent_changes = []
        for i in range(1, min(6, len(self.equilibrium_history))):
            change = abs(self.equilibrium_history[-i]["C_m"] -
                        self.equilibrium_history[-(i+1)]["C_m"])
            recent_changes.append(change)

        avg_change = np.mean(recent_changes)
        stability = max(0.0, 1.0 - avg_change * 10)  # تحويل التغير إلى استقرار
        return float(stability)

    def _analyze_equilibrium_trend(self) -> str:
        """تحليل اتجاه التوازن المعرفي"""
        if len(self.equilibrium_history) < 3:
            return "ثابت"

        recent = [state["C_m"] for state in list(self.equilibrium_history)[-5:]]
        if len(recent) < 2:
            return "ثابت"

        changes = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        avg_change = np.mean(changes)

        if abs(avg_change) < 0.02:
            return "ثابت"
        elif avg_change > 0.05:
            return "تحسن سريع"
        elif avg_change > 0.02:
            return "تحسن تدريجي"
        elif avg_change < -0.05:
            return "تراجع سريع"
        else:
            return "تراجع تدريجي"

    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """تحليل شامل للتوازن المعرفي"""
        current_state = self.equilibrium_history[-1] if self.equilibrium_history else {}

        # تحليل التوزيع التاريخي
        if self.equilibrium_history:
            c_m_values = [state["C_m"] for state in self.equilibrium_history]
            tau_values = [state["tau"] for state in self.equilibrium_history]
            gamma_values = [state["gamma"] for state in self.equilibrium_history]
        else:
            c_m_values, tau_values, gamma_values = [self.C_m], [self.tau], [self.gamma]

        return {
            "current_equilibrium": current_state,
            "statistical_analysis": {
                "c_m_mean": float(np.mean(c_m_values)),
                "c_m_std": float(np.std(c_m_values)),
                "tau_mean": float(np.mean(tau_values)),
                "gamma_mean": float(np.mean(gamma_values)),
                "volatility_index": float(np.std(c_m_values) * 10),
                "preferred_state": self._get_most_common_affective_state()
            },
            "performance_metrics": {
                "total_records": len(self.equilibrium_history),
                "stability_percentage": self._calculate_stability_percentage(),
                "optimal_balance_percentage": self._calculate_optimal_balance_percentage(),
                "recent_trend": self._analyze_equilibrium_trend()
            },
            "recommendations": self._generate_equilibrium_recommendations()
        }

    def _get_most_common_affective_state(self) -> str:
        """الحصول على الحالة العاطفية الأكثر شيوعًا"""
        if not self.affective_states_log:
            return "متوازن"

        from collections import Counter
        counter = Counter(self.affective_states_log)
        return counter.most_common(1)[0][0]

    def _calculate_stability_percentage(self) -> float:
        """حساب نسبة الاستقرار في السجل"""
        if len(self.equilibrium_history) < 2:
            return 100.0

        stable_count = sum(1 for state in self.equilibrium_history
                          if state["stability_index"] > 0.7)
        return (stable_count / len(self.equilibrium_history)) * 100

    def _calculate_optimal_balance_percentage(self) -> float:
        """حساب نسبة الوقت في التوازن الأمثل"""
        if not self.equilibrium_history:
            return 0.0

        optimal_count = sum(1 for state in self.equilibrium_history
                           if 0.4 <= state["C_m"] <= 0.6)
        return (optimal_count / len(self.equilibrium_history)) * 100

    def _generate_equilibrium_recommendations(self) -> List[str]:
        """توليد توصيات لتحسين التوازن المعرفي"""
        recommendations = []
        current_state = self.equilibrium_history[-1] if self.equilibrium_history else {}

        if not current_state:
            return ["لا توجد بيانات كافية للتوصيات"]

        c_m = current_state["C_m"]

        if c_m < 0.3:
            recommendations.extend([
                "زيادة التركيز على الأهداف الداخلية",
                "تقليل التعرض للمحفزات الخارجية",
                "ممارسة تمارين الاسترخاء المعرفي"
            ])
        elif c_m > 0.7:
            recommendations.extend([
                "تحسين التكيف مع الواقع الخارجي",
                "زيادة المرونة في تحقيق الأهداف",
                "موازنة الطموحات مع الإمكانيات"
            ])
        else:
            recommendations.append("الحفاظ على الاستراتيجيات الحالية")

        if current_state.get("stability_index", 1.0) < 0.6:
            recommendations.append("تحسين استقرار التوازن المعرفي")

        return recommendations


class EmbodiedBodyModel:
    """
    نموذج جسدي محسّن للوكيل يدمج الحالات الجسدية المعقدة
    مع التحديث الديناميكي بناءً على الحالة المعرفية
    """

    def __init__(self):
        # الحالة الأساسية للجسد
        self.energy = 1.0
        self.arousal = 0.1
        self.vitality = 0.8

        # الحالة الجسدية التفصيلية
        self.physical_state = {
            "posture": "واقف",
            "posture_intensity": 0.5,
            "movement": "هادئ",
            "movement_fluidity": 0.6,
            "expression": "محايد",
            "expression_intensity": 0.4,
            "gesture": "لا شيء",
            "gesture_expressiveness": 0.3,
            "breathing_pattern": "منتظم",
            "muscle_tension": 0.3
        }

        # سجل الحالات الجسدية
        self.physical_history = deque(maxlen=500)
        self.movement_patterns = defaultdict(int)

        # معاملات التعب الجسدي
        self.fatigue_coefficient = 0.1
        self.recovery_rate = 0.05

        # تسجيل الحالة الأولية
        self._record_physical_state("تهيئة أولية")

    def apply_embodied_action(self, action: Dict[str, Any],
                            cognitive_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        تطبيق فعل مجسد مع التحديث الشامل للحالة الجسدية

        Args:
            action: الفعل المطلوب تنفيذه
            cognitive_state: الحالة المعرفية الحالية

        Returns:
            الحالة الجسدية المحدثة مع التحليل
        """
        # استخراج معاملات الفعل
        cost = action.get("cost", 0.02)
        arousal_delta = action.get("arousal_delta", 0.0)
        physical_intensity = action.get("physical_intensity", 0.3)

        # تحديث الحالة الأساسية
        self.energy = max(0.0, self.energy - cost * (1 + physical_intensity))
        self.arousal = min(1.0, max(0.0, self.arousal + arousal_delta))

        # تحديث الحيوية بناءً على الطاقة والإثارة
        self.vitality = (self.energy * 0.6 + (1 - abs(self.arousal - 0.5)) * 0.4)

        # تحديث التجسيد الجسدي بناءً على الحالة المعرفية
        self._update_physical_embodiment(cognitive_state, physical_intensity)

        # تسجيل الحالة الجسدية
        physical_analysis = self._record_physical_state(action.get("type", "غير محدد"))

        return {
            **self.physical_state,
            "energy": self.energy,
            "arousal": self.arousal,
            "vitality": self.vitality,
            "physical_analysis": physical_analysis
        }

    def _update_physical_embodiment(self, cognitive_state: Dict[str, Any],
                                  intensity: float):
        """تحديث التجسيد الجسدي بناءً على الحالة المعرفية"""
        C_m = cognitive_state.get("C_m", 0.5)
        affective_state = cognitive_state.get("affective_state", "متوازن")

        # تحديد نمط التجسيد بناءً على C_m
        if C_m < 0.3:
            # نمط القلق - توتر وانقباض
            self.physical_state.update({
                "posture": "منقبض",
                "posture_intensity": 0.8,
                "movement": "متقطع",
                "movement_fluidity": 0.3,
                "expression": "قلق",
                "expression_intensity": 0.7,
                "gesture": "فرك اليدين",
                "gesture_expressiveness": 0.6,
                "breathing_pattern": "سريع",
                "muscle_tension": 0.8
            })
        elif C_m < 0.45:
            # نمط القلق الخفيف - حذر
            self.physical_state.update({
                "posture": "حذر",
                "posture_intensity": 0.6,
                "movement": "متريث",
                "movement_fluidity": 0.5,
                "expression": "جدي",
                "expression_intensity": 0.5,
                "gesture": "طي الذراعين",
                "gesture_expressiveness": 0.4,
                "breathing_pattern": "منتظم",
                "muscle_tension": 0.5
            })
        elif 0.45 <= C_m <= 0.55:
            # نمط التوازن - انسجام
            self.physical_state.update({
                "posture": "مستقيم",
                "posture_intensity": 0.5,
                "movement": "انسجام",
                "movement_fluidity": 0.8,
                "expression": "هادئ",
                "expression_intensity": 0.4,
                "gesture": "إيماءات طبيعية",
                "gesture_expressiveness": 0.5,
                "breathing_pattern": "منتظم",
                "muscle_tension": 0.3
            })
        elif C_m <= 0.7:
            # نمط الحماس - نشاط
            self.physical_state.update({
                "posture": "منفتح",
                "posture_intensity": 0.6,
                "movement": "نشيط",
                "movement_fluidity": 0.7,
                "expression": "متحمس",
                "expression_intensity": 0.6,
                "gesture": "إيماءات واسعة",
                "gesture_expressiveness": 0.7,
                "breathing_pattern": "عميق",
                "muscle_tension": 0.4
            })
        else:
            # نمط الحماس الشديد - طاقة عالية
            self.physical_state.update({
                "posture": "متفجر",
                "posture_intensity": 0.9,
                "movement": "سريع",
                "movement_fluidity": 0.9,
                "expression": "متحمس بشدة",
                "expression_intensity": 0.8,
                "gesture": "إيماءات دراماتيكية",
                "gesture_expressiveness": 0.9,
                "breathing_pattern": "سريع وعميق",
                "muscle_tension": 0.6
            })

        # تطبيق شدة النشاط
        for key in ["posture_intensity", "movement_fluidity", "expression_intensity",
                   "gesture_expressiveness"]:
            self.physical_state[key] = min(1.0, self.physical_state[key] * (1 + intensity * 0.5))

    def _record_physical_state(self, action_type: str) -> Dict[str, Any]:
        """تسجيل الحالة الجسدية الحالية"""
        state_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "action_type": action_type,
            **self.physical_state,
            "energy": self.energy,
            "arousal": self.arousal,
            "vitality": self.vitality
        }

        self.physical_history.append(state_record)

        # تحديث أنماط الحركة
        movement_key = f"{self.physical_state['movement']}_{self.physical_state['posture']}"
        self.movement_patterns[movement_key] += 1

        return {
            "movement_pattern": movement_key,
            "physical_coherence": self._calculate_physical_coherence(),
            "fatigue_level": 1.0 - self.energy,
            "expressiveness_score": self._calculate_expressiveness_score()
        }

    def _calculate_physical_coherence(self) -> float:
        """حساب تماسك الحالة الجسدية"""
        # تماسك عالي عندما تكون الشدة متناسبة مع الطاقة
        intensity_avg = np.mean([
            self.physical_state["posture_intensity"],
            self.physical_state["expression_intensity"],
            self.physical_state["gesture_expressiveness"]
        ])

        coherence = 1.0 - abs(intensity_avg - self.energy)
        return max(0.0, min(1.0, coherence))

    def _calculate_expressiveness_score(self) -> float:
        """حساب درجة التعبيرية الجسدية"""
        factors = [
            self.physical_state["expression_intensity"],
            self.physical_state["gesture_expressiveness"],
            self.physical_state["movement_fluidity"],
            self.arousal  # الإثارة تزيد التعبيرية
        ]

        return float(np.mean(factors))

    def get_physical_analysis_report(self) -> Dict[str, Any]:
        """تقرير تحليلي مفصل عن الحالة الجسدية"""
        if not self.physical_history:
            return {"status": "لا توجد بيانات كافية"}

        recent_states = list(self.physical_history)[-10:]

        return {
            "current_state": self.physical_state,
            "vitality_metrics": {
                "energy_level": self.energy,
                "arousal_level": self.arousal,
                "vitality_score": self.vitality,
                "fatigue_percentage": (1.0 - self.energy) * 100
            },
            "movement_analysis": {
                "common_patterns": dict(self.movement_patterns),
                "current_fluidity": self.physical_state["movement_fluidity"],
                "expressiveness_trend": self._analyze_expressiveness_trend()
            },
            "coherence_metrics": {
                "physical_coherence": self._calculate_physical_coherence(),
                "energy_efficiency": self._calculate_energy_efficiency(),
                "stability_score": self._calculate_physical_stability()
            }
        }

    def _analyze_expressiveness_trend(self) -> str:
        """تحليل اتجاه التعبيرية الجسدية"""
        if len(self.physical_history) < 3:
            return "ثابت"

        recent_scores = [state["expression_intensity"] for state in list(self.physical_history)[-5:]]
        avg_score = np.mean(recent_scores)
        prev_avg = np.mean([state["expression_intensity"] for state in list(self.physical_history)[-10:-5]])

        if avg_score > prev_avg + 0.1:
            return "متزايد"
        elif avg_score < prev_avg - 0.1:
            return "متناقص"
        else:
            return "مستقر"

    def _calculate_energy_efficiency(self) -> float:
        """حساب كفاءة استخدام الطاقة"""
        if not self.physical_history:
            return 1.0

        # كفاءة عالية عندما يكون استهلاك الطاقة متناسب مع النشاط
        recent_states = list(self.physical_history)[-10:]
        if not recent_states:
            return 1.0

        total_intensity = sum(state["posture_intensity"] + state["expression_intensity"]
                             for state in recent_states)
        total_energy_used = sum(1 - state["energy"] for state in recent_states)

        if total_energy_used > 0:
            efficiency = total_intensity / total_energy_used
            return min(2.0, efficiency) / 2.0  # تطبيع إلى 0-1
        return 1.0

    def _calculate_physical_stability(self) -> float:
        """حساب استقرار الحالة الجسدية"""
        if len(self.physical_history) < 5:
            return 1.0

        recent_states = list(self.physical_history)[-10:]
        changes = []

        for i in range(1, len(recent_states)):
            prev = recent_states[i-1]
            curr = recent_states[i]

            # حساب التغير في المعاملات الرئيسية
            posture_change = abs(prev["posture_intensity"] - curr["posture_intensity"])
            expression_change = abs(prev["expression_intensity"] - curr["expression_intensity"])
            movement_change = abs(prev["movement_fluidity"] - curr["movement_fluidity"])

            avg_change = (posture_change + expression_change + movement_change) / 3
            changes.append(avg_change)

        avg_change = np.mean(changes) if changes else 0
        stability = max(0.0, 1.0 - avg_change * 5)  # تحويل التغير إلى استقرار
        return stability


class EmbodimentSystem:
    """
    نظام تجسيد متعدد الوسائط مع إمكانيات تعبيرية متقدمة
    يدعم وسائط متعددة مع تحويل سلس بينها
    """

    def __init__(self):
        self.available_modes = {
            "text": TextEmbodiment(),
            "avatar": AvatarEmbodiment(),
            "hardware": HardwareEmbodiment(),
            "reverse": ReverseEmbodiment(),
            "multimodal": MultimodalEmbodiment()
        }

        self.current_mode = "text"
        self.embodiment_history = deque(maxlen=1000)
        self.mode_effectiveness = defaultdict(lambda: {"uses": 0, "success_rate": 0.0})

        # إعدادات النظام
        self.auto_switch_threshold = 0.7
        self.expression_intensity = 0.8

    def set_embodiment_mode(self, mode: str, intensity: float = None) -> Dict[str, Any]:
        """تغيير نمط التجسيد مع التحكم بالشدة"""
        if mode not in self.available_modes:
            return {
                "success": False,
                "message": f"نمط التجسيد غير مدعوم: {mode}",
                "available_modes": list(self.available_modes.keys())
            }

        previous_mode = self.current_mode
        self.current_mode = mode

        if intensity is not None:
            self.expression_intensity = max(0.1, min(1.0, intensity))

        # تسجيل تغيير النمط
        switch_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_mode": previous_mode,
            "to_mode": mode,
            "intensity": self.expression_intensity,
            "type": "mode_switch"
        }

        self.embodiment_history.append(switch_record)

        return {
            "success": True,
            "message": f"✅ تم تغيير نمط التجسيد إلى: {mode}",
            "previous_mode": previous_mode,
            "new_mode": mode,
            "intensity": self.expression_intensity
        }

    def generate_embodied_response(self,
                                 text_response: str,
                                 cognitive_equilibrium: CognitiveEquilibrium,
                                 physical_state: Dict[str, Any],
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        توليد استجابة مجسدة شاملة تجمع بين جميع المكونات

        Args:
            text_response: الرد النصي الأساسي
            cognitive_equilibrium: حالة التوازن المعرفي
            physical_state: الحالة الجسدية
            context: السياق الإضافي

        Returns:
            استجابة مجسدة كاملة متعددة الوسائط
        """
        if context is None:
            context = {}

        # الحصول على المحلل الحالي
        embodiment_module = self.available_modes[self.current_mode]

        # توليد الاستجابة المجسدة
        embodied_output = embodiment_module.express(
            text=text_response,
            cognitive_equilibrium=cognitive_equilibrium,
            physical_state=physical_state,
            context=context,
            intensity=self.expression_intensity
        )

        # إضافة تحليل التكامل
        integration_analysis = self._analyze_embodiment_integration(
            embodied_output, cognitive_equilibrium, physical_state
        )

        # تسجيل الاستجابة
        response_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": self.current_mode,
            "input_text": text_response,
            "output": embodied_output,
            "integration_analysis": integration_analysis,
            "cognitive_state": cognitive_equilibrium.get_comprehensive_analysis(),
            "physical_state": physical_state,
            "type": "embodied_response"
        }

        self.embodiment_history.append(response_record)

        # تحديث فعالية النمط
        self._update_mode_effectiveness(integration_analysis)

        return {
            "embodied_response": embodied_output,
            "integration_analysis": integration_analysis,
            "embodiment_mode": self.current_mode,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _analyze_embodiment_integration(self, embodied_output: Dict[str, Any],
                                      cognitive_equilibrium: CognitiveEquilibrium,
                                      physical_state: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل تكامل التجسيد مع الحالة المعرفية والجسدية"""
        C_m = cognitive_equilibrium.C_m

        # تحليل التوافق بين النمط والحالة
        mode_compatibility = self._calculate_mode_compatibility(
            self.current_mode, C_m, physical_state
        )

        # تحليل التماسك التعبيري
        expressive_coherence = self._calculate_expressive_coherence(
            embodied_output, cognitive_equilibrium, physical_state
        )

        return {
            "mode_compatibility": mode_compatibility,
            "expressive_coherence": expressive_coherence,
            "overall_integration_score": (mode_compatibility + expressive_coherence) / 2,
            "recommended_improvements": self._generate_integration_recommendations(
                mode_compatibility, expressive_coherence
            )
        }

    def _calculate_mode_compatibility(self, mode: str, C_m: float,
                                    physical_state: Dict[str, Any]) -> float:
        """حساب توافق نمط التجسيد مع الحالة الحالية"""
        base_compatibility = 0.7

        # تعديل التوافق بناءً على C_m
        if mode == "text" and (C_m < 0.3 or C_m > 0.7):
            base_compatibility += 0.2  # النص أفضل في الحالات المتطرفة
        elif mode == "avatar" and 0.4 <= C_m <= 0.6:
            base_compatibility += 0.3  # الصورة الرمزية أفضل في التوازن

        # تعديل بناءً على الحالة الجسدية
        vitality = physical_state.get("vitality", 0.5)
        if mode in ["avatar", "hardware"] and vitality > 0.7:
            base_compatibility += 0.2

        return min(1.0, base_compatibility)

    def _calculate_expressive_coherence(self, embodied_output: Dict[str, Any],
                                      cognitive_equilibrium: CognitiveEquilibrium,
                                      physical_state: Dict[str, Any]) -> float:
        """حساب تماسك التعبير بين المكونات المختلفة"""
        # تحليل التوافق بين الحالة المعرفية والتعبير الجسدي
        C_m = cognitive_equilibrium.C_m
        physical_intensity = physical_state.get("expression_intensity", 0.5)

        # التماسك المثالي عندما يتناسب التعبير مع الحالة المعرفية
        expected_intensity = abs(C_m - 0.5) * 2  # 0 عند التوازن، 1 عند التطرف
        intensity_coherence = 1.0 - abs(expected_intensity - physical_intensity)

        # تحليل تناسق المحتوى
        content_coherence = self._analyze_content_coherence(embodied_output, C_m)

        return (intensity_coherence + content_coherence) / 2

    def _analyze_content_coherence(self, embodied_output: Dict[str, Any],
                                 C_m: float) -> float:
        """تحليل تماسك المحتوى مع الحالة المعرفية"""
        # هذا تنفيذ أساسي - يمكن تطويره لتحليل النص والسياق
        output_type = embodied_output.get("type", "unknown")

        if output_type == "text" and "content" in embodied_output:
            content = embodied_output["content"]
            # تحليل بسيط للتناسب العاطفي
            if C_m < 0.4 and "😊" in content:
                return 0.3  # عدم تناسب
            elif C_m > 0.6 and "😰" in content:
                return 0.3  # عدم تناسب
            else:
                return 0.8  # تناسب معقول

        return 0.7  # قيمة افتراضية

    def _generate_integration_recommendations(self, mode_compatibility: float,
                                           expressive_coherence: float) -> List[str]:
        """توليد توصيات لتحسين التكامل"""
        recommendations = []

        if mode_compatibility < 0.6:
            recommendations.append("考慮 تغيير نمط التجسيد لتحسين التوافق")

        if expressive_coherence < 0.7:
            recommendations.append("تحسين تماسك التعبير بين الحالة المعرفية والجسدية")

        if not recommendations:
            recommendations.append("الأداء جيد - الحفاظ على الإعدادات الحالية")

        return recommendations

    def _update_mode_effectiveness(self, integration_analysis: Dict[str, Any]):
        """تحديث فعالية نمط التجسيد"""
        integration_score = integration_analysis.get("overall_integration_score", 0.5)
        mode = self.current_mode

        self.mode_effectiveness[mode]["uses"] += 1

        # تحديث معدل النجاح
        current_stats = self.mode_effectiveness[mode]
        current_success_rate = current_stats["success_rate"]
        current_uses = current_stats["uses"]

        # حساب معدل النجاح المتحرك
        new_success_rate = (current_success_rate * (current_uses - 1) + integration_score) / current_uses
        self.mode_effectiveness[mode]["success_rate"] = new_success_rate

    def get_embodiment_system_report(self) -> Dict[str, Any]:
        """تقرير مفصل عن أداء نظام التجسيد"""
        total_responses = len([r for r in self.embodiment_history if r["type"] == "embodied_response"])

        return {
            "system_overview": {
                "current_mode": self.current_mode,
                "available_modes": list(self.available_modes.keys()),
                "total_embodied_responses": total_responses,
                "expression_intensity": self.expression_intensity
            },
            "mode_performance": {
                mode: {
                    "usage_count": stats["uses"],
                    "success_rate": stats["success_rate"],
                    "effectiveness_rating": self._get_effectiveness_rating(stats["success_rate"])
                }
                for mode, stats in self.mode_effectiveness.items()
            },
            "integration_analytics": self._get_integration_analytics(),
            "recommendations": self._generate_system_recommendations()
        }

    def _get_effectiveness_rating(self, success_rate: float) -> str:
        """تقييم فعالية نمط التجسيد"""
        if success_rate >= 0.8:
            return "ممتاز"
        elif success_rate >= 0.7:
            return "جيد جدًا"
        elif success_rate >= 0.6:
            return "جيد"
        elif success_rate >= 0.5:
            return "مقبول"
        else:
            return "يحتاج تحسين"

    def _get_integration_analytics(self) -> Dict[str, Any]:
        """تحليلات التكامل من السجل"""
        integration_scores = [
            r["integration_analysis"]["overall_integration_score"]
            for r in self.embodiment_history
            if r["type"] == "embodied_response" and "integration_analysis" in r
        ]

        if not integration_scores:
            return {"status": "لا توجد بيانات كافية"}

        return {
            "average_integration_score": float(np.mean(integration_scores)),
            "integration_consistency": float(np.std(integration_scores)),
            "improvement_trend": self._analyze_integration_trend(integration_scores),
            "performance_level": self._get_performance_level(np.mean(integration_scores))
        }

    def _analyze_integration_trend(self, scores: List[float]) -> str:
        """تحليل اتجاه أداء التكامل"""
        if len(scores) < 5:
            return "غير محدد"

        recent = scores[-5:]
        older = scores[-10:-5] if len(scores) >= 10 else scores[:5]

        if not older:
            return "غير محدد"

        recent_avg = np.mean(recent)
        older_avg = np.mean(older)

        if recent_avg > older_avg + 0.05:
            return "تحسن"
        elif recent_avg < older_avg - 0.05:
            return "تراجع"
        else:
            return "مستقر"

    def _get_performance_level(self, score: float) -> str:
        """تحديد مستوى الأداء"""
        if score >= 0.8:
            return "متميز"
        elif score >= 0.7:
            return "جيد جدًا"
        elif score >= 0.6:
            return "جيد"
        else:
            return "يحتاج تحسين"

    def _generate_system_recommendations(self) -> List[str]:
        """توليد توصيات لنظام التجسيد"""
        recommendations = []
        performance = self.get_embodiment_system_report()

        # تحليل أداء الأنماط
        for mode, stats in performance["mode_performance"].items():
            if stats["success_rate"] < 0.6 and stats["usage_count"] > 5:
                recommendations.append(f"تحسين أداء نمط {mode} أو تقليل استخدامه")

        # تحليل التكامل
        integration_analytics = performance["integration_analytics"]
        if integration_analytics.get("average_integration_score", 0) < 0.7:
            recommendations.append("العمل على تحسين تكامل النظام بشكل عام")

        if not recommendations:
            recommendations.append("أداء النظام جيد - الاستمرار في المراقبة")

        return recommendations


# تنفيذ وحدات التجسيد (سيتم استكمالها)
class TextEmbodiment:
    def express(self, text, cognitive_equilibrium, physical_state, context, intensity):
        return {"type": "text", "content": text}

class AvatarEmbodiment:
    def express(self, text, cognitive_equilibrium, physical_state, context, intensity):
        return {"type": "avatar", "animation": "neutral"}

class HardwareEmbodiment:
    def express(self, text, cognitive_equilibrium, physical_state, context, intensity):
        return {"type": "hardware", "commands": []}

class ReverseEmbodiment:
    def express(self, text, cognitive_equilibrium, physical_state, context, intensity):
        return {"type": "reverse", "analysis": {}}

class MultimodalEmbodiment:
    def express(self, text, cognitive_equilibrium, physical_state, context, intensity):
        return {"type": "multimodal", "outputs": {}}


class EmbodiedUDMMAgent:
    """
    الوكيل الرئيسي المجسد الذي يدمج جميع المكونات المحسنة
    لإنتاج استجابات ديناميكية تعكس الحالة الداخلية بشكل شامل
    """

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}

        # تهيئة المكونات الأساسية من UDMM
        # (سيتم دمجها مع الكود الحالي)

        # المكونات الجديدة المحسنة
        self.cognitive_equilibrium = CognitiveEquilibrium(
            initial_C_m=config.get('initial_C_m', 0.5)
        )
        self.embodied_body = EmbodiedBodyModel()
        self.embodiment_system = EmbodimentSystem()

        # إحصائيات وتقارير
        self.interaction_history = deque(maxlen=2000)
        self.performance_metrics = {
            "total_interactions": 0,
            "successful_embodiments": 0,
            "average_response_quality": 0.0,
            "system_uptime": time.time()
        }

        # إعدادات الوكيل
        self.learning_enabled = config.get('learning_enabled', True)
        self.auto_optimization = config.get('auto_optimization', True)

        print("🤖 تم تهيئة وكيل UDMM المجسد المحسّن بنجاح!")

    def perceive_and_act(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        التفاعل الرئيسي مع الوكيل - معالجة المدخلات وإنتاج استجابة مجسدة

        Args:
            user_input: مدخلات المستخدم
            context: سياق إضافي للتفاعل

        Returns:
            استجابة مجسدة كاملة مع التحليل
        """
        self.performance_metrics["total_interactions"] += 1
        start_time = time.time()

        if context is None:
            context = {}

        # 1. معالجة المدخلات الأساسية (سيتم دمجها مع UDMM الحالي)
        processed_input = self._process_user_input(user_input, context)

        # 2. حساب التوازن المعرفي
        goal_alignment = self._calculate_goal_alignment(processed_input, context)
        reality_tension = self._calculate_reality_tension(processed_input, context)

        equilibrium_analysis = self.cognitive_equilibrium.update_equilibrium(
            goal_alignment, reality_tension
        )

        # 3. تحديث التجسيد الجسدي
        physical_action = {
            "type": "interaction",
            "cost": 0.015,
            "arousal_delta": 0.01,
            "physical_intensity": 0.4
        }

        body_state = self.embodied_body.apply_embodied_action(
            physical_action, equilibrium_analysis
        )

        # 4. توليد الرد الأساسي (سيتم دمجه مع LLM)
        base_response = self._generate_base_response(processed_input, context)

        # 5. توليد الاستجابة المجسدة
        embodied_response = self.embodiment_system.generate_embodied_response(
            text_response=base_response["text"],
            cognitive_equilibrium=self.cognitive_equilibrium,
            physical_state=body_state,
            context=context
        )

        # 6. تجميع النتيجة النهائية
        processing_time = time.time() - start_time

        final_response = {
            "input_analysis": processed_input,
            "base_response": base_response,
            "embodied_response": embodied_response,
            "cognitive_analysis": equilibrium_analysis,
            "physical_analysis": body_state,
            "performance_metrics": {
                "processing_time": processing_time,
                "system_load": self._calculate_system_load(),
                "response_quality": self._assess_response_quality(embodied_response)
            },
            "interaction_id": self.performance_metrics["total_interactions"],
            "timestamp": datetime.utcnow().isoformat()
        }

        # 7. تحديث الإحصائيات والسجل
        self._update_interaction_history(final_response)

        return final_response

    def _process_user_input(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """معالجة مدخلات المستخدم (سيتم دمجه مع النظام الحالي)"""
        return {
            "text": user_input,
            "length": len(user_input),
            "complexity": min(1.0, len(user_input) / 500),
            "detected_intent": "general",
            "emotional_tone": "neutral"
        }

    def _calculate_goal_alignment(self, processed_input: Dict[str, Any],
                                context: Dict[str, Any]) -> float:
        """حساب محاذاة الأهداف (سيتم تطويره)"""
        base_alignment = 0.3

        # محاكاة تحسين المحاذاة بناءً على المحتوى
        complexity = processed_input.get("complexity", 0.5)
        if complexity > 0.7:
            base_alignment += 0.3  # المهام المعقدة تحفز الأهداف

        return min(1.0, base_alignment)

    def _calculate_reality_tension(self, processed_input: Dict[str, Any],
                                 context: Dict[str, Any]) -> float:
        """حساب توتر الواقع (سيتم تطويره)"""
        base_tension = 0.3

        # محاكاة زيادة التوتر مع صعوبة المهمة
        complexity = processed_input.get("complexity", 0.5)
        if complexity > 0.8:
            base_tension += 0.4

        return min(1.0, base_tension)

    def _generate_base_response(self, processed_input: Dict[str, Any],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """توليد رد أساسي (سيتم دمجه مع LLM)"""
        return {
            "text": "هذا رد تجريبي من الوكيل المجسد المحسّن.",
            "type": "informational",
            "confidence": 0.8
        }

    def _calculate_system_load(self) -> float:
        """حساب حمل النظام الحالي"""
        history_size = len(self.interaction_history)
        return min(1.0, history_size / 1000)

    def _assess_response_quality(self, embodied_response: Dict[str, Any]) -> float:
        """تقييم جودة الاستجابة"""
        integration_score = embodied_response.get(
            "integration_analysis", {}
        ).get("overall_integration_score", 0.5)

        return integration_score

    def _update_interaction_history(self, response: Dict[str, Any]):
        """تحديث سجل التفاعلات"""
        self.interaction_history.append(response)

        # تحديث إحصائيات النجاح
        quality = response["performance_metrics"]["response_quality"]
        if quality > 0.7:
            self.performance_metrics["successful_embodiments"] += 1

        # تحديث متوسط الجودة
        total = self.performance_metrics["total_interactions"]
        current_avg = self.performance_metrics["average_response_quality"]
        new_avg = (current_avg * (total - 1) + quality) / total
        self.performance_metrics["average_response_quality"] = new_avg

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """تقرير شامل عن أداء الوكيل"""
        cognitive_report = self.cognitive_equilibrium.get_comprehensive_analysis()
        physical_report = self.embodied_body.get_physical_analysis_report()
        embodiment_report = self.embodiment_system.get_embodiment_system_report()

        uptime = time.time() - self.performance_metrics["system_uptime"]

        return {
            "agent_overview": {
                "type": "EmbodiedUDMMAgent",
                "uptime_seconds": uptime,
                "learning_enabled": self.learning_enabled,
                "auto_optimization": self.auto_optimization
            },
            "performance_summary": self.performance_metrics,
            "cognitive_performance": cognitive_report,
            "physical_performance": physical_report,
            "embodiment_performance": embodiment_report,
            "system_health": {
                "overall_health": self._calculate_system_health(),
                "recommendations": self._generate_system_recommendations(),
                "maintenance_suggestions": self._generate_maintenance_suggestions()
            }
        }

    def _calculate_system_health(self) -> float:
        """حساب الصحة العامة للنظام"""
        factors = []

        # الصحة المعرفية
        cognitive_analysis = self.cognitive_equilibrium.get_comprehensive_analysis()
        cognitive_health = cognitive_analysis.get("performance_metrics", {}).get("stability_percentage", 50.0) / 100.0
        factors.append(cognitive_health)

        return 0.0

    def _generate_system_recommendations(self) -> List[str]:
        return []

    def _generate_maintenance_suggestions(self) -> List[str]:
        return []

if __name__ == "__main__":
    agent = EmbodiedUDMMAgent()
    response = agent.perceive_and_act("Hello, world!")
    import pprint
    pprint.pprint(response)
