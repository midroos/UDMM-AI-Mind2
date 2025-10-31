"""
التنفيذ النهائي المحسّن لوكيل UDMM المجسد
Final Enhanced Embodied UDMM Agent Implementation
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
from collections import deque, defaultdict
import numpy as np


class CognitiveEquilibrium:
    """
    نموذج متطور للتوازن المعرفي مع تحليل مفصل وتتبع التاريخ
    """

    def __init__(self, initial_C_m: float = 0.5):
        self.C_m = initial_C_m
        self.tau = 0.3
        self.gamma = 0.3
        self.equilibrium_history = deque(maxlen=1000)
        self.affective_states_log = deque(maxlen=500)
        self._record_equilibrium_state()

    def update_equilibrium(self, goal_alignment: float, reality_tension: float,
                         learning_rate: float = 0.1) -> Dict[str, Any]:
        """
        تحديث التوازن المعرفي مع معالجة الأخطاء المحتملة
        """
        try:
            # تطبيق التنعيم
            smoothed_tau = (1 - learning_rate) * self.tau + learning_rate * goal_alignment
            smoothed_gamma = (1 - learning_rate) * self.gamma + learning_rate * reality_tension

            self.tau = max(0.001, min(1.0, smoothed_tau))  # تجنب الصفر
            self.gamma = max(0.001, min(1.0, smoothed_gamma))  # تجنب الصفر

            # حساب C_m مع التحقق من القسمة على الصفر
            denominator = self.tau + self.gamma
            if denominator > 0:
                self.C_m = self.tau / denominator
            else:
                self.C_m = 0.5

            return self._record_equilibrium_state()

        except (ZeroDivisionError, ValueError) as e:
            print(f"⚠️ خطأ في تحديث التوازن المعرفي: {e}")
            self.C_m = 0.5
            self.tau = 0.3
            self.gamma = 0.3
            return self._record_equilibrium_state()

    def _record_equilibrium_state(self) -> Dict[str, Any]:
        """تسجيل حالة التوازن مع معالجة الأخطاء"""
        try:
            affective_state, emoji, color = self._get_detailed_affective_state()

            state_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
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

        except Exception as e:
            print(f"⚠️ خطأ في تسجيل حالة التوازن: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "C_m": 0.5,
                "tau": 0.3,
                "gamma": 0.3,
                "affective_state": "غير محدد",
                "emoji": "❓",
                "color_code": "#CCCCCC",
                "stability_index": 1.0,
                "trend": "ثابت"
            }

    def _get_detailed_affective_state(self) -> Tuple[str, str, str]:
        """الحصول على حالة عاطفية مفصلة"""
        if self.C_m < 0.2:
            return "قلق حاد", "😰", "#FF6B6B"
        elif self.C_m < 0.35:
            return "قلق معتدل", "😟", "#FFA726"
        elif self.C_m < 0.45:
            return "قلق طفيف", "🫤", "#FFD54F"
        elif 0.45 <= self.C_m <= 0.55:
            return "توازن مثالي", "😊", "#4CAF50"
        elif self.C_m <= 0.65:
            return "حماس معتدل", "😄", "#66BB6A"
        elif self.C_m <= 0.8:
            return "حماس عالي", "🚀", "#29B6F6"
        else:
            return "حماس مفرط", "🔥", "#AB47BC"

    def _calculate_stability_index(self) -> float:
        """حساب مؤشر الاستقرار مع معالجة الأخطاء"""
        try:
            if len(self.equilibrium_history) < 5:
                return 1.0

            recent_changes = []
            for i in range(1, min(6, len(self.equilibrium_history))):
                change = abs(self.equilibrium_history[-i]["C_m"] -
                            self.equilibrium_history[-(i+1)]["C_m"])
                recent_changes.append(change)

            avg_change = np.mean(recent_changes) if recent_changes else 0
            stability = max(0.0, 1.0 - avg_change * 10)
            return float(stability)

        except Exception:
            return 1.0

    def _analyze_equilibrium_trend(self) -> str:
        """تحليل اتجاه التوازن المعرفي"""
        try:
            if len(self.equilibrium_history) < 3:
                return "ثابت"

            recent = [state["C_m"] for state in list(self.equilibrium_history)[-5:]]
            if len(recent) < 2:
                return "ثابت"

            changes = [recent[i] - recent[i-1] for i in range(1, len(recent))]
            avg_change = np.mean(changes) if changes else 0

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

        except Exception:
            return "ثابت"

    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """تحليل شامل للتوازن المعرفي"""
        try:
            current_state = self.equilibrium_history[-1] if self.equilibrium_history else {}

            if self.equilibrium_history:
                c_m_values = [state["C_m"] for state in self.equilibrium_history]
                tau_values = [state["tau"] for state in self.equilibrium_history]
                gamma_values = [state["gamma"] for state in self.equilibrium_history]
            else:
                c_m_values, tau_values, gamma_values = [self.C_m], [self.tau], [self.gamma]

            return {
                "current_equilibrium": current_state,
                "statistical_analysis": {
                    "c_m_mean": float(np.mean(c_m_values)) if c_m_values else 0.5,
                    "c_m_std": float(np.std(c_m_values)) if len(c_m_values) > 1 else 0.0,
                    "tau_mean": float(np.mean(tau_values)) if tau_values else 0.3,
                    "gamma_mean": float(np.mean(gamma_values)) if gamma_values else 0.3,
                    "volatility_index": float(np.std(c_m_values) * 10) if len(c_m_values) > 1 else 0.0,
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
        except Exception as e:
            print(f"⚠️ خطأ في التحليل الشامل: {e}")
            return {"error": "فشل في التحليل الشامل"}

    def _get_most_common_affective_state(self) -> str:
        """الحصول على الحالة العاطفية الأكثر شيوعًا"""
        try:
            if not self.affective_states_log:
                return "متوازن"

            from collections import Counter
            counter = Counter(self.affective_states_log)
            return counter.most_common(1)[0][0]
        except Exception:
            return "متوازن"

    def _calculate_stability_percentage(self) -> float:
        """حساب نسبة الاستقرار في السجل"""
        try:
            if len(self.equilibrium_history) < 2:
                return 100.0

            stable_count = sum(1 for state in self.equilibrium_history
                              if state.get("stability_index", 1.0) > 0.7)
            return (stable_count / len(self.equilibrium_history)) * 100
        except Exception:
            return 100.0

    def _calculate_optimal_balance_percentage(self) -> float:
        """حساب نسبة الوقت في التوازن الأمثل"""
        try:
            if not self.equilibrium_history:
                return 0.0

            optimal_count = sum(1 for state in self.equilibrium_history
                               if 0.4 <= state.get("C_m", 0.5) <= 0.6)
            return (optimal_count / len(self.equilibrium_history)) * 100
        except Exception:
            return 0.0

    def _generate_equilibrium_recommendations(self) -> List[str]:
        """توليد توصيات لتحسين التوازن المعرفي"""
        try:
            recommendations = []
            current_state = self.equilibrium_history[-1] if self.equilibrium_history else {}

            if not current_state:
                return ["لا توجد بيانات كافية للتوصيات"]

            c_m = current_state.get("C_m", 0.5)

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
        except Exception:
            return ["تحسين عام في استراتيجيات التوازن"]


class EmbodiedBodyModel:
    """
    نموذج جسدي محسّن مع معالجة شاملة للأخطاء
    """

    def __init__(self):
        self.energy = 1.0
        self.arousal = 0.1
        self.vitality = 0.8

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

        self.physical_history = deque(maxlen=500)
        self.movement_patterns = defaultdict(int)
        self.fatigue_coefficient = 0.1
        self.recovery_rate = 0.05

        self._record_physical_state("تهيئة أولية")

    def apply_embodied_action(self, action: Dict[str, Any],
                            cognitive_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        تطبيق فعل مجسد مع معالجة الأخطاء
        """
        try:
            cost = action.get("cost", 0.02)
            arousal_delta = action.get("arousal_delta", 0.0)
            physical_intensity = action.get("physical_intensity", 0.3)

            # تحديث الحالة الأساسية
            self.energy = max(0.001, self.energy - cost * (1 + physical_intensity))
            self.arousal = min(1.0, max(0.001, self.arousal + arousal_delta))

            # تحديث الحيوية
            self.vitality = (self.energy * 0.6 + (1 - abs(self.arousal - 0.5)) * 0.4)

            # تحديث التجسيد الجسدي
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

        except Exception as e:
            print(f"⚠️ خطأ في تطبيق الفعل المجسد: {e}")
            return self._get_fallback_state()

    def _update_physical_embodiment(self, cognitive_state: Dict[str, Any],
                                  intensity: float):
        """تحديث التجسيد الجسدي"""
        try:
            C_m = cognitive_state.get("C_m", 0.5)

            if C_m < 0.3:
                self.physical_state.update({
                    "posture": "منقبض", "posture_intensity": 0.8,
                    "movement": "متقطع", "movement_fluidity": 0.3,
                    "expression": "قلق", "expression_intensity": 0.7,
                    "gesture": "فرك اليدين", "gesture_expressiveness": 0.6,
                    "breathing_pattern": "سريع", "muscle_tension": 0.8
                })
            elif C_m < 0.45:
                self.physical_state.update({
                    "posture": "حذر", "posture_intensity": 0.6,
                    "movement": "متريث", "movement_fluidity": 0.5,
                    "expression": "جدي", "expression_intensity": 0.5,
                    "gesture": "طي الذراعين", "gesture_expressiveness": 0.4,
                    "breathing_pattern": "منتظم", "muscle_tension": 0.5
                })
            elif 0.45 <= C_m <= 0.55:
                self.physical_state.update({
                    "posture": "مستقيم", "posture_intensity": 0.5,
                    "movement": "انسجام", "movement_fluidity": 0.8,
                    "expression": "هادئ", "expression_intensity": 0.4,
                    "gesture": "إيماءات طبيعية", "gesture_expressiveness": 0.5,
                    "breathing_pattern": "منتظم", "muscle_tension": 0.3
                })
            elif C_m <= 0.7:
                self.physical_state.update({
                    "posture": "منفتح", "posture_intensity": 0.6,
                    "movement": "نشيط", "movement_fluidity": 0.7,
                    "expression": "متحمس", "expression_intensity": 0.6,
                    "gesture": "إيماءات واسعة", "gesture_expressiveness": 0.7,
                    "breathing_pattern": "عميق", "muscle_tension": 0.4
                })
            else:
                self.physical_state.update({
                    "posture": "متفجر", "posture_intensity": 0.9,
                    "movement": "سريع", "movement_fluidity": 0.9,
                    "expression": "متحمس بشدة", "expression_intensity": 0.8,
                    "gesture": "إيماءات دراماتيكية", "gesture_expressiveness": 0.9,
                    "breathing_pattern": "سريع وعميق", "muscle_tension": 0.6
                })

            # تطبيق شدة النشاط
            for key in ["posture_intensity", "movement_fluidity", "expression_intensity",
                       "gesture_expressiveness"]:
                self.physical_state[key] = min(1.0, self.physical_state[key] * (1 + intensity * 0.5))

        except Exception as e:
            print(f"⚠️ خطأ في تحديث التجسيد الجسدي: {e}")

    def _record_physical_state(self, action_type: str) -> Dict[str, Any]:
        """تسجيل الحالة الجسدية"""
        try:
            state_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action_type": action_type,
                **self.physical_state,
                "energy": self.energy,
                "arousal": self.arousal,
                "vitality": self.vitality
            }

            self.physical_history.append(state_record)

            movement_key = f"{self.physical_state['movement']}_{self.physical_state['posture']}"
            self.movement_patterns[movement_key] += 1

            return {
                "movement_pattern": movement_key,
                "physical_coherence": self._calculate_physical_coherence(),
                "fatigue_level": 1.0 - self.energy,
                "expressiveness_score": self._calculate_expressiveness_score()
            }

        except Exception as e:
            print(f"⚠️ خطأ في تسجيل الحالة الجسدية: {e}")
            return {"error": "فشل في تسجيل الحالة الجسدية"}

    def _calculate_physical_coherence(self) -> float:
        """حساب تماسك الحالة الجسدية"""
        try:
            intensity_avg = np.mean([
                self.physical_state["posture_intensity"],
                self.physical_state["expression_intensity"],
                self.physical_state["gesture_expressiveness"]
            ])

            coherence = 1.0 - abs(intensity_avg - self.energy)
            return max(0.0, min(1.0, coherence))
        except Exception:
            return 0.7

    def _calculate_expressiveness_score(self) -> float:
        """حساب درجة التعبيرية الجسدية"""
        try:
            factors = [
                self.physical_state["expression_intensity"],
                self.physical_state["gesture_expressiveness"],
                self.physical_state["movement_fluidity"],
                self.arousal
            ]

            return float(np.mean(factors))
        except Exception:
            return 0.5

    def _get_fallback_state(self) -> Dict[str, Any]:
        """الحصول على حالة احتياطية في حالة الخطأ"""
        return {
            "posture": "واقف",
            "posture_intensity": 0.5,
            "movement": "هادئ",
            "movement_fluidity": 0.5,
            "expression": "محايد",
            "expression_intensity": 0.5,
            "gesture": "لا شيء",
            "gesture_expressiveness": 0.3,
            "breathing_pattern": "منتظم",
            "muscle_tension": 0.3,
            "energy": 0.8,
            "arousal": 0.3,
            "vitality": 0.7,
            "physical_analysis": {"error": "حالة احتياطية"}
        }

    def get_physical_analysis_report(self) -> Dict[str, Any]:
        """تقرير تحليلي مفصل عن الحالة الجسدية"""
        try:
            if not self.physical_history:
                return {"status": "لا توجد بيانات كافية"}

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
        except Exception as e:
            print(f"⚠️ خطأ في التقرير الجسدي: {e}")
            return {"error": "فشل في إنشاء التقرير الجسدي"}

    def _analyze_expressiveness_trend(self) -> str:
        """تحليل اتجاه التعبيرية الجسدية"""
        try:
            if len(self.physical_history) < 3:
                return "ثابت"

            recent_scores = [state["expression_intensity"] for state in list(self.physical_history)[-5:]]
            if len(recent_scores) < 2:
                return "ثابت"

            avg_score = np.mean(recent_scores)
            prev_scores = [state["expression_intensity"] for state in list(self.physical_history)[-10:-5]]
            prev_avg = np.mean(prev_scores) if prev_scores else avg_score

            if avg_score > prev_avg + 0.1:
                return "متزايد"
            elif avg_score < prev_avg - 0.1:
                return "متناقص"
            else:
                return "مستقر"
        except Exception:
            return "ثابت"

    def _calculate_energy_efficiency(self) -> float:
        """حساب كفاءة استخدام الطاقة"""
        try:
            if not self.physical_history:
                return 1.0

            recent_states = list(self.physical_history)[-10:]
            if not recent_states:
                return 1.0

            total_intensity = sum(state["posture_intensity"] + state["expression_intensity"]
                                 for state in recent_states)
            total_energy_used = sum(1 - state["energy"] for state in recent_states)

            if total_energy_used > 0:
                efficiency = total_intensity / total_energy_used
                return min(2.0, efficiency) / 2.0
            return 1.0
        except Exception:
            return 1.0

    def _calculate_physical_stability(self) -> float:
        """حساب استقرار الحالة الجسدية"""
        try:
            if len(self.physical_history) < 5:
                return 1.0

            recent_states = list(self.physical_history)[-10:]
            changes = []

            for i in range(1, len(recent_states)):
                prev = recent_states[i-1]
                curr = recent_states[i]

                posture_change = abs(prev["posture_intensity"] - curr["posture_intensity"])
                expression_change = abs(prev["expression_intensity"] - curr["expression_intensity"])
                movement_change = abs(prev["movement_fluidity"] - curr["movement_fluidity"])

                avg_change = (posture_change + expression_change + movement_change) / 3
                changes.append(avg_change)

            avg_change = np.mean(changes) if changes else 0
            stability = max(0.0, 1.0 - avg_change * 5)
            return stability
        except Exception:
            return 1.0


class EmbodimentSystem:
    """
    نظام تجسيد متعدد الوسائط مع معالجة محسنة للأخطاء
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

        self.auto_switch_threshold = 0.7
        self.expression_intensity = 0.8

    def set_embodiment_mode(self, mode: str, intensity: float = None) -> Dict[str, Any]:
        """تغيير نمط التجسيد"""
        try:
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

            switch_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "from_mode": previous_mode,
                "to_mode": mode,
                "intensity": self.expression_intensity,
                "type": "mode_switch"
            }

            self.embodiment_history.append(switch_record)
            self.mode_effectiveness[mode]["uses"] += 1

            return {
                "success": True,
                "message": f"✅ تم تغيير نمط التجسيد إلى: {mode}",
                "previous_mode": previous_mode,
                "new_mode": mode,
                "intensity": self.expression_intensity
            }
        except Exception as e:
            print(f"⚠️ خطأ في تغيير نمط التجسيد: {e}")
            return {
                "success": False,
                "message": f"فشل في تغيير نمط التجسيد: {e}"
            }

    def generate_embodied_response(self,
                                 text_response: str,
                                 cognitive_equilibrium: CognitiveEquilibrium,
                                 physical_state: Dict[str, Any],
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        توليد استجابة مجسدة شاملة
        """
        try:
            if context is None:
                context = {}

            embodiment_module = self.available_modes[self.current_mode]

            embodied_output = embodiment_module.express(
                text=text_response,
                cognitive_equilibrium=cognitive_equilibrium,
                physical_state=physical_state,
                context=context,
                intensity=self.expression_intensity
            )

            integration_analysis = self._analyze_embodiment_integration(
                embodied_output, cognitive_equilibrium, physical_state
            )

            response_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": self.current_mode,
                "input_text": text_response,
                "output": embodied_output,
                "integration_analysis": integration_analysis,
                "cognitive_state": cognitive_equilibrium.get_comprehensive_analysis(),
                "physical_state": physical_state,
                "type": "embodied_response"
            }

            self.embodiment_history.append(response_record)
            self._update_mode_effectiveness(integration_analysis)

            return {
                "embodied_response": embodied_output,
                "integration_analysis": integration_analysis,
                "embodiment_mode": self.current_mode,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            print(f"⚠️ خطأ في توليد الاستجابة المجسدة: {e}")
            return self._get_fallback_response(text_response)

    def _analyze_embodiment_integration(self, embodied_output: Dict[str, Any],
                                      cognitive_equilibrium: CognitiveEquilibrium,
                                      physical_state: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل تكامل التجسيد"""
        try:
            C_m = cognitive_equilibrium.C_m

            mode_compatibility = self._calculate_mode_compatibility(
                self.current_mode, C_m, physical_state
            )

            expressive_coherence = self._calculate_expressive_coherence(
                embodied_output, cognitive_equilibrium, physical_state
            )

            # تجنب القسمة على الصفر في الحساب النهائي
            overall_score = (mode_compatibility + expressive_coherence) / 2

            return {
                "mode_compatibility": mode_compatibility,
                "expressive_coherence": expressive_coherence,
                "overall_integration_score": overall_score,
                "recommended_improvements": self._generate_integration_recommendations(
                    mode_compatibility, expressive_coherence
                )
            }
        except Exception as e:
            print(f"⚠️ خطأ في تحليل التكامل: {e}")
            return {
                "mode_compatibility": 0.5,
                "expressive_coherence": 0.5,
                "overall_integration_score": 0.5,
                "recommended_improvements": ["تحسين عام في التكامل"]
            }

    def _calculate_mode_compatibility(self, mode: str, C_m: float,
                                    physical_state: Dict[str, Any]) -> float:
        """حساب توافق نمط التجسيد"""
        try:
            base_compatibility = 0.7

            if mode == "text" and (C_m < 0.3 or C_m > 0.7):
                base_compatibility += 0.2
            elif mode == "avatar" and 0.4 <= C_m <= 0.6:
                base_compatibility += 0.3

            vitality = physical_state.get("vitality", 0.5)
            if mode in ["avatar", "hardware"] and vitality > 0.7:
                base_compatibility += 0.2

            return min(1.0, base_compatibility)
        except Exception:
            return 0.7

    def _calculate_expressive_coherence(self, embodied_output: Dict[str, Any],
                                      cognitive_equilibrium: CognitiveEquilibrium,
                                      physical_state: Dict[str, Any]) -> float:
        """حساب تماسك التعبير"""
        try:
            C_m = cognitive_equilibrium.C_m
            physical_intensity = physical_state.get("expression_intensity", 0.5)

            expected_intensity = abs(C_m - 0.5) * 2
            intensity_coherence = 1.0 - abs(expected_intensity - physical_intensity)

            content_coherence = self._analyze_content_coherence(embodied_output, C_m)

            return (intensity_coherence + content_coherence) / 2
        except Exception:
            return 0.7

    def _analyze_content_coherence(self, embodied_output: Dict[str, Any],
                                 C_m: float) -> float:
        """تحليل تماسك المحتوى"""
        try:
            output_type = embodied_output.get("type", "unknown")

            if output_type == "text" and "content" in embodied_output:
                content = embodied_output["content"]
                if C_m < 0.4 and "😊" in content:
                    return 0.3
                elif C_m > 0.6 and "😰" in content:
                    return 0.3
                else:
                    return 0.8

            return 0.7
        except Exception:
            return 0.7

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
        try:
            integration_score = integration_analysis.get("overall_integration_score", 0.5)
            mode = self.current_mode

            current_stats = self.mode_effectiveness[mode]
            current_success_rate = current_stats["success_rate"]
            current_uses = current_stats["uses"]

            if current_uses > 0:  # تجنب القسمة على الصفر
                new_success_rate = (current_success_rate * (current_uses - 1) + integration_score) / current_uses
                self.mode_effectiveness[mode]["success_rate"] = new_success_rate

        except Exception as e:
            print(f"⚠️ خطأ في تحديث فعالية النمط: {e}")

    def _get_fallback_response(self, text_response: str) -> Dict[str, Any]:
        """الحصول على استجابة احتياطية"""
        return {
            "embodied_response": {
                "type": "text",
                "content": f"🔄 {text_response} [وضع الاستجابة الأساسية]"
            },
            "integration_analysis": {
                "mode_compatibility": 0.5,
                "expressive_coherence": 0.5,
                "overall_integration_score": 0.5,
                "recommended_improvements": ["استخدام الوضع الأساسي بسبب خطأ في النظام"]
            },
            "embodiment_mode": "text",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get_embodiment_system_report(self) -> Dict[str, Any]:
        """تقرير مفصل عن أداء نظام التجسيد"""
        try:
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
        except Exception as e:
            print(f"⚠️ خطأ في تقرير نظام التجسيد: {e}")
            return {"error": "فشل في إنشاء تقرير نظام التجسيد"}

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
        try:
            integration_scores = [
                r["integration_analysis"]["overall_integration_score"]
                for r in self.embodiment_history
                if r["type"] == "embodied_response" and "integration_analysis" in r
            ]

            if not integration_scores:
                return {"status": "لا توجد بيانات كافية"}

            return {
                "average_integration_score": float(np.mean(integration_scores)),
                "integration_consistency": float(np.std(integration_scores)) if len(integration_scores) > 1 else 0.0,
                "improvement_trend": self._analyze_integration_trend(integration_scores),
                "performance_level": self._get_performance_level(np.mean(integration_scores))
            }
        except Exception:
            return {"status": "خطأ في تحليل التكامل"}

    def _analyze_integration_trend(self, scores: List[float]) -> str:
        """تحليل اتجاه أداء التكامل"""
        try:
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
        except Exception:
            return "غير محدد"

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
        try:
            recommendations = []

            for mode, stats in self.mode_effectiveness.items():
                if stats["success_rate"] < 0.6 and stats["uses"] > 5:
                    recommendations.append(f"تحسين أداء نمط {mode} أو تقليل استخدامه")

            integration_analytics = performance["integration_analytics"]
            if integration_analytics.get("average_integration_score", 0) < 0.7:
                recommendations.append("العمل على تحسين تكامل النظام بشكل عام")

            if not recommendations:
                recommendations.append("أداء النظام جيد - الاستمرار في المراقبة")

            return recommendations
        except Exception:
            return ["مراقبة وتحسين أداء النظام العام"]


# وحدات التجسيد الأساسية
class TextEmbodiment:
    def express(self, text, cognitive_equilibrium, physical_state, context, intensity):
        return {
            "type": "text",
            "content": f"📝 {text}",
            "metadata": {
                "intensity": intensity,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

class AvatarEmbodiment:
    def express(self, text, cognitive_equilibrium, physical_state, context, intensity):
        return {
            "type": "avatar",
            "animation": physical_state.get("movement", "neutral"),
            "expression": physical_state.get("expression", "neutral"),
            "text_content": text,
            "metadata": {
                "intensity": intensity,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

class HardwareEmbodiment:
    def express(self, text, cognitive_equilibrium, physical_state, context, intensity):
        return {
            "type": "hardware",
            "commands": [
                {"action": "display", "content": text},
                {"action": "led", "color": "#4CAF50" if cognitive_equilibrium.C_m > 0.5 else "#FF6B6B"}
            ],
            "metadata": {
                "intensity": intensity,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

class ReverseEmbodiment:
    def express(self, text, cognitive_equilibrium, physical_state, context, intensity):
        return {
            "type": "reverse",
            "analysis": {
                "emotional_tone": "محايد",
                "suggested_expression": physical_state.get("expression", "neutral"),
                "text_analysis": f"تحليل النص: {len(text)} حرف"
            },
            "metadata": {
                "intensity": intensity,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

class MultimodalEmbodiment:
    def express(self, text, cognitive_equilibrium, physical_state, context, intensity):
        return {
            "type": "multimodal",
            "outputs": {
                "text": f"📝 {text}",
                "avatar": {
                    "animation": physical_state.get("movement", "neutral"),
                    "expression": physical_state.get("expression", "neutral")
                },
                "analysis": {
                    "cognitive_state": cognitive_equilibrium.C_m,
                    "physical_state": physical_state
                }
            },
            "metadata": {
                "intensity": intensity,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }


class EmbodiedUDMMAgent:
    """
    النسخة النهائية المحسّنة من وكيل UDMM المجسد
    مع معالجة شاملة للأخطاء وتحسين الأداء
    """

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}

        # تهيئة المكونات الأساسية
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
            "system_uptime": time.time(),
            "error_count": 0
        }

        # إعدادات الوكيل
        self.learning_enabled = config.get('learning_enabled', True)
        self.auto_optimization = config.get('auto_optimization', True)

        print("🎉 تم تهيئة النسخة النهائية من وكيل UDMM المجسد بنجاح!")

    def perceive_and_act(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        التفاعل الرئيسي مع الوكيل - مع معالجة محسنة للأخطاء
        """
        start_time = time.time()

        try:
            if context is None:
                context = {}

            # 1. معالجة المدخلات الأساسية
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

            # 4. توليد الرد الأساسي
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
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # 7. تحديث الإحصائيات والسجل
            self._update_interaction_history(final_response)
            self.performance_metrics["total_interactions"] += 1

            return final_response

        except Exception as e:
            self.performance_metrics["error_count"] += 1
            print(f"⚠️ خطأ جسيم في التفاعل: {e}")
            return self._get_error_response(user_input, str(e))

    def _process_user_input(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """معالجة مدخلات المستخدم"""
        try:
            return {
                "text": user_input,
                "length": len(user_input),
                "complexity": min(1.0, len(user_input) / 500),
                "detected_intent": "general",
                "emotional_tone": "neutral"
            }
        except Exception:
            return {"text": user_input, "length": len(user_input), "complexity": 0.5}

    def _calculate_goal_alignment(self, processed_input: Dict[str, Any],
                                context: Dict[str, Any]) -> float:
        """حساب محاذاة الأهداف"""
        try:
            base_alignment = 0.3
            complexity = processed_input.get("complexity", 0.5)
            if complexity > 0.7:
                base_alignment += 0.3
            return min(1.0, base_alignment)
        except Exception:
            return 0.3

    def _calculate_reality_tension(self, processed_input: Dict[str, Any],
                                 context: Dict[str, Any]) -> float:
        """حساب توتر الواقع"""
        try:
            base_tension = 0.3
            complexity = processed_input.get("complexity", 0.5)
            if complexity > 0.8:
                base_tension += 0.4
            return min(1.0, base_tension)
        except Exception:
            return 0.3

    def _generate_base_response(self, processed_input: Dict[str, Any],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """توليد رد أساسي"""
        return {
            "text": "هذا رد من النسخة النهائية المحسّنة للوكيل المجسد.",
            "type": "informational",
            "confidence": 0.8
        }

    def _calculate_system_load(self) -> float:
        """حساب حمل النظام الحالي"""
        try:
            history_size = len(self.interaction_history)
            return min(1.0, history_size / 1000)
        except Exception:
            return 0.1

    def _assess_response_quality(self, embodied_response: Dict[str, Any]) -> float:
        """تقييم جودة الاستجابة"""
        try:
            integration_score = embodied_response.get(
                "integration_analysis", {}
            ).get("overall_integration_score", 0.5)

            return integration_score
        except Exception:
            return 0.5

    def _update_interaction_history(self, response: Dict[str, Any]):
        """تحديث سجل التفاعلات"""
        try:
            self.interaction_history.append(response)

            quality = response["performance_metrics"]["response_quality"]
            if quality > 0.7:
                self.performance_metrics["successful_embodiments"] += 1

            # تحديث متوسط الجودة مع تجنب القسمة على الصفر
            total = self.performance_metrics["total_interactions"]
            if total > 0:
                current_avg = self.performance_metrics["average_response_quality"]
                new_avg = (current_avg * (total - 1) + quality) / total
                self.performance_metrics["average_response_quality"] = new_avg

        except Exception as e:
            print(f"⚠️ خطأ في تحديث سجل التفاعل: {e}")

    def _get_error_response(self, user_input: str, error: str) -> Dict[str, Any]:
        """الحصول على استجابة خطأ"""
        return {
            "input_analysis": {"text": user_input, "error": True},
            "base_response": {
                "text": f"عذرًا، حدث خطأ في المعالجة: {error}",
                "type": "error",
                "confidence": 0.1
            },
            "embodied_response": {
                "type": "text",
                "content": f"⚠️ عذرًا، حدث خطأ: {error}"
            },
            "cognitive_analysis": self.cognitive_equilibrium.get_comprehensive_analysis(),
            "physical_analysis": self.embodied_body._get_fallback_state(),
            "performance_metrics": {
                "processing_time": 0.0,
                "system_load": 1.0,
                "response_quality": 0.1
            },
            "interaction_id": self.performance_metrics["total_interactions"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": True
        }

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """تقرير شامل عن أداء الوكيل"""
        try:
            cognitive_report = self.cognitive_equilibrium.get_comprehensive_analysis()
            physical_report = self.embodied_body.get_physical_analysis_report()
            embodiment_report = self.embodiment_system.get_embodiment_system_report()

            uptime = time.time() - self.performance_metrics["system_uptime"]

            # حساب نسبة النجاح مع تجنب القسمة على الصفر
            total_interactions = self.performance_metrics["total_interactions"]
            success_rate = 0.0
            if total_interactions > 0:
                success_rate = (self.performance_metrics["successful_embodiments"] / total_interactions) * 100

            return {
                "agent_overview": {
                    "type": "EmbodiedUDMMAgent",
                    "uptime_seconds": uptime,
                    "learning_enabled": self.learning_enabled,
                    "auto_optimization": self.auto_optimization
                },
                "performance_summary": {
                    **self.performance_metrics,
                    "success_rate_percentage": success_rate
                },
                "cognitive_performance": cognitive_report,
                "physical_performance": physical_report,
                "embodiment_performance": embodiment_report,
                "system_health": {
                    "overall_health": self._calculate_system_health(),
                    "recommendations": self._generate_system_recommendations(),
                    "maintenance_suggestions": self._generate_maintenance_suggestions()
                }
            }
        except Exception as e:
            print(f"⚠️ خطأ في التقرير الشامل: {e}")
            return {"error": "فشل في إنشاء التقرير الشامل"}

    def _calculate_system_health(self) -> float:
        """حساب الصحة العامة للنظام"""
        try:
            factors = []

            # الصحة المعرفية
            cognitive_analysis = self.cognitive_equilibrium.get_comprehensive_analysis()
            cognitive_health = cognitive_analysis.get("performance_metrics", {}).get("optimal_balance_percentage", 0) / 100
            factors.append(cognitive_health)

            # الصحة الجسدية
            physical_report = self.embodied_body.get_physical_analysis_report()
            vitality = physical_report.get("vitality_metrics", {}).get("vitality_score", 0.5)
            factors.append(vitality)

            # صحة التجسيد
            embodiment_report = self.embodiment_system.get_embodiment_system_report()
            integration_analytics = embodiment_report.get("integration_analytics", {})
            embodiment_health = integration_analytics.get("average_integration_score", 0.5)
            factors.append(embodiment_health)

            # معدل النجاح
            total_interactions = self.performance_metrics["total_interactions"]
            if total_interactions > 0:
                success_rate = self.performance_metrics["successful_embodiments"] / total_interactions
                factors.append(success_rate)

            return float(np.mean(factors)) if factors else 0.7

        except Exception:
            return 0.7

    def _generate_system_recommendations(self) -> List[str]:
        """توليد توصيات للنظام"""
        recommendations = []

        # تحليل الصحة العامة
        system_health = self._calculate_system_health()
        if system_health < 0.6:
            recommendations.append("تحسين الصحة العامة للنظام")

        # تحليل معدل الأخطاء
        error_count = self.performance_metrics["error_count"]
        if error_count > 10:
            recommendations.append("فحص وعلاج الأخطاء المتكررة")

        if not recommendations:
            recommendations.append("الأداء جيد - الاستمرار في المراقبة")

        return recommendations

    def _generate_maintenance_suggestions(self) -> List[str]:
        """توليد اقتراحات صيانة"""
        uptime = time.time() - self.performance_metrics["system_uptime"]
        suggestions = []

        if uptime > 86400:  # أكثر من يوم
            suggestions.append("إعادة تشغيل النظام للتحسين الأداء")

        if self.performance_metrics["total_interactions"] > 1000:
            suggestions.append("تنظيف السجلات القديمة لتحسين الأداء")

        if not suggestions:
            suggestions.append("لا توجد حاجة للصيانة الفورية")

        return suggestions

    def set_embodiment_mode(self, mode: str, intensity: float = None) -> Dict[str, Any]:
        """تغيير نمط التجسيد"""
        return self.embodiment_system.set_embodiment_mode(mode, intensity)

    def get_available_embodiment_modes(self) -> List[str]:
        """الحصول على أنماط التجسيد المتاحة"""
        return list(self.embodiment_system.available_modes.keys())


# مثال الاستخدام والتشغيل
def main():
    """دالة التشغيل الرئيسية"""
    print("🚀 بدء تشغيل النسخة النهائية من وكيل UDMM المجسد...")

    # إنشاء الوكيل
    agent = EmbodiedUDMMAgent({
        'initial_C_m': 0.5,
        'learning_enabled': True,
        'auto_optimization': True
    })

    # اختبار التفاعلات
    test_inputs = [
        "مرحبًا، كيف حالك اليوم؟",
        "أحتاج مساعدة في فهم موضوع معين",
        "أشعر بالقلق من المستقبل",
        "هذا مثير للاهتمام! أريد تعلم المزيد"
    ]

    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n{'='*50}")
        print(f"التفاعل {i}: {user_input}")
        print(f"{'='*50}")

        response = agent.perceive_and_act(user_input)

        # عرض النتائج
        embodied_response = response.get("embodied_response", {})
        if embodied_response.get("type") == "text":
            print(f"الرد: {embodied_response.get('content', '')}")

        cognitive_state = response.get("cognitive_analysis", {})
        print(f"الحالة المعرفية: {cognitive_state.get('affective_state', 'غير معروف')} (C_m: {cognitive_state.get('C_m', 0):.2f})")

        physical_state = response.get("physical_analysis", {})
        print(f"الحالة الجسدية: {physical_state.get('expression', 'غير معروف')} - {physical_state.get('movement', 'غير معروف')}")

    # عرض التقرير النهائي
    print(f"\n{'='*50}")
    print("التقرير النهائي للأداء")
    print(f"{'='*50}")

    report = agent.get_comprehensive_report()
    performance = report.get("performance_summary", {})
    print(f"إجمالي التفاعلات: {performance.get('total_interactions', 0)}")
    print(f"معدل النجاح: {performance.get('success_rate_percentage', 0):.1f}%")
    print(f"متوسط جودة الرد: {performance.get('average_response_quality', 0):.2f}")

    system_health = report.get("system_health", {})
    print(f"صحة النظام: {system_health.get('overall_health', 0):.2f}")


if __name__ == "__main__":
    main()
