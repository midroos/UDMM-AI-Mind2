import numpy as np
import time as os_time

class UDMM_Complete:
    """
    محاكاة مبسطة لديناميكيات العقل الموحد (UDMM).
    يحاكي هذا النموذج ثلاثة مكونات رئيسية:
    - التوتر المعلوماتي (IT): يقيس درجة الغموض أو التعقيد في المدخلات.
    - الزمن الإدراكي (Time): يمثل تدفق الزمن الذاتي للوكيل، والذي يتباطأ مع زيادة التوتر.
    - السياسة (Policy): تحدد استراتيجية المعالجة الحالية (e.g., reactive, symbolic).
    """

    def __init__(self, initial_it=0.1, time_step_base=0.1, it_increase_factor=0.2, it_decay=0.95):
        """
        إعداد الحالة الأولية للمحاكاة.
        """
        self.it = initial_it  # InfoTension
        self.time = 0.0  # Perceptual Time
        self.policy = "reactive"  # Initial policy
        self.state_history = []

        # Parameters
        self._time_step_base = time_step_base
        self._it_increase_factor = it_increase_factor
        self._it_decay = it_decay

        self.last_update_timestamp = os_time.time()

        # حفظ الحالة الأولية
        self._log_state()

    def _log_state(self):
        """يسجل الحالة الحالية في الـ history."""
        state = {
            "timestamp": self.last_update_timestamp,
            "IT": self.it,
            "time": self.time,
            "policy": self.policy
        }
        self.state_history.append(state)
        return state

    def update_state(self, sensory_input: dict, action: str = None):
        """
        تحديث الحالة بناءً على المدخلات الحسية والفعل الناتج.
        """
        # 1. تحديث التوتر المعلوماتي (IT)
        # يزداد التوتر مع وجود مدخلات نصية جديدة، ويقل مع مرور الوقت أو عند اتخاذ إجراء.
        if sensory_input and sensory_input.get("text"):
            # يزداد التوتر بناءً على طول وتعقيد المدخل (تقدير بسيط)
            input_complexity = len(sensory_input.get("text", "").split()) / 10.0
            self.it += self._it_increase_factor * (1 + input_complexity)

        if action:
            # يقل التوتر عند إنتاج استجابة (فعل)
            self.it *= (1 - self._it_increase_factor)

        # تطبيق اضمحلال طبيعي للتوتر مع مرور الوقت
        self.it *= self._it_decay
        self.it = max(0.05, self.it) # Ensure IT doesn't go to zero

        # 2. تحديث الزمن الإدراكي
        # يتباطأ الزمن الإدراكي (خطوة زمنية أصغر) كلما زاد التوتر
        time_step = self._time_step_base / (1 + self.it)
        self.time += time_step

        # 3. تحديث السياسة
        # تتغير السياسة بناءً على عتبات التوتر
        if self.it > 0.7:
            self.policy = "symbolic_abstract"
        elif self.it > 0.4:
            self.policy = "symbolic_deliberative"
        else:
            self.policy = "reactive"

        self.last_update_timestamp = os_time.time()

        return self._log_state()

    def step(self, sensory_input: dict, action: str = None):
        """
        ينفذ خطوة واحدة من المحاكاة.

        Args:
            sensory_input (dict): المدخلات الحسية (e.g., {"text": "..."}).
            action (str, optional): الفعل أو الاستجابة التي ينتجها الوكيل.

        Returns:
            dict: الحالة الجديدة للمحاكاة.
        """
        return self.update_state(sensory_input, action)

    def get_current_state(self):
        """
        يسترجع آخر حالة مسجلة.
        """
        return self.state_history[-1] if self.state_history else {}
