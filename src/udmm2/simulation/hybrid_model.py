import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

class HybridUDMMSimulation:
    """
    نموذج حاسوبي للنظام الديناميكي الهجين المستمر-المنفصل
    ضمن إطار النموذج الديناميكي الموحد للعقل (UDMM)

    هذه الفئة تحاكي الاقتران ثنائي الاتجاه بين التطور المستمر
    للمتغيرات المعرفية والتحولات المنفصلة بين الحالات المعرفية الكلية.
    """

    def __init__(self, params: Dict[str, float]):
        """
        تهيئة المحاكاة بمجموعة معطاة من المعاملات.
        """
        self.params = params
        self.states: List[str] = ['روتين', 'إبداعي', 'قلق', 'مُجبر']
        self.num_states: int = len(self.states)

        # مصفوفة الانتقال الأساسية (مهيمن عليها الزمن الأسبوعي)
        self.base_transition_matrix: np.ndarray = np.array([
            [0.90, 0.05, 0.04, 0.01],  # من روتين
            [0.10, 0.80, 0.05, 0.05],  # من إبداعي
            [0.10, 0.10, 0.75, 0.05],  # من قلق
            [0.15, 0.05, 0.10, 0.70]   # من مُجبر
        ])

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """يضمن أن مجموع كل صف في مصفوفة الانتقال يساوي 1."""
        return matrix / matrix.sum(axis=1, keepdims=True)

    def _continuous_system(self, t: float, y: np.ndarray, p: Dict[str, float]) -> List[float]:
        """
        يحدد نظام المعادلات التفاضلية للمتغيرات المستمرة.
        """
        IT, M_affect, C, Agency, alpha, beta, gamma = y

        dC_dt_val = p['delta_C'] * M_affect * (1 - C) - p['epsilon_C'] * C + p['theta_C'] * IT

        dIT_dt = p['eta'] * IT * (1 - IT) - p['alpha_IT'] * IT * C
        dM_dt = p['beta_M'] * IT * (1 - M_affect) - p['gamma_M'] * M_affect
        dAgency_dt = p['zeta_A'] * dC_dt_val * (1 - Agency)

        # ديناميكيات أوزان القصد الشمولي
        d_alpha_dt = p['k_alpha'] * (p['structural_need'] - alpha)
        d_beta_dt = p['k_beta'] * (p['phenomenal_need'] - beta)
        d_gamma_dt = p['k_gamma'] * (p['symbolic_need'] - gamma)

        return [dIT_dt, dM_dt, dC_dt_val, dAgency_dt, d_alpha_dt, d_beta_dt, d_gamma_dt]

    def _calculate_proxy_deviation(self, holistic_intent: tuple) -> float:
        """يحسب الانحراف بالوكالة (ثيتا) من متجه القصد."""
        alpha, beta, gamma = holistic_intent
        # تبسيط: الانحراف يزداد مع هيمنة القصد الرمزي على البنيوي
        return np.arctan2(gamma, alpha) * 180 / np.pi

    def update_transition_matrix(self, IT: float, holistic_intent: tuple) -> np.ndarray:
        """
        تعديل من أعلى إلى أسفل: تحديث مصفوفة انتقال ماركوف بناءً على المتغيرات المستمرة.
        """
        matrix = self.base_transition_matrix.copy()
        alpha, beta, gamma = holistic_intent

        # 1. تعديل التوتر المعلوماتي: ارتفاعه يزيد عدم الاستقرار واحتمال الانتقال
        it_factor = (1 / (1 + np.exp(-10 * (IT - 0.5)))) # مفتاح سيجمويد
        matrix += (np.full_like(matrix, 0.05) - np.diag([0.2, 0.2, 0.2, 0.2])) * it_factor

        # 2. تعديل القصد الشمولي
        # القصد الظاهراتي (بيتا) يعزز الانتقال إلى 'إبداعي'
        matrix[:, 1] += beta * 0.1
        # القصد الرمزي (غاما) يعزز الانتقال إلى 'مُجبر'
        matrix[:, 3] += gamma * 0.08

        # 3. تعديل الانحراف بالوكالة: الانحراف العالي يعزز الانتقال إلى 'قلق'
        theta = self._calculate_proxy_deviation(holistic_intent)
        if theta > 60:
            matrix[:, 2] += (theta / 90.0) * 0.15

        return self._normalize_matrix(matrix)

    def update_continuous_params(self, discrete_state: int) -> Dict[str, float]:
        """
        تأثير من أسفل إلى أعلى: تحديث معاملات الديناميكيات المستمرة بناءً على الحالة المنفصلة.
        """
        p = self.params.copy()
        if self.states[discrete_state] == 'قلق':
            p['eta'] *= 1.2  # التوتر المعلوماتي ينمو أسرع
            p['zeta_A'] *= 0.7 # الفاعلية تثبط
        elif self.states[discrete_state] == 'إبداعي':
            p['theta_C'] *= 1.3 # للتوتر المعلوماتي تأثير أقوى على الوعي
            p['k_beta'] *= 1.5   # القصد الظاهراتي يعزز
        elif self.states[discrete_state] == 'مُجبر':
            p['k_gamma'] *= 1.5  # القصد الرمزي يعزز
        return p

    def run_simulation(self, initial_y: List[float], duration: int = 100, dt: float = 0.1) -> Dict[str, np.ndarray]:
        """
        تشغيل المحاكاة الهجينة الكاملة.
        """
        num_steps = int(duration / dt)
        y = np.array(initial_y)

        # سجلات لتخزين النتائج
        history: Dict[str, np.ndarray] = {
            'time': np.linspace(0, duration, num_steps),
            'continuous': np.zeros((num_steps, len(y))),
            'discrete': np.zeros(num_steps, dtype=int)
        }

        discrete_state = 0  # البدء في حالة 'روتين'

        for i in range(num_steps):
            # تخزين الحالة الحالية
            history['continuous'][i, :] = y
            history['discrete'][i] = discrete_state

            # --- الاقتران ثنائي الاتجاه ---
            # 1. من أعلى إلى أسفل: تحديث مصفوفة ماركوف من الحالة المستمرة
            IT, _, _, _, alpha, beta, gamma = y
            holistic_intent = (alpha, beta, gamma)
            transition_matrix = self.update_transition_matrix(IT, holistic_intent)

            # 2. أخذ عينة للحالة المنفصلة التالية
            discrete_state = np.random.choice(self.num_states, p=transition_matrix[discrete_state, :])

            # 3. من أسفل إلى أعلى: تحديث معاملات المستمر من الحالة المنفصلة الجديدة
            current_params = self.update_continuous_params(discrete_state)

            # 4. تطوير النظام المستمر لخطوة واحدة (طريقة أويلر للتبسيط)
            derivatives = self._continuous_system(i * dt, y, current_params)
            y += np.array(derivatives) * dt

            # تقييم القيم لتكون ضمن [0, 1] للاستقرار
            y = np.clip(y, 0, 1)
            # تطبيع متجه القصد
            y[4:] /= np.sum(y[4:])

        return history

    def plot_results(self, history: Dict[str, np.ndarray]) -> plt.Figure:
        """
        تصور نتائج المحاكاة.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # رسم المتغيرات المستمرة
        cont = history['continuous']
        time = history['time']
        ax1.plot(time, cont[:, 0], label='التوتر المعلوماتي (IT)', lw=2)
        ax1.plot(time, cont[:, 2], label='الوعي (C)', lw=2)
        ax1.plot(time, cont[:, 3], label='الفاعلية', lw=2)
        ax1.set_title('تطور المتغيرات المعرفية المستمرة')
        ax1.set_ylabel('مستوى التنشيط')
        ax1.legend()
        ax1.grid(True)

        # رسم تحولات الحالة المنفصلة
        disc = history['discrete']
        ax2.plot(time, disc, drawstyle='steps-post', label='الحالة المعرفية المنفصلة', color='black', lw=2)
        ax2.set_yticks(range(self.num_states))
        ax2.set_yticklabels(self.states)
        ax2.set_title('التحولات بين الحالات المعرفية الكلية المنفصلة')
        ax2.set_xlabel('الزمن (وحدات اعتباطية)')
        ax2.set_ylabel('الحالة')
        ax2.grid(True)

        plt.tight_layout()
        return fig

# --- تنفيذ مثال ---
def main():
    # تعريف المعاملات الأساسية للمحاكاة
    base_params = {
        'eta': 0.3, 'alpha_IT': 0.5, 'beta_M': 0.4, 'gamma_M': 0.3,
        'delta_C': 0.6, 'epsilon_C': 0.4, 'theta_C': 0.25, 'zeta_A': 0.5,
        'k_alpha': 0.05, 'k_beta': 0.05, 'k_gamma': 0.05,
        'structural_need': 0.4, 'phenomenal_need': 0.4, 'symbolic_need': 0.2
    }

    # تعريف الشروط الأولية: [IT, M_affect, C, Agency, alpha, beta, gamma]
    initial_conditions_y = [0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.2]

    # إنشاء وتشغيل المحاكاة
    simulation = HybridUDMMSimulation(base_params)
    simulation_history = simulation.run_simulation(initial_conditions_y)

    # تصور النتائج
    fig = simulation.plot_results(simulation_history)
    plt.show()

if __name__ == '__main__':
    main()