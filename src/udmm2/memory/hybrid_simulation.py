# -*- coding: utf-8 -*-
"""
الملحق أ: المحاكاة الكاملة للذاكرة الموزعة في الإطار الهجين (UDMM)
المؤلف: محمد أحمد عيدروس
التاريخ: أكتوبر 2025
"""

import numpy as np

class HybridUDMMSimulation:
    def __init__(self, steps=2000, dt=0.01):
        self.steps = steps
        self.dt = dt

        # المعاملات الديناميكية
        self.alpha_mem = 0.4   # معدل الترميز الداخلي
        self.beta_mem = 0.3    # معدل النسيان الداخلي
        self.gamma_mem = 0.35  # بناء السقالات البيئية
        self.delta_mem = 0.25  # تضاؤل السقالات
        self.epsilon_mem = 0.3 # التكامل الزمني
        self.zeta_mem = 0.2    # التشظي الزمني
        self.eta_mem = 0.15    # تأثير الذاكرة على التوتر المعلوماتي

        # تهيئة المتغيرات
        self.IT = 0.8
        self.M_affect = 0.6
        self.C = 0.7
        self.Agency = 0.7
        self.M_internal = 0.3
        self.M_env = 0.4
        self.M_temp = 0.5

        # سجلات التتبع
        self.history = {k: [] for k in
                        ["IT", "M_affect", "C", "Agency",
                         "M_internal", "M_env", "M_temp"]}

    def step(self):
        """خطوة زمنية واحدة وفق معادلات UDMM"""
        dM_internal = self.alpha_mem * self.C * (1 - self.M_internal) - \
                      self.beta_mem * self.M_internal * self.IT

        dM_env = self.gamma_mem * self.IT * (1 - self.M_env) - \
                 self.delta_mem * self.M_env * (1 - self.C)

        dM_temp = self.epsilon_mem * self.M_affect * (1 - self.M_temp) - \
                  self.zeta_mem * self.M_temp * (1 - self.Agency)

        # تحديث الذاكرة
        self.M_internal += dM_internal * self.dt
        self.M_env += dM_env * self.dt
        self.M_temp += dM_temp * self.dt

        # تحديث التوتر المعلوماتي
        self.IT -= self.eta_mem * self.M_internal * self.IT * self.dt

        # حفظ البيانات
        for k in self.history.keys():
            self.history[k].append(getattr(self, k))
            if len(self.history[k]) > self.steps:
                self.history[k].pop(0)

    def run(self):
        for _ in range(self.steps):
            self.step()

        return self.history
