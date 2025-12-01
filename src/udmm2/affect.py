from collections import deque, defaultdict
import numpy as np
import time

class AffectiveSystem:
    def __init__(self):
        self.A_r = 0.2
        self.affective_history = deque(maxlen=50)
        self.decay_rate = 0.05
        self.affective_priming = defaultdict(float)
        self.valence = 0.0

    def update_affect(self, stimulus_strength: float, valence: float):
        self.A_r = min(1.0, max(0.0, self.A_r + stimulus_strength))
        self.valence = valence
        self.affective_history.append((self.A_r, self.valence, time.time()))

    def decrease(self, amount: float):
        self.A_r = max(0.0, self.A_r - amount)

    def natural_decay(self):
        self.A_r *= (1 - self.decay_rate)
        self.valence *= 0.9

    def get_affective_trajectory(self):
        return [affect[0] for affect in list(self.affective_history)]

    def compute_affective_variance(self):
        if len(self.affective_history) < 2:
            return 0.0
        affects = [a[0] for a in self.affective_history]
        return np.var(affects)
