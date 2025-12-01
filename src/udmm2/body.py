import random

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
        self.arousal = max(0.05, min(1.0,
            self.arousal + random.uniform(-0.03, 0.03) + sensory_input * 0.1))

        energy_drain = self.fatigue_rate * (1 + self.arousal * 0.5)
        self.energy = max(0.1, self.energy - energy_drain + self.homeostatic_drift)

    def stabilize(self):
        self.arousal *= 0.6
        self.energy = min(1.0, self.energy + 0.1)

    def get_body_state_vector(self):
        return [self.energy, self.arousal, self.sensory_input]
