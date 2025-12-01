import math
from typing import Dict, Tuple

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
        self.attractor_strength = max(0.3, min(0.9,
            self.attractor_strength + prediction_error * 0.1))
