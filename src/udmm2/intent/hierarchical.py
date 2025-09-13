# hierarchical.py
class HierarchicalIntentManager:
    def __init__(self):
        self.global_intent = {}

    def compute_global_intent(self, attractors: dict):
        total = sum(attractors.values()) or 1.0
        self.global_intent = {k: v / total for k, v in attractors.items()}
        return self.global_intent

    def decompose(self, global_intent):
        subs = []
        for k, v in sorted(global_intent.items(), key=lambda x: -x[1]):
            if v > 0.12:
                subs.append({"goal": k, "priority": v})
        return subs
