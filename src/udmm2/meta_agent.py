import time
import math
from typing import Dict

from .body import DigitalBody
from .affect import AffectiveSystem
from .symbolic import SymbolicStore
from .memory import MemorySystem
from .attractor import VirtualAttractor

class MetaAgent:
    def __init__(self, body: DigitalBody, affect: AffectiveSystem, symbols: SymbolicStore, memory: MemorySystem, attractor: VirtualAttractor):
        self.body = body
        self.affect = affect
        self.symbols = symbols
        self.memory = memory
        self.attractor = attractor

        self.C = 0.5
        self.IT = 0.0
        self.KL_divergence = 0.0
        self.temporal_depth = 3

        self.aar_history = []

    def compute_prediction_error(self, expectation: str, reality: str, context_affect: float) -> float:
        semantic_pe = abs(len(reality) - len(expectation)) / max(1, len(expectation))
        affective_pe = abs(context_affect - self.affect.A_r)

        pe = (semantic_pe * 0.6 + affective_pe * 0.4)
        self.IT = pe

        self.attractor.update_attractor_strength(pe)
        self.affect.update_affect(pe * 0.3, valence=0.0)

        return pe

    def compute_KL_divergence(self, prior_state: Dict, current_state: Dict) -> float:
        try:
            prior_vec = [
                prior_state.get("body", [0.5, 0.5, 0.0])[0],
                prior_state.get("body", [0.5, 0.5, 0.0])[1],
                prior_state.get("affect", {}).get("A_r", 0.5)
            ]
            current_vec = [
                current_state.get("body", [0.5, 0.5, 0.0])[0],
                current_state.get("body", [0.5, 0.5, 0.0])[1],
                current_state.get("affect", {}).get("A_r", 0.5)
            ]

            prior_sum = sum(prior_vec)
            current_sum = sum(current_vec)

            if prior_sum == 0 or current_sum == 0:
                return 0.0

            prior_norm = [p/prior_sum for p in prior_vec]
            current_norm = [c/current_sum for c in current_vec]

            kl = 0.0
            for i in range(len(prior_norm)):
                if prior_norm[i] > 0 and current_norm[i] > 0:
                    kl += prior_norm[i] * math.log(prior_norm[i] / current_norm[i])

            self.KL_divergence = max(0.0, kl)
            return self.KL_divergence

        except (ZeroDivisionError, ValueError) as e:
            print(f"Debug: Handled KL-divergence calculation error: {e}")
            return 0.0

    def relabel(self, old_symbol: str, new_symbol: str, confidence: float = 0.7):
        if old_symbol in self.symbols.symbols:
            self.symbols.rename(old_symbol, new_symbol)
            self.memory.add_knowledge(
                f"relabel_{old_symbol}_to_{new_symbol}",
                f"إعادة تسمية من {old_symbol} إلى {new_symbol}",
                confidence,
                "AAR"
            )
            self.affect.decrease(0.15)
            self.aar_history.append({
                "type": "relabel",
                "old": old_symbol,
                "new": new_symbol,
                "timestamp": time.time()
            })

    def reanchor(self, intensity: float = 0.8):
        self.body.stabilize()
        self.affect.A_r *= (1 - intensity * 0.5)
        self.affect.valence *= 0.5
        current_state = self._get_current_state()
        self.attractor.current_basin, _ = self.attractor.compute_attractor_force(current_state)
        self.aar_history.append({
            "type": "reanchor",
            "intensity": intensity,
            "timestamp": time.time()
        })

    def redistribute_affect(self, redistribution_factor: float = 0.8):
        total_affect = sum(data["affect"] for data in self.symbols.symbols.values())
        num_symbols = max(1, len(self.symbols.symbols))
        target_affect = total_affect / num_symbols

        for symbol, data in self.symbols.symbols.items():
            current_affect = data["affect"]
            new_affect = current_affect * (1 - redistribution_factor) + target_affect * redistribution_factor
            self.symbols.symbols[symbol]["affect"] = new_affect

        self.affect.A_r *= 0.6
        affective_variance = self.affect.compute_affective_variance()
        self.aar_history.append({
            "type": "redistribute_affect",
            "factor": redistribution_factor,
            "variance_reduction": affective_variance,
            "timestamp": time.time()
        })

    def _get_current_state(self) -> Dict:
        return {
            "energy": self.body.energy,
            "arousal": self.body.arousal,
            "affect": self.affect.A_r
        }

    def intervene(self, current_state: Dict, prediction_error: float):
        intervention_triggered = False
        if prediction_error > 0.6:
            self.redistribute_affect(0.7)
            intervention_triggered = True
        if self.IT > 0.7:
            self.reanchor(0.9)
            intervention_triggered = True
        if self.KL_divergence > 0.5:
            highly_charged = [(s, d) for s, d in self.symbols.symbols.items() if d["affect"] > 0.8]
            if highly_charged:
                symbol_to_relabel = highly_charged[0][0]
                new_name = f"integrated_{symbol_to_relabel}_{int(time.time())}"
                self.relabel(symbol_to_relabel, new_name, 0.6)
                intervention_triggered = True
        return intervention_triggered
