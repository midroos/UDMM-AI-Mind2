from collections import defaultdict
import time
from typing import List

class SymbolicStore:
    def __init__(self):
        self.symbols = defaultdict(lambda: {
            "affect": 0.1,
            "frequency": 0,
            "last_accessed": time.time(),
            "semantic_links": defaultdict(float)
        })
        self.symbol_network = defaultdict(dict)

    def reinforce(self, symbol: str, value: float, context: str = ""):
        self.symbols[symbol]["affect"] = min(1.0, self.symbols[symbol]["affect"] + value)
        self.symbols[symbol]["frequency"] += 1
        self.symbols[symbol]["last_accessed"] = time.time()

        if context:
            self.symbols[symbol]["semantic_links"][context] += value

    def weaken(self, symbol: str, value: float):
        self.symbols[symbol]["affect"] = max(0.0, self.symbols[symbol]["affect"] - value)

    def rename(self, old: str, new: str):
        if old in self.symbols:
            self.symbols[new] = self.symbols.pop(old)

    def get_activation_level(self, symbol: str) -> float:
        base = self.symbols[symbol]["affect"]
        recency = 1.0 / (time.time() - self.symbols[symbol]["last_accessed"] + 1)
        return base * 0.7 + recency * 0.3

    def find_related_symbols(self, symbol: str, threshold: float = 0.3) -> List[str]:
        related = []
        if symbol in self.symbols:
            for linked_symbol, strength in self.symbols[symbol]["semantic_links"].items():
                if strength > threshold:
                    related.append(linked_symbol)
        return related
