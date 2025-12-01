from collections import deque, defaultdict
import uuid
import time
from typing import Dict

class MemorySystem:
    def __init__(self):
        self.episodic = deque(maxlen=500)
        self.semantic = defaultdict(lambda: {"value": "", "confidence": 0.5, "sources": []})
        self.working_memory = deque(maxlen=7)
        self.retrieval_threshold = 0.3

    def add_episode(self, user_msg: str, agent_msg: str, state: Dict, affective_state: float):
        episode = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "user": user_msg,
            "agent": agent_msg,
            "state": state,
            "affect": affective_state,
            "salience": self._compute_salience(user_msg, affective_state)
        }
        self.episodic.append(episode)
        self.working_memory.append(episode)

    def add_knowledge(self, key: str, value: str, confidence: float = 0.5, source: str = ""):
        self.semantic[key] = {
            "value": value,
            "confidence": confidence,
            "sources": [source] if source else []
        }

    def retrieve_contextual(self, query: str, affective_context: float, recency_weight: float = 0.3) -> Dict:
        results = {
            "episodic": [],
            "semantic": {},
            "working_memory": list(self.working_memory)
        }

        for episode in self.episodic:
            relevance = self._compute_relevance(episode, query, affective_context)
            if relevance > self.retrieval_threshold:
                results["episodic"].append({
                    "episode": episode,
                    "relevance": relevance
                })

        for key, data in self.semantic.items():
            if query in key or query in data["value"]:
                results["semantic"][key] = data

        return results

    def _compute_salience(self, message: str, affect: float) -> float:
        length_factor = min(len(message) / 100, 1.0)
        emotional_factor = affect
        return (length_factor + emotional_factor) / 2

    def _compute_relevance(self, episode: Dict, query: str, affective_context: float) -> float:
        text_similarity = 1.0 if query in episode["user"] else 0.3
        affective_similarity = 1.0 - abs(episode["affect"] - affective_context)
        recency = 1.0 / (time.time() - episode["timestamp"] + 1)

        return (text_similarity * 0.4 + affective_similarity * 0.4 + recency * 0.2)
