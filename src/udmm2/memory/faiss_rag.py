# faiss_rag.py
import os, json, numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class FaissRAG:
    def __init__(self, model_name, index_path="data/faiss.index", meta_path="data/faiss_meta.json"):
        os.makedirs("data", exist_ok=True)
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.meta_path = meta_path
        self.meta = []
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self._load_index()

    def _load_index(self):
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
            else:
                self.index = faiss.IndexFlatIP(self.dim)
                self.meta = []
        except Exception:
            # fallback to empty
            self.index = faiss.IndexFlatIP(self.dim)
            self.meta = []

    def add(self, text: str, answer: str, source: str = "user"):
        emb = self.model.encode(text, convert_to_numpy=True)
        faiss.normalize_L2(emb.reshape(1, -1))
        self.index.add(emb.reshape(1, -1))
        self.meta.append({"text": text, "answer": answer, "source": source})
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def query(self, text: str, top_k: int = 4):
        if len(self.meta) == 0:
            return []
        q = self.model.encode(text, convert_to_numpy=True)
        faiss.normalize_L2(q.reshape(1, -1))
        scores, idxs = self.index.search(q.reshape(1, -1), top_k)
        results = []
        for s, i in zip(scores[0], idxs[0]):
            if i < len(self.meta):
                results.append({"meta": self.meta[i], "score": float(s)})
        return results
