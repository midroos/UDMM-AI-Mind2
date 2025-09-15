import json
import os
import sys
from tqdm import tqdm

# Add the src directory to the Python path to allow for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.udmm2.memory.faiss_rag import FaissRAG
from src.udmm2.config import EMBEDDING_MODEL, FAISS_INDEX_PATH, FAISS_META_PATH

def main():
    """
    Reads a knowledge base file and indexes its contents using FaissRAG.
    """
    knowledge_base_path = "data/knowledge_base.json"

    if not os.path.exists(knowledge_base_path):
        print(f"Error: Knowledge base file not found at '{knowledge_base_path}'")
        return

    print("Loading knowledge base...")
    with open(knowledge_base_path, "r", encoding="utf-8") as f:
        knowledge_data = json.load(f)

    # Clean up old index files if they exist
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
        print(f"Removed old index file: {FAISS_INDEX_PATH}")
    if os.path.exists(FAISS_META_PATH):
        os.remove(FAISS_META_PATH)
        print(f"Removed old meta file: {FAISS_META_PATH}")

    print(f"Initializing FaissRAG with model: {EMBEDDING_MODEL}")
    # This will create a new, empty index because the old files were removed
    rag = FaissRAG(EMBEDDING_MODEL, index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH)

    print(f"Indexing {len(knowledge_data)} entries...")

    # Using tqdm for a progress bar
    for entry in tqdm(knowledge_data, desc="Indexing entries"):
        question = entry.get("question")
        answer = entry.get("answer")
        if question and answer:
            rag.add(text=question, answer=answer, source="knowledge_base")
        else:
            print(f"Skipping invalid entry: {entry}")

    print("\nIndexing complete.")
    print(f"Faiss index saved to: {FAISS_INDEX_PATH}")
    print(f"Metadata saved to: {FAISS_META_PATH}")
    print(f"Total entries in memory: {len(rag.meta)}")

if __name__ == "__main__":
    main()
