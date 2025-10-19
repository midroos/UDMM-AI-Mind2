# UDMM LLM Agent

This project is a Python implementation of the 'Unified Dynamic Model of Mind (UDMM)' cognitive architecture. The agent uses a FastAPI backend and a Streamlit frontend.

## Features

-   **Dynamic Memory Simulation**: The agent includes a dynamic memory simulation based on the paper "Memory as a Distributed Predictive Simulation: A Unified Dynamic Model within the UDMM Hybrid Framework". The simulation's state influences the agent's behavior, and its history is visualized in the UI.
-   **RAG-based Memory**: The agent uses a FAISS-based RAG (Retrieval-Augmented Generation) system for its long-term memory.
-   **Hierarchical Intent Manager**: The agent uses a hierarchical intent manager to decompose high-level goals into subgoals.
-   **Dynamic Emotional Precision**: The agent's emotional precision is influenced by its body state and the information tension from the memory simulation.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Backend**:
    ```bash
    uvicorn src.udmm2.api.app:app --host 0.0.0.0 --port 8000
    ```
3.  **Run the Frontend**:
    ```bash
    streamlit run ui_streamlit.py
    ```
