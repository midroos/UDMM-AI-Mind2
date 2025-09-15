# app.py
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
# Import the new UDMM v4 Core
from ..udmm_v4_core import UDMMCore
from ..config import API_HOST, API_PORT

app = FastAPI(
    title="UDMM v4 Cognitive Agent API",
    description="An API to interact with the UDMM v4 architecture.",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Instantiate the new UDMM Core
udmm_core = UDMMCore(dimensionality=5)

@app.post("/ask", summary="Process user input")
async def ask(req: Request):
    """
    Receives user input, processes it through the full UDMM v4 core,
    and returns the generated response.

    The core logic involves:
    - Calculating informational tension.
    - Updating attractor and intent dynamics.
    - Retrieving relevant memory.
    - Building a context-rich prompt.
    - (Placeholder) Calling an LLM to get a response.
    """
    data = await req.json()
    user_input = data.get("question")

    if not user_input:
        return {"error": "A 'question' field is required."}

    # Process the input using the new core
    response_text = udmm_core.process_input(user_input)

    # The new core also stores detailed history. We can return parts of it.
    last_history_entry = udmm_core.history[-1] if udmm_core.history else None

    return {
        "response": response_text,
        "state_after_processing": last_history_entry
    }

# The old endpoints like /teach, /status, and /simulate are deprecated
# as the new UDMMCore has a more unified `process_input` method.
# They can be re-added later if needed, adapted to the new architecture.
