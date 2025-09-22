import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json

# Correctly import from the parent directory (src/udmm2)
from ..udmm_v4_core import UDMMCore

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)

app = FastAPI(
    title="UDMM v4 Gemini Core API",
    description="API to interact with the Gemini-exclusive UDMM v4 Core. Requires GEMINI_API_KEY.",
    version="4.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the core, sourcing the API key from environment variables.
# The API will not function without this environment variable.
gemini_key = os.environ.get("GEMINI_API_KEY")
core = UDMMCore(gemini_api_key=gemini_key)

@app.post("/ask")
async def ask(payload: dict):
    """
    Receives a question, processes it through the full UDMM core,
    and returns the agent's response along with its internal state.
    """
    question = payload.get("question", "")
    if not question:
        return {"error": "Question not provided"}

    response_data = core.process_input(question)

    # Use the custom encoder to handle numpy types in the response
    return json.loads(json.dumps(response_data, cls=NumpyEncoder))

@app.get("/status")
async def status():
    """
    Returns the current high-level status of the UDMM agent.
    """
    return {
        "api_key_loaded": bool(gemini_key),
        "time_step": core.time_step,
        "system_health": core.system_health,
        "schemas_in_memory": len(core.memory.schemas),
        "current_attractor_state": dict(zip(core.attractor.labels, core.attractor.get_state().tolist()))
    }

@app.get("/")
async def root():
    message = "UDMM v4 Core API is running."
    if not gemini_key:
        message += " WARNING: GEMINI_API_KEY environment variable is not set. LLM calls will fail."
    return {"message": message}
