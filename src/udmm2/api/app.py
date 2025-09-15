# app.py
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ..udmm_core import UDMMAgent
from ..config import API_HOST, API_PORT

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
agent = UDMMAgent()

@app.post("/ask")
async def ask(req: Request):
    data = await req.json()
    q = data.get("question")
    if not q:
        return {"error": "no question"}
    return agent.perceive_and_answer(q)

@app.post("/teach")
async def teach(req: Request):
    data = await req.json()
    q = data.get("question")
    a = data.get("answer")
    if not q or not a:
        return {"error": "provide question and answer"}
    return agent.teach(q, a)

@app.get("/status")
async def status():
    return {
        "body": {"energy": agent.body.energy, "arousal": agent.body.arousal},
        "memory_count": len(agent.rag.meta),
        "simulation": agent.simulation.get_current_state()
    }

@app.post("/simulate")
async def simulate(req: Request):
    """
    يسمح بتشغيل خطوة محاكاة بشكل مباشر لاختبار ديناميكيات UDMM.
    """
    try:
        data = await req.json()
        user_input = data.get("text", "")
        action = data.get("action", None)
        state = agent.simulation.step({"text": user_input}, action=action)
        return {"state": state}
    except Exception:
        # Fallback for empty body
        state = agent.simulation.step({"text": ""}, action=None)
        return {"state": state}
