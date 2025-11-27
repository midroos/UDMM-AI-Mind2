from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import time

# Corrected import path
from ..udmm_core import UDMM_Agent

app = FastAPI(title="UDMM Dynamic Agent API", description="الوكيل الديناميكي الكامل مع UDMM + AAR")

# تخزين الوكائل النشطة
active_agents = {}

class AgentRequest(BaseModel):
    message: str
    agent_id: str = "default"
    reset: bool = False

class AgentResponse(BaseModel):
    response: str
    agent_id: str
    state: Dict[str, Any]
    intervention_occurred: bool

@app.post("/chat", response_model=AgentResponse)
async def chat_with_agent(request: AgentRequest):
    try:
        # إنشاء أو إعادة تعيين الوكيل
        if request.agent_id not in active_agents or request.reset:
            active_agents[request.agent_id] = UDMM_Agent(request.agent_id)

        agent = active_agents[request.agent_id]

        # توليد الرد والحصول على حالة التدخل مباشرة
        response, intervention_occurred = agent.generate_response(request.message)

        return AgentResponse(
            response=response,
            agent_id=request.agent_id,
            state=agent.get_detailed_state(),
            intervention_occurred=intervention_occurred
        )

    except Exception as e:
        # Log the exception for debugging
        print(f"Error during chat: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/{agent_id}/state")
async def get_agent_state(agent_id: str):
    if agent_id not in active_agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    return active_agents[agent_id].get_detailed_state()

@app.post("/agent/{agent_id}/reset")
async def reset_agent(agent_id: str):
    active_agents[agent_id] = UDMM_Agent(agent_id)
    return {"message": f"Agent {agent_id} reset successfully"}

if __name__ == "__main__":
    # Ensure this runs from the root of the project for correct module resolution
    uvicorn.run("src.udmm2.api.app:app", host="0.0.0.0", port=8000, reload=True)
