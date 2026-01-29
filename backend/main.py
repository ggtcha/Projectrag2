from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from src.rag_query import chat_with_warehouse_system, get_session_history, clear_session_history
import json
import os
import asyncio
from pydantic import BaseModel

class SessionUpdate(BaseModel):
    title: str

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONSTANTS
# ============================================================================

HISTORY_FILE = "chat_sessions.json"

# ============================================================================
# SESSION MANAGEMENT FUNCTIONS
# ============================================================================

def update_chat_sessions(session_id, title):
    sessions = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                sessions = json.load(f)
        except: 
            sessions = []
    
    existing_session = next((s for s in sessions if s['id'] == session_id), None)
    
    if existing_session:
        sessions = [s for s in sessions if s['id'] != session_id]
        new_entry = existing_session
    else:
        display_title = title[:30] + "..." if len(title) > 30 else title
        new_entry = {"id": session_id, "title": display_title}

    sessions.insert(0, new_entry)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

# ============================================================================
# API ENDPOINTS - SESSION MANAGEMENT
# ============================================================================

@app.get("/api/sessions")
async def get_sessions():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            sessions = json.load(f)
        updated_sessions = [s for s in sessions if s['id'] != session_id]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(updated_sessions, f, ensure_ascii=False, indent=2)
    
    try:
        clear_session_history(session_id)
    except Exception as e:
        print(f"Error clearing DB: {e}")
    
    return {"status": "deleted"}

@app.get("/api/messages/{session_id}")
async def get_chat_messages(session_id: str):
    try:
        history = get_session_history(session_id)
        return [{"role": "user" if m.type == "human" else "assistant", "content": m.content} 
                for m in history.messages]
    except Exception as e:
        return []

# ============================================================================
# API ENDPOINTS - CHAT
# ============================================================================

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id")
        question = data.get("question")

        if not session_id or not question:
            raise HTTPException(status_code=400, detail="Missing session_id or question")

        update_chat_sessions(session_id, question)

        async def event_generator():
            try:
                for chunk in chat_with_warehouse_system(session_id, question):
                    if not chunk:
                        continue

                    yield {
                        "data": chunk
                    }

                    await asyncio.sleep(0)

            except Exception as e:
                yield {
                    "data": f"[Error]: {str(e)}"
                }

        return EventSourceResponse(event_generator())

    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.put("/api/sessions/{session_id}")
async def update_session_title(session_id: str, payload: SessionUpdate):
    if not os.path.exists(HISTORY_FILE):
        raise HTTPException(status_code=404, detail="History file not found")
    
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            sessions = json.load(f)
        
        # ค้นหาและอัปเดต title
        updated = False
        for s in sessions:
            if s['id'] == session_id:
                s['title'] = payload.title
                updated = True
                break
        
        if not updated:
            raise HTTPException(status_code=404, detail="Session not found")
            
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
            
        return {"status": "success", "updated_title": payload.title}
        
    except Exception as e:
        print(f"Update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# ============================================================================
# SERVER ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)