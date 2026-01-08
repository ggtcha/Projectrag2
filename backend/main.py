from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from src.rag_query import chat_with_warehouse_system, get_session_history, clear_session_history
import json
import os
import asyncio

app = FastAPI()

# อนุญาตให้ React (Port 5173) เชื่อมต่อได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_FILE = "chat_sessions.json"

def update_chat_sessions(session_id, title):
    """บันทึกรายชื่อ Session ลงไฟล์ JSON สำหรับแสดงใน Sidebar"""
    sessions = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                sessions = json.load(f)
        except: sessions = []
    
    # เพิ่ม Session ใหม่ไว้บนสุดถ้ายังไม่มี
    if not any(s['id'] == session_id for s in sessions):
        sessions.insert(0, {"id": session_id, "title": title[:40]})
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)

@app.get("/api/sessions")
async def get_sessions():
    """ดึงรายชื่อแชททั้งหมด"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """ลบแชทจาก Sidebar (JSON) และ Database (PostgreSQL)"""
    # ลบจากไฟล์ JSON
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            sessions = json.load(f)
        updated_sessions = [s for s in sessions if s['id'] != session_id]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(updated_sessions, f, ensure_ascii=False, indent=2)
    
    # ลบจาก PostgreSQL
    try:
        clear_session_history(session_id)
    except Exception as e:
        print(f"Error clearing DB: {e}")
    return {"status": "deleted"}

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """ส่งคำตอบแบบ Streaming"""
    data = await request.json()
    session_id = data.get("session_id")
    question = data.get("question")
    
    # บันทึกหัวข้อทันที
    update_chat_sessions(session_id, question)
    
    def event_generator():
        for chunk in chat_with_warehouse_system(session_id, question):
            if chunk:
                yield chunk

    return EventSourceResponse(event_generator())

@app.get("/api/messages/{session_id}")
async def get_chat_messages(session_id: str):
    """ดึงประวัติแชทเก่ามาแสดง"""
    history = get_session_history(session_id)
    return [{"role": "user" if m.type == "human" else "assistant", "content": m.content} 
            for m in history.messages]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)