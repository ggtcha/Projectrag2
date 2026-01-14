from fastapi import FastAPI, Request, HTTPException
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
    """บันทึกรายชื่อ Session และย้ายอันล่าสุดขึ้นบนสุดเสมอ"""
    sessions = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                sessions = json.load(f)
        except: 
            sessions = []
    
    # 1. หาว่ามี session นี้อยู่เดิมไหม
    existing_session = next((s for s in sessions if s['id'] == session_id), None)
    
    if existing_session:
        # ถ้ามีอยู่แล้ว ให้ลบออกจากตำแหน่งเดิม
        sessions = [s for s in sessions if s['id'] != session_id]
        # ใช้ Title เดิม หรือจะอัปเดตตามคำถามล่าสุดก็ได้ (ในที่นี้ขอใช้ title เดิมเพื่อความนิ่ง)
        new_entry = existing_session
    else:
        # ถ้าไม่มี (แชทใหม่) สร้าง entry ใหม่
        display_title = title[:30] + "..." if len(title) > 30 else title
        new_entry = {"id": session_id, "title": display_title}

    # 2. เอาไปใส่ไว้ที่ลำดับ 0 (บนสุด) เสมอ
    sessions.insert(0, new_entry)

    # 3. เซฟลงไฟล์
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

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id")
        question = data.get("question")

        if not session_id or not question:
            raise HTTPException(status_code=400, detail="Missing session_id or question")

        # บันทึกหัวข้อแชท
        update_chat_sessions(session_id, question)

        async def event_generator():
            try:
                for chunk in chat_with_warehouse_system(session_id, question):
                    if not chunk:
                        continue

                    # SSE ต้องเป็น dict ที่มี key "data"
                    yield {
                        "data": chunk
                    }

                    # ปล่อย control ให้ event loop
                    await asyncio.sleep(0)

            except Exception as e:
                yield {
                    "data": f"[Error]: {str(e)}"
                }

        return EventSourceResponse(event_generator())

    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/messages/{session_id}")
async def get_chat_messages(session_id: str):
    """ดึงประวัติแชทเก่ามาแสดง"""
    try:
        history = get_session_history(session_id)
        return [{"role": "user" if m.type == "human" else "assistant", "content": m.content} 
                for m in history.messages]
    except Exception as e:
        return []

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)