from typing import Generator, List
import os
import re
from dotenv import load_dotenv
from functools import lru_cache
from datetime import datetime
import psycopg2

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

load_dotenv()

# Configuration
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DATABASE = os.getenv("PG_DATABASE")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = "http://localhost:11434"

# Database Connection
SQLALCHEMY_DB_URL = (
    f"postgresql+psycopg2://"
    f"{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
)

PSYCOPG_CONN_INFO = (
    f"dbname={PG_DATABASE} "
    f"user={PG_USER} "
    f"password={PG_PASSWORD} "
    f"host={PG_HOST} "
    f"port={PG_PORT}"
)

# Lazy Initialization
_embeddings = None
_vectorstore = None
_retriever = None
_llm = None
_db_conn = None

# Core Functions

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL
        )
    return _embeddings

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = PGVector(
            connection_string=SQLALCHEMY_DB_URL,
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
        )
    return _vectorstore

def get_retriever():
    global _retriever
    if _retriever is None:
        _vectorstore = get_vectorstore()
        _retriever = _vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    return _retriever

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0,
            stream=True,
            base_url=OLLAMA_BASE_URL,
            num_ctx=4096,
            num_predict=512,
            repeat_penalty=1.2
        )
    return _llm

def get_db_connection():
    global _db_conn
    if _db_conn is None or _db_conn.closed:
        _db_conn = psycopg2.connect(PSYCOPG_CONN_INFO)
    return _db_conn

# Keyword Extraction

def extract_search_patterns(question: str) -> dict:
    patterns = {
        "serials": [],
        "assets": [],
        "models": [],
        "locations": [],
        "keywords": []
    }
    
    serials = re.findall(r'\b[a-zA-Z0-9-]{4,20}\b', question)
    patterns["serials"].extend([s.upper() for s in serials])
    
    assets = re.findall(r'\b\d{7,10}\b', question)
    patterns["assets"].extend(assets)
    
    model_keywords = ["thinkpad", "thinkcentre", "thinkstation", "switch", 
                      "router", "printer", "mac", "elitebook", "optiplex",
                      "g100", "6100", "neverstop"]
    
    q_lower = question.lower()
    for mk in model_keywords:
        if mk in q_lower:
            patterns["models"].append(mk)
    
    location_keywords = ["sriracha", "ศรีราชา", "chonburi", "ชลบุรี", 
                         "custom", "customs"]
    
    for lk in location_keywords:
        if lk in q_lower:
            patterns["locations"].append(lk)
    
    if any(k in q_lower for k in ["spare", "พร้อมใช้", "สำรอง"]):
        patterns["keywords"].append("spare")
    
    if any(k in q_lower for k in ["obsolete", "เลิกใช้", "เสื่อม"]):
        patterns["keywords"].append("obsolete")
    
    return patterns

# Hybrid Retrieval

def keyword_search_direct(patterns: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    all_docs = []
    
    search_terms = patterns["serials"] + patterns["assets"]
    
    try:
        for term in search_terms:
            query = """
            SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE (cmetadata->>'Serial')::text = %s 
                   OR (cmetadata->>'Asset No')::text = %s
                LIMIT 5
            """
            cursor.execute(query, (term.upper(), term))
            for doc_content, metadata in cursor.fetchall():
                all_docs.append(Document(page_content=doc_content, metadata=metadata or {}))
    finally:
        cursor.close()
        conn.close()
    return all_docs

def hybrid_retrieve(question: str) -> List[Document]:
    patterns = extract_search_patterns(question)
    
    keyword_docs = keyword_search_direct(patterns)
    
    if patterns["serials"]:
        target_s = patterns["serials"][0].upper()
        exact_matches = [d for d in keyword_docs if str(d.metadata.get('Serial')).upper() == target_s]
        if exact_matches:
            print(f"[SUCCESS] Found Exact Serial: {target_s}")
            return exact_matches[:1]

    semantic_docs = get_vectorstore().as_retriever(search_kwargs={"k": 3}).invoke(question)
    return (keyword_docs + semantic_docs)[:3]

# Enhanced Prompts
# --- แก้ไข IT_ASSET_PROMPT ---
IT_ASSET_PROMPT = ChatPromptTemplate.from_template("""
You are an IT Support Assistant. 
Answer in the SAME LANGUAGE as the user's question (Thai or English).

## Current Date: {current_date}

## Data from system:
{context}

## User Question:
{question}

## Instructions:
1. Read all data carefully.
2. Answer based ONLY on facts from the data above.
3. Use bullet points and emojis for readability.
4. If multiple items found, show all or at least first 5.
5. If data doesn't match the question, say "I couldn't find information for [item]" in the user's language.

Answer:
""")

# --- แก้ไข GENERAL_PROMPT ---
GENERAL_PROMPT = ChatPromptTemplate.from_template("""
You are a friendly IT Support Assistant. 
Answer in the SAME LANGUAGE as the user's question (Thai or English).

User: {question}

Answer (friendly, helpful tone):
""")
# Chat History

@lru_cache(maxsize=10)
def get_session_history(session_id: str):
    return PostgresChatMessageHistory(
        connection_string=PSYCOPG_CONN_INFO,
        session_id=session_id
    )

# Intent Classification

IT_ASSET_KEYWORDS = [
    "serial", "s/n", "sn", "asset", "model", "รุ่น", "เครื่อง", "อุปกรณ์",
    "มี", "เหลือ", "กี่", "จำนวน", "spare", "obsolete", "ค้นหา", "หา",
    "thinkpad", "laptop", "switch", "router", "printer", "computer",
    "location", "ตำแหน่ง", "อยู่ที่", "sriracha", "ศรีราชา", "Model No",
    "model no", "asset no", "asset no.", "serial number"
]

def classify_intent(question: str) -> str:
    q_lower = question.lower()
    
    if re.search(r'[A-Z0-9]{7,}', question):
        return "it_asset"
    
    if any(k in q_lower for k in IT_ASSET_KEYWORDS):
        return "it_asset"
    
    return "general"

# Context Formatting

def format_context_for_llm(docs, max_docs: int = 50) -> str:
    if not docs:
        return "ไม่พบข้อมูล"
    
    docs = docs[:max_docs]
    parts = [f"พบข้อมูล {len(docs)} รายการที่เกี่ยวข้อง", ""]
    
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        
        def get_val(keys_list):
            low_meta = {k.lower(): v for k, v in meta.items()}
            for k in keys_list:
                if k in meta: return meta[k]
                if k.lower() in low_meta: return low_meta[k.lower()]
            return "N/A"

        model_name = get_val(['model', 'Model Name'])
        model_no = get_val(['model no.', 'model no', 'model_no'])
        serial = get_val(['serial', 'serial number', 's/n'])
        asset = get_val(['asset no', 'asset no.', 'asset_no'])
        status = get_val(['status'])
        location = get_val(['location', 'locations'])

        parts.append(f"### รายการที่ {i}:")
        parts.append(f"- รุ่น: {model_name} (Model No: {model_no})")
        parts.append(f"- Serial: {serial}")
        parts.append(f"- Asset No: {asset}")
        parts.append(f"- สถานะ: {status}")
        parts.append(f"- ตำแหน่ง: {location}")
        parts.append(f"รายละเอียดเพิ่มเติม: {doc.page_content}")
        parts.append("-" * 30)
    
    return "\n".join(parts)

# Main Chat Function

def chat_with_warehouse_system(
    session_id: str,
    question: str,
    image: bytes | None = None
) -> Generator[str, None, None]:
    
    llm = get_llm()
    history = get_session_history(session_id)
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    intent = classify_intent(question)
    print(f"\n[INTENT] {intent}")
    
    if intent == "it_asset":
        docs = hybrid_retrieve(question)
        
        if not docs:
            yield "🔍 ไม่พบข้อมูลในระบบ\n\n"
            yield "💡 ลองตรวจสอบ:\n"
            yield "• Serial Number ถูกต้องหรือไม่\n"
            yield "• ค้นหาด้วย Model หรือ Asset Number\n"
            return
        
        context = format_context_for_llm(docs)
        
        chain = (
            {
                "context": lambda _: context,
                "question": RunnablePassthrough(),
                "current_date": lambda _: current_date
            }
            | IT_ASSET_PROMPT
            | llm
        )
        
        full_response = ""
        for chunk in chain.stream(question):
            content = getattr(chunk, "content", str(chunk))
            full_response += content
            yield content
        
        history.add_user_message(question)
        history.add_ai_message(full_response)
        return
    
    chain = (
        {
            "question": RunnablePassthrough(),
            "current_date": lambda _: current_date
        }
        | GENERAL_PROMPT
        | llm
    )
    
    full_response = ""
    for chunk in chain.stream(question):
        content = getattr(chunk, "content", str(chunk))
        full_response += content
        yield content
    
    history.add_user_message(question)
    history.add_ai_message(full_response)

# Utilities

def clear_session_history(session_id: str):
    history = get_session_history(session_id)
    history.clear()
    get_session_history.cache_clear()

def cleanup_resources():
    global _vectorstore, _embeddings, _llm, _retriever, _db_conn
    if _db_conn and not _db_conn.closed:
        _db_conn.close()
    _vectorstore = None
    _embeddings = None
    _llm = None
    _retriever = None
    _db_conn = None
    get_session_history.cache_clear()