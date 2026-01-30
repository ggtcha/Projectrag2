from typing import Generator, List, Dict, Optional
import os
import re
from dotenv import load_dotenv
from functools import lru_cache
from datetime import datetime
from contextlib import contextmanager
import psycopg2
import html

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

load_dotenv()

#============================================================================
# CONFIGURATION
#============================================================================

PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DATABASE = os.getenv("PG_DATABASE")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = "http://localhost:11434"

TICKET_RESPONSE_MODE = os.getenv("TICKET_RESPONSE_MODE", "rule")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

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

MAX_KEYWORD_RESULTS = 100
MAX_EXACT_MATCHES = 3
MAX_LOCATION_MATCHES = 5
MAX_SEMANTIC_RESULTS = 10
MAX_FINAL_RESULTS = 10

#============================================================================
# GLOBAL INSTANCES
#============================================================================

_embeddings = None
_vectorstore = None
_retriever = None
_llm = None
_db_conn = None

def log(message: str, level: str = "info"):
    levels = {"debug": 0, "info": 1, "error": 2, "none": 3}
    current_level = levels.get(LOG_LEVEL, 1)
    message_level = levels.get(level, 1)
    
    if message_level >= current_level:
        print(message)

#============================================================================
# DATABASE CONNECTION MANAGEMENT
#============================================================================
def get_db_connection():
    global _db_conn
    try:
        if _db_conn is None or _db_conn.closed:
            log("[DB] Connecting...", "debug")
            _db_conn = psycopg2.connect(PSYCOPG_CONN_INFO)
        return _db_conn
    except Exception as e:
        log(f"[ERROR] Database connection failed: {e}", "error")
        raise

@contextmanager
def get_db_cursor():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        log(f"[ERROR] Database operation failed: {e}", "error")
        raise e
    finally:
        cursor.close()

#============================================================================
# COMPONENT INITIALIZATION
#============================================================================

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        log("[LLM] Creating embeddings...", "debug")
        _embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL
        )
    return _embeddings

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        log("[LLM] Connecting to vectorstore...", "debug")
        _vectorstore = PGVector(
            connection_string=SQLALCHEMY_DB_URL,
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
        )
    return _vectorstore

def get_llm():
    global _llm
    if _llm is None:
        log("[LLM] Connecting to LLM with Strict Settings...", "debug")
        _llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.3,
            stream=False,
            base_url=OLLAMA_BASE_URL,
            num_ctx=4096,
            repeat_penalty=1.1, # ลดลงเพื่อแก้ปัญหาภาษาไทยเว้นวรรค
            top_p=0.9,
        )
    return _llm

#============================================================================
# TEXT PROCESSING UTILITIES
#============================================================================

def is_code_like(text: str) -> bool:
    if not text:
        return False
    compact = re.sub(r'\s+', '', text.upper())
    return bool(
        re.fullmatch(r'[A-Z0-9\-]{4,}', compact) or
        re.search(r'(THINKPAD|ELITEBOOK|OPTIPLEX)\d*', compact)
    )

def deduplicate_documents(docs: List[Document]) -> List[Document]:
    seen = set()
    unique_docs = []
    for doc in docs:
        serial = (doc.metadata.get('serial') or '').strip().upper()
        asset = (doc.metadata.get('asset_no') or '').strip()
        if serial:
            unique_key = f"serial_{serial}"
        elif asset:
            unique_key = f"asset_{asset}"
        else:
            unique_key = f"row_{doc.metadata.get('row', '')}_{doc.metadata.get('model', '')}"
        
        if unique_key not in seen:
            seen.add(unique_key)
            unique_docs.append(doc)
    return unique_docs

def clean_text_formatting(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    # ลบช่องว่างระหว่างอักษรไทย
    text = re.sub(r'(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'_{2,}', ' ', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

#============================================================================
# INTENT CLASSIFICATION
#============================================================================

def classify_intent(question: str) -> str:
    q_lower = question.lower()
    if re.search(r'(serial|serial number|sn|asset|asset number|ขอserial|ขอ serial)', q_lower):
        log(f"[Intent] inventory (serial request)", "debug")
        return "inventory"
    if is_code_like(question):
        log(f"[Intent] inventory (code pattern)", "debug")
        return "inventory"
    
    strong_ticket_patterns = [
        r'ทำ(อย่างไร|ยังไง|ไง)', r'วิธี(การ|แก้|ใช้)', r'(how to|how do|how can)',
        r'(แก้|fix|solve|แก้ไข)', r'ปัญหา', r'(ไม่ทำงาน|not working|broken|เสีย)',
        r'(ลูกค้า|user|ผู้ใช้)(แจ้ง|report|ถาม)', r'(ticket|issue|problem|help|support)',
        r'(รีเซ็ต|reset|configure|setup|install|ติดตั้ง)',
        r'(vpn|network|connection|password|login|email)',
        r'(เชื่อมต่อ|เข้าสู่ระบบ|ความปลอดภัย)',
        r'(ขอขั้นตอน|step)',
    ]
    for pattern in strong_ticket_patterns:
        if re.search(pattern, q_lower):
            log(f"[Intent] ticket (matched: {pattern})", "debug")
            return "ticket"
    
    patterns = extract_search_patterns(question)
    inventory_score = (
        len(patterns["serials"]) * 3 + len(patterns["assets"]) * 3 +
        len(patterns["model_nos"]) * 2 + len(patterns["models"]) * 2 +
        len(patterns["locations"]) * 1
    )
    ticket_keywords = [
        "ช่วย", "assist", "support", "help", "แจ้ง", "report",
        "ปัญหา", "problem", "issue", "แก้", "fix", "solve",
        "ติดตั้ง", "install", "configure", "setup"
    ]
    ticket_score = sum(1 for k in ticket_keywords if k in q_lower)
    
    if inventory_score >= 3: return "inventory"
    if ticket_score >= 2: return "ticket"
    if inventory_score > ticket_score: return "inventory"
    elif ticket_score > inventory_score: return "ticket"
    return "general"

#============================================================================
# KEYWORD EXTRACTION & SEARCH
#============================================================================

def extract_search_patterns(question: str) -> Dict[str, list]:
    patterns = {
        "serials": [], "assets": [], "models": [], "model_nos": [],
        "locations": [], "keywords": [], "specific_model": None
    }
    device_words = {
        'PRINTER', 'MACBOOK', 'THINKPAD', 'LAPTOP', 'SWITCH', 
        'ROUTER', 'DESKTOP', 'MONITOR', 'KEYBOARD', 'SCANNER'
    }
    potential_serials = re.findall(r'\b[A-Z0-9]{8,20}\b', question.upper())
    serials = [s for s in potential_serials if s not in device_words]
    patterns["serials"].extend(serials)
    assets = re.findall(r'\b\d{7,10}\b', question)
    patterns["assets"].extend(assets)
    model_nos = re.findall(r'\b[A-Z0-9]{2,3}-[A-Z0-9-]+\b', question.upper())
    patterns["model_nos"].extend(model_nos)
    
    q_upper = question.upper()
    model_pattern_checks = [
        r'\b([A-Z0-9]{2,3}-[A-Z0-9]+)\b', r'\b(FR-\d+)\b', r'\b(2930F|2930M)\b',
        r'\b(JL\d+[A-Z])\b', r'\b(THINKPAD\w*)\b', r'\b(ELITEBOOK\w*)\b',
    ]
    for pattern in model_pattern_checks:
        match = re.search(pattern, q_upper)
        if match:
            detected = match.group(1).strip()
            patterns["specific_model"] = detected
            patterns["models"].append(detected.lower())
            break
            
    if not patterns["specific_model"] and is_code_like(question):
        clean_q = re.sub(r'\s+', '', question.upper())
        patterns["specific_model"] = clean_q
        patterns["models"].append(clean_q.lower())
        
    model_keywords = [
        "fr-4080", "2930f", "2930m", "h2-pncn", "thinkpad", "thinkcentre", 
        "thinkstation", "switch", "router", "beacon", "gateway", "access point", 
        "ups", "elitebook", "optiplex", "prodesk",
    ]
    for mk in model_keywords:
        if mk in question.lower() and mk not in patterns["models"]:
            patterns["models"].append(mk)
            
    location_keywords = ["sriracha", "ศรีราชา", "chonburi", "ชลบุรี", "custom", "customs", "server room", "building"]
    for lk in location_keywords:
        if lk in question.lower():
            patterns["locations"].append(lk)
            
    return patterns

def keyword_search_direct(patterns: Dict[str, list]) -> List[Document]:
    all_docs = []
    try:
        with get_db_cursor() as cursor:
            base_filter = "AND cmetadata->>'source' = 'inventory'"
            if patterns.get("specific_model"):
                specific_model = patterns["specific_model"]
                query = f"SELECT document, cmetadata FROM langchain_pg_embedding WHERE (UPPER(cmetadata->>'model') LIKE UPPER(%s) OR UPPER(cmetadata->>'model_no') LIKE UPPER(%s)) {base_filter} LIMIT {MAX_KEYWORD_RESULTS}"
                cursor.execute(query, (f'%{specific_model}%', f'%{specific_model}%'))
                all_docs.extend([Document(page_content=doc, metadata=meta or {}) for doc, meta in cursor.fetchall()])
            
            for model in patterns["models"]:
                query = f"SELECT document, cmetadata FROM langchain_pg_embedding WHERE (LOWER(cmetadata->>'model') LIKE LOWER(%s) OR LOWER(cmetadata->>'model_no') LIKE LOWER(%s)) {base_filter} LIMIT 20"
                cursor.execute(query, (f'%{model}%', f'%{model}%'))
                all_docs.extend([Document(page_content=doc, metadata=meta or {}) for doc, meta in cursor.fetchall()])
                
            for serial in patterns["serials"]:
                query = f"SELECT document, cmetadata FROM langchain_pg_embedding WHERE UPPER(cmetadata->>'serial') = UPPER(%s) {base_filter} LIMIT {MAX_EXACT_MATCHES}"
                cursor.execute(query, (serial,))
                all_docs.extend([Document(page_content=doc, metadata=meta or {}) for doc, meta in cursor.fetchall()])
                
            for asset in patterns["assets"]:
                query = f"SELECT document, cmetadata FROM langchain_pg_embedding WHERE cmetadata->>'asset_no' = %s {base_filter} LIMIT {MAX_EXACT_MATCHES}"
                cursor.execute(query, (asset,))
                all_docs.extend([Document(page_content=doc, metadata=meta or {}) for doc, meta in cursor.fetchall()])
    except Exception as e:
        log(f"[ERROR] Keyword search failed: {e}", "error")
    return all_docs

def hybrid_retrieve(question: str) -> List[Document]:
    if is_code_like(question):
        compact = re.sub(r'\s+', '', question.upper())
        try:
            with get_db_cursor() as cursor:
                query = "SELECT document, cmetadata FROM langchain_pg_embedding WHERE (UPPER(cmetadata->>'serial') = %s OR UPPER(cmetadata->>'asset_no') = %s OR UPPER(cmetadata->>'model_no') = %s OR UPPER(cmetadata->>'model') = %s) AND cmetadata->>'source' = 'inventory' LIMIT %s"
                cursor.execute(query, (compact, compact, compact, compact, MAX_EXACT_MATCHES))
                rows = cursor.fetchall()
                if rows: return [Document(page_content=doc, metadata=meta or {}) for doc, meta in rows]
        except Exception: pass
    
    patterns = extract_search_patterns(question)
    keyword_docs = keyword_search_direct(patterns)
    keyword_docs = deduplicate_documents(keyword_docs)
    
    if patterns["serials"]:
        exact = [d for d in keyword_docs if (d.metadata.get('serial') or '').upper() == patterns["serials"][0].upper()]
        if exact: return exact
    if patterns["assets"]:
        exact = [d for d in keyword_docs if (d.metadata.get('asset_no') or '') == patterns["assets"][0]]
        if exact: return exact
    if patterns["specific_model"]: return keyword_docs
    if keyword_docs: return keyword_docs[:MAX_FINAL_RESULTS]
    
    try:
        semantic_docs = get_vectorstore().similarity_search(question, k=MAX_SEMANTIC_RESULTS, filter={"source": "inventory"})
        if patterns["specific_model"]:
            semantic_docs = [d for d in semantic_docs if patterns["specific_model"].upper() in (d.metadata.get('model') or '').upper()]
        return semantic_docs
    except Exception: return []

def retrieve_tickets(question: str) -> List[Document]:
    try:
        return get_vectorstore().similarity_search(question, k=30, filter={"source": "support_tickets"})
    except Exception: return []

#============================================================================
# CONTEXT FORMATTING
#============================================================================

def format_inventory_context(docs: List[Document], max_docs: int = 3) -> str:
    if not docs: return "No data found."
    lines = []
    for i, doc in enumerate(docs[:max_docs], 1):
        meta = doc.metadata
        lines.append(f"ITEM_{i}:")
        lines.append(f"  MODEL={meta.get('model', 'N/A')}")
        lines.append(f"  SERIAL_NUMBER={meta.get('serial', 'N/A')}")
        lines.append(f"  ASSET_NUMBER={meta.get('asset_no', 'N/A')}")
        lines.append(f"  STATUS={meta.get('status', 'N/A')}")
        lines.append(f"  LOCATION={meta.get('location', 'N/A')}")
        lines.append("")
    return "\n".join(lines)

def format_ticket_context(docs: List[Document], max_docs: int = 10) -> str:
    if not docs: return "No related tickets found."
    lines = [f"Found {len(docs)} related tickets:\n"]
    for i, doc in enumerate(docs[:max_docs], 1):
        lines.append(f"\n--- Ticket #{i} ---")
        lines.append(f"Subject: {clean_text_formatting(doc.metadata.get('subject', 'N/A'))}")
        lines.append(f"Details:\n{clean_text_formatting(doc.page_content)}")
    return "\n".join(lines)

#============================================================================
# PROMPT TEMPLATES (HERE IS THE FIX!)
#============================================================================

IT_ASSET_PROMPT = ChatPromptTemplate.from_template("""
You are an intelligent IT Asset Database Assistant.

INSTRUCTIONS:
1. Analyze the USER QUESTION to detect the target language (Thai or English).
2. Respond in the SAME language as the USER QUESTION.
3. If responding in Thai, write naturally without extra spaces between characters.
4. Search the provided DATA context for asset details.

DATA:
{context}

USER QUESTION:
{question}

RESPONSE FORMAT:
For EACH asset found, provide the following details (preserve the exact values):
- Model: [Value]
- Serial Number: [Value] (NEVER omit this)
- Asset Number: [Value]
- Status: [Value]
- Location: [Value]

If multiple assets match, list all of them.

ANSWER:
""")

SUPPORT_TICKET_PROMPT = ChatPromptTemplate.from_template("""
You are a professional IT Support Expert.

CORE INSTRUCTIONS:
1. **Language Detection**: Detect the language of the USER QUESTION and respond in the SAME language.
   - If Thai: Write continuous, natural sentences. DO NOT insert spaces between characters.
   - If English: Use professional IT support tone.
   - DO NOT use Chinese characters.

2. **Knowledge Source**:
   - First, check the "SUPPORT TICKETS" context below.
   - If the context contains the solution, use it.
   - If the context is empty, irrelevant, or insufficient, USE YOUR GENERAL IT KNOWLEDGE to provide a helpful solution immediately.
   - Do not say "I don't know" if you can offer general troubleshooting steps.

3. **Format**:
   - Provide a step-by-step guide (1., 2., 3.).
   - Be polite and professional.

CONTEXT (SUPPORT TICKETS):
{context}

USER QUESTION:
{question}

YOUR ANSWER:
""")

GENERAL_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful and polite AI Assistant.

INSTRUCTIONS:
1. Respond in the SAME language as the USER QUESTION (Thai for Thai, English for English).
2. For Thai output: Ensure text is continuous without spaces between characters (e.g., "สวัสดีครับ" NOT "ส ว ั ส ด ี").
3. Answer the question directly using your general knowledge.
4. Be concise and friendly.

USER QUESTION:
{question}

ANSWER:
""")

#============================================================================
# CHAT HISTORY
#============================================================================

@lru_cache(maxsize=10)
def get_session_history(session_id: str):
    return PostgresChatMessageHistory(connection_string=PSYCOPG_CONN_INFO, session_id=session_id)

#============================================================================
# MAIN CHAT FUNCTION
#============================================================================

def chat_with_warehouse_system(session_id: str, question: str) -> Generator[str, None, None]:
    log(f"[Chat] Session: {session_id} | Query: {question}", "info")
    try:
        llm = get_llm()
        history = get_session_history(session_id)
        intent = classify_intent(question)
        if is_code_like(question): intent = "inventory"
        log(f"[Chat] Intent: {intent}", "info")
        
        docs = hybrid_retrieve(question)

        if intent == "inventory":
            log("[Chat] Processing inventory query", "debug")
            inventory_display_text = ""
            
           
            if docs:
                header = f"Found {len(docs)} items:\n\n"
                yield header
                inventory_display_text += header
                for i, doc in enumerate(docs, 1):
                    meta = doc.metadata
                 
                    item_info = (f"**Item {i}**\n"
                                 f"- Model: {meta.get('model', 'N/A')}\n"
                                 f"- Serial: {meta.get('serial', 'N/A')}\n"
                                 f"- Asset No: {meta.get('asset_no', 'N/A')}\n"
                                 f"- Status: {meta.get('status', 'N/A')}\n"
                                 f"- Location: {meta.get('location', 'N/A')}\n\n")
                    yield item_info
                    inventory_display_text += item_info
                
               
                history.add_user_message(question)
                history.add_ai_message(inventory_display_text)
                return 

          
            context = "No asset data found matching this query."
            chain = {"context": lambda _: context, "question": RunnablePassthrough()} | IT_ASSET_PROMPT | llm
            
            full_ai_part = ""
            for chunk in chain.stream(question):
                content = getattr(chunk, "content", str(chunk))
                content = clean_text_formatting(content)
                full_ai_part += content
                yield content
            
            history.add_user_message(question)
            history.add_ai_message(full_ai_part)
            return
        elif intent == "ticket":
            log("[Chat] Processing ticket query (Hybrid Mode)", "debug")
            ticket_docs = retrieve_tickets(question)
            
            if ticket_docs:
                context = format_ticket_context(ticket_docs, max_docs=10)
            else:
                context = "No direct matches found in database. Please use your general IT knowledge to assist the user."
            
            chain = {"context": lambda _: context, "question": RunnablePassthrough()} | SUPPORT_TICKET_PROMPT | llm
            
            full_response = ""
            for chunk in chain.stream(question):
                content = getattr(chunk, "content", str(chunk))
                content = clean_text_formatting(content)
                full_response += content
                yield content
            
            history.add_user_message(question)
            history.add_ai_message(full_response)
            return
        
        else:
            log("[Chat] Processing general query", "debug")
            chain = {"question": RunnablePassthrough()} | GENERAL_PROMPT | llm
            full_response = ""
            for chunk in chain.stream(question):
                content = getattr(chunk, "content", str(chunk))
                full_response += content
                yield content
            history.add_user_message(question)
            history.add_ai_message(full_response)
    
    except Exception as e:
        log(f"[ERROR] Chat failed: {e}", "error")
        yield f"Error: {str(e)}\n"

def clear_session_history(session_id: str):
    history = get_session_history(session_id)
    history.clear()
    get_session_history.cache_clear()
def cleanup_resources():
    global _vectorstore, _embeddings, _llm, _retriever, _db_conn
    
    if _db_conn and not _db_conn.closed:
        _db_conn.close()
        log("[Cleanup] Database connection closed", "info")
    
    _vectorstore = None
    _embeddings = None
    _llm = None
    _retriever = None
    _db_conn = None
    
    get_session_history.cache_clear()
    log("[Cleanup] Resources cleaned up", "info")