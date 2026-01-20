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

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DATABASE = os.getenv("PG_DATABASE")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:0.5b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = "http://localhost:11434"

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

# ============================================================================
# GLOBAL VARIABLES (Lazy Initialization)
# ============================================================================

_embeddings = None
_vectorstore = None
_retriever = None
_llm = None
_db_conn = None

# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("[INIT] Creating embeddings...")
        _embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL
        )
    return _embeddings

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        print("[INIT] Connecting to vectorstore...")
        _vectorstore = PGVector(
            connection_string=SQLALCHEMY_DB_URL,
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
        )
    return _vectorstore

def get_retriever():
    global _retriever
    if _retriever is None:
        print("[INIT] Creating retriever...")
        _vectorstore = get_vectorstore()
        _retriever = _vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    return _retriever

def get_llm():
    global _llm
    if _llm is None:
        print("[INIT] Connecting to LLM...")
        _llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.1,
            stream=True,
            base_url=OLLAMA_BASE_URL,
            num_ctx=4096,
            num_predict=256,
            repeat_penalty=1.1,
            top_p=0.9,
            top_k=40
        )
    return _llm

def get_db_connection():
    global _db_conn
    try:
        if _db_conn is None or _db_conn.closed:
            print("[INIT] Connecting to database...")
            _db_conn = psycopg2.connect(PSYCOPG_CONN_INFO)
        return _db_conn
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        raise

# ============================================================================
# KEYWORD EXTRACTION
# ============================================================================

def extract_search_patterns(question: str) -> dict:
    """ดึง keywords และ patterns จากคำถาม"""
    patterns = {
        "serials": [],
        "assets": [],
        "models": [],
        "model_nos": [],
        "locations": [],
        "keywords": [],
        "specific_model": None
    }
    
    # Serial patterns
    serials = re.findall(r'\b[A-Z0-9]{6,20}\b', question.upper())
    patterns["serials"].extend(serials)
    
    # Asset Number patterns
    assets = re.findall(r'\b\d{7,10}\b', question)
    patterns["assets"].extend(assets)
    
    # Model No. patterns
    model_nos = re.findall(r'\b[A-Z]{2,}-[A-Z0-9-]+\b', question.upper())
    patterns["model_nos"].extend(model_nos)
    
    # ✅ ตรวจจับ Specific Model ด้วย patterns ที่แม่นยำขึ้น
    specific_model_patterns = [
        r'\b(2930F)\b',  # HP Switch 2930F
        r'\b(2930M)\b',
        r'\b(JL\d+[A-Z])\b',  # Model No เช่น JL693A
        r'\b(YOGA\s+\S+)',
        r'\b(THINKPAD\s+\S+)',
        r'\b(THINKCENTRE\s+\S+)',
        r'\b(ELITEBOOK\s+\S+)',
        r'\b(OPTIPLEX\s+\S+)',
        r'\b(MACBOOK\s+\S+)',
        r'\b(HP\s+\S+\s+\S+)',
        r'\b([A-Z]+\d+[A-Z]*\s*\d*[A-Z]*)',  # เช่น TC26BK, JL693A
    ]
    
    for pattern in specific_model_patterns:
        match = re.search(pattern, question.upper())
        if match:
            patterns["specific_model"] = match.group(1).strip()
            print(f"[PATTERN MATCH] Detected specific model: {patterns['specific_model']}")
            break
    
    # Model keywords (ทั่วไป)
    model_keywords = [
        "2930f", "2930m",  # HP Switch models
        "thinkpad", "thinkcentre", "thinkstation", 
        "switch", "router", "printer", "beacon",
        "gateway", "access point", "ups",
        "elitebook", "optiplex", "prodesk",
        "6100", "g100", "neverstop", "air420",
        "yoga", "scanner", "sim"
    ]
    
    q_lower = question.lower()
    for mk in model_keywords:
        if mk in q_lower:
            patterns["models"].append(mk)
    
    # Location keywords
    location_keywords = [
        "sriracha", "ศรีราชา", 
        "chonburi", "ชลบุรี",
        "custom", "customs",
        "server room", "building",
        "kp 4.0", "kp4.0"
    ]
    
    for lk in location_keywords:
        if lk in q_lower:
            patterns["locations"].append(lk)
    
    # Status keywords
    if any(k in q_lower for k in ["spare", "พร้อมใช้", "สำรอง", "ว่าง"]):
        patterns["keywords"].append("spare")
    
    if any(k in q_lower for k in ["obsolete", "เลิกใช้", "เสื่อม"]):
        patterns["keywords"].append("obsolete")
    
    print(f"[SEARCH PATTERNS] {patterns}")
    return patterns

# ============================================================================
# HYBRID RETRIEVAL SYSTEM
# ============================================================================

def keyword_search_direct(patterns: dict) -> List[Document]:
    """ค้นหาด้วย SQL โดยตรงจาก metadata (เฉพาะ inventory)"""
    all_docs = []
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # ✅ เพิ่ม filter: เฉพาะ source = 'inventory'
        base_filter = "AND cmetadata->>'source' = 'inventory'"
        
        # ✅ ค้นหาด้วย Specific Model (สำคัญที่สุด)
        if patterns.get("specific_model"):
            specific_model = patterns["specific_model"]
            print(f"[SQL SEARCH] Specific Model: {specific_model}")
            query = f"""
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE UPPER(cmetadata->>'model') LIKE UPPER(%s)
            {base_filter}
            LIMIT 10
            """
            cursor.execute(query, (f'%{specific_model}%',))
            rows = cursor.fetchall()
            print(f"[SQL RESULT] Found {len(rows)} matches for specific model {specific_model}")
            for doc_content, metadata in rows:
                all_docs.append(Document(page_content=doc_content, metadata=metadata or {}))
        
        # ค้นหาด้วย Serial Number
        for serial in patterns["serials"]:
            print(f"[SQL SEARCH] Serial: {serial}")
            query = f"""
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE UPPER(cmetadata->>'serial') = UPPER(%s)
            {base_filter}
            LIMIT 3
            """
            cursor.execute(query, (serial,))
            rows = cursor.fetchall()
            print(f"[SQL RESULT] Found {len(rows)} matches for serial {serial}")
            for doc_content, metadata in rows:
                all_docs.append(Document(page_content=doc_content, metadata=metadata or {}))
        
        # ค้นหาด้วย Asset Number
        for asset in patterns["assets"]:
            print(f"[SQL SEARCH] Asset: {asset}")
            query = f"""
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE cmetadata->>'asset_no' = %s
            {base_filter}
            LIMIT 3
            """
            cursor.execute(query, (asset,))
            rows = cursor.fetchall()
            print(f"[SQL RESULT] Found {len(rows)} matches for asset {asset}")
            for doc_content, metadata in rows:
                all_docs.append(Document(page_content=doc_content, metadata=metadata or {}))
        
        # ค้นหาด้วย Model No
        for model_no in patterns["model_nos"]:
            print(f"[SQL SEARCH] Model No: {model_no}")
            query = f"""
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE UPPER(cmetadata->>'model_no') LIKE UPPER(%s)
            {base_filter}
            LIMIT 3
            """
            cursor.execute(query, (f'%{model_no}%',))
            rows = cursor.fetchall()
            print(f"[SQL RESULT] Found {len(rows)} matches for model_no {model_no}")
            for doc_content, metadata in rows:
                all_docs.append(Document(page_content=doc_content, metadata=metadata or {}))
        
        # ค้นหาด้วย Location
        for loc in patterns["locations"]:
            print(f"[SQL SEARCH] Location: {loc}")
            query = f"""
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE LOWER(cmetadata->>'location') LIKE LOWER(%s)
            {base_filter}
            LIMIT 5
            """
            cursor.execute(query, (f'%{loc}%',))
            rows = cursor.fetchall()
            print(f"[SQL RESULT] Found {len(rows)} matches for location {loc}")
            for doc_content, metadata in rows:
                all_docs.append(Document(page_content=doc_content, metadata=metadata or {}))
        
        cursor.close()
                
    except Exception as e:
        print(f"[ERROR] Keyword search failed: {e}")
    
    return all_docs

def hybrid_retrieve(question: str) -> List[Document]:
    """ผสมผสาน keyword search และ semantic search"""
    print(f"\n{'='*70}")
    print(f"[RETRIEVE START]")
    print(f"Question: {question}")
    print(f"{'='*70}")
    
    patterns = extract_search_patterns(question)
    
    # 1. Keyword Search
    print("\n[STEP 1] Keyword Search")
    keyword_docs = keyword_search_direct(patterns)
    print(f"[STEP 1 RESULT] Found {len(keyword_docs)} docs from keyword search")
    
    # ✅ ถ้าเจอ exact match จาก Serial -> return เฉพาะตัวนั้น
    if patterns["serials"]:
        exact_serial = patterns["serials"][0].upper()
        exact_matches = [d for d in keyword_docs if d.metadata.get('serial', '').upper() == exact_serial]
        if exact_matches:
            print(f"\n[EXACT MATCH] Serial: {exact_serial}")
            print(f"[RETURN] {len(exact_matches)} document(s)")
            print(f"{'='*70}\n")
            return exact_matches
    
    # ✅ ถ้าเจอ exact match จาก Asset -> return เฉพาะตัวนั้น
    if patterns["assets"]:
        exact_asset = patterns["assets"][0]
        exact_matches = [d for d in keyword_docs if d.metadata.get('asset_no', '') == exact_asset]
        if exact_matches:
            print(f"\n[EXACT MATCH] Asset: {exact_asset}")
            print(f"[RETURN] {len(exact_matches)} document(s)")
            print(f"{'='*70}\n")
            return exact_matches
    
    # ✅ ถ้าถาม Specific Model -> ค้นหาแบบเข้มงวด (ไม่ใช้ semantic search)
    if patterns["specific_model"]:
        specific_model = patterns["specific_model"].upper()
        print(f"\n[STEP 2] Specific Model Search: {specific_model}")
        
        # Filter จาก keyword_docs
        specific_matches = [
            d for d in keyword_docs 
            if specific_model in d.metadata.get('model', '').upper()
        ]
        
        if specific_matches:
            print(f"[STEP 2 RESULT] Found {len(specific_matches)} exact matches in keyword_docs")
            print(f"[RETURN] {len(specific_matches)} document(s)")
            print(f"{'='*70}\n")
            return specific_matches[:20]  # เพิ่มเป็น 20 เผื่อมีเยอะ
        
        # ค้นหาใน database โดยตรง (เฉพาะ inventory)
        print(f"[STEP 2B] Searching in database...")
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            query = """
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE UPPER(cmetadata->>'model') LIKE UPPER(%s)
            AND cmetadata->>'source' = 'inventory'
            LIMIT 20
            """
            cursor.execute(query, (f'%{specific_model}%',))
            rows = cursor.fetchall()
            cursor.close()
            
            if rows:
                print(f"[STEP 2B RESULT] Found {len(rows)} matches in database")
                print(f"[RETURN] {len(rows)} document(s)")
                print(f"{'='*70}\n")
                return [Document(page_content=doc, metadata=meta or {}) for doc, meta in rows]
            else:
                # ถ้าไม่เจออะไรเลย -> return empty
                print(f"[STEP 2B] No matches found for {specific_model}")
                print(f"{'='*70}\n")
                return []
        except Exception as e:
            print(f"[ERROR] Database search failed: {e}")
            return []
    
    # 2. ถ้ามี keyword results -> ใช้ keyword results
    if keyword_docs:
        print(f"\n[STEP 3] Using keyword results")
        print(f"[RETURN] {len(keyword_docs[:10])} document(s)")
        print(f"{'='*70}\n")
        return keyword_docs[:10]
    
    # 3. Semantic Search (เฉพาะ inventory)
    print(f"\n[STEP 4] Semantic Search (fallback)")
    try:
        # ✅ เพิ่ม filter เฉพาะ inventory
        semantic_docs = get_vectorstore().as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,
                "filter": {"source": "inventory"}  # ← เพิ่ม filter
            }
        ).invoke(question)
        print(f"[STEP 4 RESULT] Found {len(semantic_docs)} docs from semantic search")
        
        # Filter ถ้ามี specific_model
        if patterns["specific_model"]:
            specific_model = patterns["specific_model"].upper()
            before_filter = len(semantic_docs)
            semantic_docs = [
                d for d in semantic_docs
                if specific_model in d.metadata.get('model', '').upper()
            ]
            print(f"[FILTER] {before_filter} -> {len(semantic_docs)} docs after filtering for {specific_model}")
        
    except Exception as e:
        print(f"[ERROR] Semantic search failed: {e}")
        semantic_docs = []
    
    # 4. รวมผลลัพธ์
    combined = keyword_docs + semantic_docs
    print(f"\n[STEP 5] Combining results: {len(combined)} total")
    
    # 5. ลบ duplicate
    seen = set()
    unique_docs = []
    for doc in combined:
        key = (
            doc.metadata.get('serial', ''),
            doc.metadata.get('asset_no', ''),
            doc.metadata.get('subject', '')
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    
    print(f"[STEP 5 RESULT] {len(unique_docs)} unique documents")
    print(f"[RETURN] {len(unique_docs[:10])} document(s)")
    print(f"{'='*70}\n")
    return unique_docs[:10]

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

IT_ASSET_PROMPT = ChatPromptTemplate.from_template("""
You are an IT Asset Database. Answer in Thai or English based on user's question.

DATA:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. For EACH item found, you MUST show ALL of these fields:
   - Model (from MODEL field)
   - Serial Number (from SERIAL_NUMBER field) 
   - Asset Number (from ASSET_NUMBER field)
   - Status (from STATUS field)
   - Location (from LOCATION field)

2. Format like this:
   Model: [MODEL value]
   Serial Number: [SERIAL_NUMBER value]
   Asset Number: [ASSET_NUMBER value]  
   Status: [STATUS value]
   Location: [LOCATION value]

3. NEVER skip the Serial Number field
4. If multiple items, show all of them
5. Use simple Thai without fancy formatting

ANSWER (show Serial Number for every item):
""")

SUPPORT_TICKET_PROMPT = ChatPromptTemplate.from_template("""
You are an IT Support Assistant for handling support tickets.
Answer in the SAME LANGUAGE as the question (Thai or English).

## Retrieved Tickets:
{context}

## User Question:
{question}

## Instructions:
1. Read ticket information carefully
2. Summarize the most relevant tickets
3. Show: Subject, Type, Queue, Priority, Tags
4. If multiple tickets match, list them briefly
5. Answer in a helpful, professional tone

Answer:
""")

GENERAL_PROMPT = ChatPromptTemplate.from_template("""
You are a friendly IT Support Assistant.
Answer in the SAME LANGUAGE as the question (Thai or English).

User Question: {question}

Give a helpful, professional response:
""")

# ============================================================================
# CONTEXT FORMATTING
# ============================================================================

def format_inventory_context(docs: List[Document], max_docs: int = 3) -> str:
    """Format inventory documents เป็น context ที่อ่านง่าย"""
    if not docs:
        return "ไม่พบข้อมูลในระบบ"
    
    docs = docs[:max_docs]
    lines = []
    
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        
        model = meta.get('model', '') or 'N/A'
        model_no = meta.get('model_no', '') or 'N/A'
        serial = meta.get('serial', '') or 'N/A'
        asset_no = meta.get('asset_no', '') or 'N/A'
        status = meta.get('status', '') or 'N/A'
        location = meta.get('location', '') or 'N/A'
        
        if not meta.get('serial') or meta.get('serial') == '':
            content = doc.page_content
            serial_match = re.search(r'Serial Number:\s*(.+)', content)
            if serial_match:
                serial = serial_match.group(1).strip()
        
        lines.append(f"ITEM_{i}:")
        lines.append(f"  MODEL={model}")
        lines.append(f"  MODEL_NO={model_no}")
        lines.append(f"  SERIAL_NUMBER={serial}")
        lines.append(f"  ASSET_NUMBER={asset_no}")
        lines.append(f"  STATUS={status}")
        lines.append(f"  LOCATION={location}")
        lines.append("")
    
    result = "\n".join(lines)
    print(f"[CONTEXT DEBUG]\n{result}")
    return result

def format_ticket_context(docs: List[Document], max_docs: int = 5) -> str:
    """Format support ticket documents"""
    if not docs:
        return "ไม่พบ ticket ในระบบ"
    
    docs = docs[:max_docs]
    lines = [f"พบ {len(docs)} tickets ที่เกี่ยวข้อง:\n"]
    
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        
        lines.append(f"\n--- Ticket #{i} ---")
        lines.append(f"Subject: {meta.get('subject', 'N/A')}")
        lines.append(f"Type: {meta.get('type', 'N/A')}")
        lines.append(f"Queue: {meta.get('queue', 'N/A')}")
        lines.append(f"Priority: {meta.get('priority', 'N/A')}")
        lines.append(f"Tags: {meta.get('tags', 'N/A')}")
        
        content_preview = doc.page_content[:200] if doc.page_content else ""
        if len(doc.page_content) > 200:
            content_preview += "..."
        lines.append(f"Details:\n{content_preview}")
    
    return "\n".join(lines)

# ============================================================================
# CHAT HISTORY MANAGEMENT
# ============================================================================

@lru_cache(maxsize=10)
def get_session_history(session_id: str):
    return PostgresChatMessageHistory(
        connection_string=PSYCOPG_CONN_INFO,
        session_id=session_id
    )

# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================

def classify_intent(question: str) -> str:
    """แยกว่าเป็นคำถามเกี่ยวกับ inventory, ticket, หรือ general"""
    q_lower = question.lower()
    
    # Inventory keywords
    inventory_keywords = [
        "serial", "s/n", "sn", "asset", "model", "รุ่น", "เครื่อง", "อุปกรณ์",
        "มี", "เหลือ", "กี่", "จำนวน", "spare", "obsolete", "ค้นหา", "หา",
        "thinkpad", "laptop", "switch", "router", "printer", "computer",
        "location", "ตำแหน่ง", "อยู่ที่", "sriracha", "ศรีราชา",
        "model no", "asset no", "serial number", "2930f", "2930m"
    ]
    
    # Ticket keywords
    ticket_keywords = [
        "ticket", "support", "issue", "problem", "ปัญหา", "คำถาม",
        "help", "assist", "request", "เคส", "แจ้ง", "bug", "error",
        "technical", "queue", "priority", "tag"
    ]
    
    # Check for serial/asset patterns
    if re.search(r'\b[A-Z0-9]{6,20}\b', question.upper()):
        return "inventory"
    
    # Count keyword matches
    inventory_score = sum(1 for k in inventory_keywords if k in q_lower)
    ticket_score = sum(1 for k in ticket_keywords if k in q_lower)
    
    if inventory_score > ticket_score:
        return "inventory"
    elif ticket_score > 0:
        return "ticket"
    else:
        return "general"

# ============================================================================
# MAIN CHAT FUNCTION
# ============================================================================

def chat_with_warehouse_system(
    session_id: str,
    question: str,
    image: bytes | None = None
) -> Generator[str, None, None]:
    """Main chat function with improved retrieval and formatting"""
    
    print("\n" + "="*70)
    print(f"[CHAT START] Session: {session_id}")
    print(f"[CHAT START] Question: {question}")
    print("="*70)
    
    try:
        llm = get_llm()
        history = get_session_history(session_id)
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        intent = classify_intent(question)
        print(f"\n[INTENT] {intent}")
        
        # ============================================================
        # INVENTORY QUERIES
        # ============================================================
        if intent == "inventory":
            print("[CHAT] Processing as INVENTORY query")
            docs = hybrid_retrieve(question)
            
            if not docs:
                print("[CHAT] No documents found")
                yield "🔍 ไม่พบข้อมูลในระบบ\n\n"
                yield "💡 ลองตรวจสอบ:\n"
                yield "• Serial Number ถูกต้องหรือไม่\n"
                yield "• ค้นหาด้วย Model หรือ Asset Number แทน\n"
                return
            
            # สร้าง response
            total_docs = len(docs)
            print(f"[CHAT] Total documents: {total_docs}")
            
            # กำหนดจำนวนที่แสดง
            if total_docs == 1:
                display_limit = 1
            elif total_docs <= 10:
                display_limit = total_docs
            else:
                display_limit = 10
            
            print(f"[CHAT] Will display: {display_limit} items")
            
            # Header
            if total_docs == 1:
                header = "พบข้อมูล 1 รายการ:\n\n"
            elif total_docs <= 10:
                header = f"พบข้อมูล {total_docs} รายการ:\n\n"
            else:
                header = f"พบข้อมูล {total_docs} รายการ (แสดง 10 รายการแรก):\n\n"
            
            yield header
            print(f"[CHAT] ✓ Sent header")
            
            # เตรียมเก็บข้อมูลสำหรับ history
            full_response_parts = [header]
            
            # แสดงแต่ละรายการ + เก็บไว้ใน history
            for i, doc in enumerate(docs[:display_limit], 1):
                meta = doc.metadata
                print(f"[CHAT] Processing item {i}/{display_limit}")
                print(f"[CHAT] Metadata: {meta}")
                
                # สร้างข้อความของรายการนี้
                item_header = f"📦 รายการที่ {i}:\n"
                yield item_header
                full_response_parts.append(item_header)
                print(f"[CHAT] ✓ Sent item {i} header")
                
                # Model (บังคับต้องมี)
                model = meta.get('model', 'N/A')
                model_line = f"• Model: {model}\n"
                yield model_line
                full_response_parts.append(model_line)
                print(f"[CHAT] ✓ Sent model: {model}")
                
                # Model No (ถ้ามี)
                model_no = meta.get('model_no', '')
                if model_no and model_no.strip() and model_no != 'N/A':
                    model_no_line = f"• Model No: {model_no}\n"
                    yield model_no_line
                    full_response_parts.append(model_no_line)
                    print(f"[CHAT] ✓ Sent model_no: {model_no}")
                
                # Serial Number (แสดงเสมอ)
                serial = meta.get('serial', '')
                if serial and serial.strip() and serial != 'N/A':
                    serial_line = f"• Serial Number: {serial}\n"
                    yield serial_line
                    full_response_parts.append(serial_line)
                    print(f"[CHAT] ✓ Sent serial: {serial}")
                else:
                    no_serial_line = f"• Serial Number: ไม่มีข้อมูล\n"
                    yield no_serial_line
                    full_response_parts.append(no_serial_line)
                    print(f"[CHAT] ✓ Sent: ไม่มี serial")
                
                # Asset Number (ถ้ามี)
                asset_no = meta.get('asset_no', '')
                if asset_no and asset_no.strip() and asset_no != 'N/A':
                    asset_line = f"• Asset Number: {asset_no}\n"
                    yield asset_line
                    full_response_parts.append(asset_line)
                    print(f"[CHAT] ✓ Sent asset: {asset_no}")
                
                # Status (ถ้ามี)
                status = meta.get('status', '')
                if status and status.strip() and status != 'N/A':
                    status_line = f"• Status: {status}\n"
                    yield status_line
                    full_response_parts.append(status_line)
                    print(f"[CHAT] ✓ Sent status: {status}")
                
                # Location (ถ้ามี)
                location = meta.get('location', '')
                if location and location.strip() and location != 'N/A':
                    location_line = f"• Location: {location}\n"
                    yield location_line
                    full_response_parts.append(location_line)
                    print(f"[CHAT] ✓ Sent location: {location}")
                
                # เว้นบรรทัด
                separator = "\n"
                yield separator
                full_response_parts.append(separator)
                print(f"[CHAT] ✓ Sent item {i} separator")
            
            # Footer (ถ้ามีมากกว่า 10)
            if total_docs > 10:
                remaining = total_docs - 10
                footer = f"\n💡 มีอีก {remaining} รายการที่ไม่แสดง\n" \
                        f"🔍 ลองค้นหาแบบเจาะจงมากขึ้น:\n" \
                        f"   • ระบุ Serial Number (เช่น 'หา serial TW33KR41B2')\n" \
                        f"   • ระบุ Asset Number (เช่น 'หา asset 10053061')\n" \
                        f"   • ระบุ Location (เช่น 'อุปกรณ์ที่ Customs Building')\n"
                yield footer
                full_response_parts.append(footer)
                print(f"[CHAT] ✓ Sent footer")
            
            # บันทึก history แบบเต็ม
            full_response = "".join(full_response_parts)
            history.add_user_message(question)
            history.add_ai_message(full_response)
            
            print(f"[CHAT] ✅ Response complete: displayed {display_limit}/{total_docs} items")
            return
        
        # ============================================================
        # TICKET QUERIES
        # ============================================================
        elif intent == "ticket":
            print("[CHAT] Processing as TICKET query")
            docs = hybrid_retrieve(question)
            
            if not docs:
                yield "🔍 ไม่พบ tickets ที่เกี่ยวข้อง\n"
                return
            
            # Filter เฉพาะ tickets
            ticket_docs = [d for d in docs if d.metadata.get('source') == 'support_tickets']
            
            if not ticket_docs:
                yield "🔍 ไม่พบ tickets ที่เกี่ยวข้อง\n"
                return
            
            context = format_ticket_context(ticket_docs)
            print(f"[CHAT] Ticket context prepared, {len(ticket_docs)} tickets")
            
            chain = (
                {
                    "context": lambda _: context,
                    "question": RunnablePassthrough()
                }
                | SUPPORT_TICKET_PROMPT
                | llm
            )
            
            print("[CHAT] Starting LLM stream...")
            full_response = ""
            for chunk in chain.stream(question):
                content = getattr(chunk, "content", str(chunk))
                full_response += content
                yield content
            
            history.add_user_message(question)
            history.add_ai_message(full_response)
            return
        
        # ============================================================
        # GENERAL QUERIES
        # ============================================================
        else:
            print("[CHAT] Processing as GENERAL query")
            chain = (
                {
                    "question": RunnablePassthrough()
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
    
    except Exception as e:
        print(f"\n[ERROR] Chat failed: {e}")
        import traceback
        traceback.print_exc()
        yield f"❌ เกิดข้อผิดพลาด: {str(e)}\n"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_session_history(session_id: str):
    """ลบ chat history"""
    history = get_session_history(session_id)
    history.clear()
    get_session_history.cache_clear()

def cleanup_resources():
    """ปิด connections"""
    global _vectorstore, _embeddings, _llm, _retriever, _db_conn
    if _db_conn and not _db_conn.closed:
        _db_conn.close()
    _vectorstore = None
    _embeddings = None
    _llm = None
    _retriever = None
    _db_conn = None
    get_session_history.cache_clear()