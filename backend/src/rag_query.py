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
# KEYWORD EXTRACTION (เดิม - ไม่แก้)
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
    
    # ✅ ตรวจจับ Specific Model
    specific_model_patterns = [
        r'\b(FR-\d+)\b',
        r'\b(2930F)\b',
        r'\b(2930M)\b',
        r'\b(JL\d+[A-Z])\b',
        r'\b(YOGA\s+\S+)',
        r'\b(THINKPAD\s+\S+)',
        r'\b(THINKCENTRE\s+\S+)',
        r'\b(ELITEBOOK\s+\S+)',
        r'\b(OPTIPLEX\s+\S+)',
        r'\b(MACBOOK\s+\S+)',
        r'\b(HP\s+\S+\s+\S+)',
        r'\b([A-Z]+\d+[A-Z]*\s*\d*[A-Z]*)',
    ]
    
    for pattern in specific_model_patterns:
        match = re.search(pattern, question.upper())
        if match:
            patterns["specific_model"] = match.group(1).strip()
            print(f"[PATTERN MATCH] Detected specific model: {patterns['specific_model']}")
            break
    
    # Model keywords
    model_keywords = [
        "fr-4080", "2930f", "2930m",
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
# HYBRID RETRIEVAL SYSTEM (เดิม - ไม่แก้)
# ============================================================================

def keyword_search_direct(patterns: dict) -> List[Document]:
    """ค้นหาด้วย SQL โดยตรงจาก metadata (เฉพาะ inventory)"""
    all_docs = []
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        base_filter = "AND cmetadata->>'source' = 'inventory'"
        
        # Specific Model
        if patterns.get("specific_model"):
            specific_model = patterns["specific_model"]
            print(f"[SQL SEARCH] Specific Model: {specific_model}")
            query = f"""
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE UPPER(cmetadata->>'model') LIKE UPPER(%s)
            {base_filter}
            LIMIT 100
            """
            cursor.execute(query, (f'%{specific_model}%',))
            rows = cursor.fetchall()
            print(f"[SQL RESULT] Found {len(rows)} matches for specific model {specific_model}")
            for doc_content, metadata in rows:
                all_docs.append(Document(page_content=doc_content, metadata=metadata or {}))
        
        # Serial Number
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
        
        # Asset Number
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
        
        # Model No
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
        
        # Location
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
    print(f"[STEP 1 RESULT] Found {len(keyword_docs)} docs from keyword search (before dedup)")
    
    # ลบ duplicate
    seen_serials = set()
    unique_keyword_docs = []
    for doc in keyword_docs:
        serial = (doc.metadata.get('serial') or '').strip().upper()
        asset = (doc.metadata.get('asset_no') or '').strip()
        
        if serial:
            unique_key = f"serial_{serial}"
        elif asset:
            unique_key = f"asset_{asset}"
        else:
            unique_key = f"row_{doc.metadata.get('row', '')}_{doc.metadata.get('model', '')}"
        
        if unique_key not in seen_serials:
            seen_serials.add(unique_key)
            unique_keyword_docs.append(doc)
            print(f"[DEDUP] Added: {unique_key}")
        else:
            print(f"[DEDUP] Skipped duplicate: {unique_key}")
    
    keyword_docs = unique_keyword_docs
    print(f"[STEP 1 RESULT] After dedup: {len(keyword_docs)} unique docs")
    
    # Exact match สำหรับ Serial
    if patterns["serials"]:
        exact_serial = patterns["serials"][0].upper()
        exact_matches = [d for d in keyword_docs if (d.metadata.get('serial') or '').upper() == exact_serial]
        if exact_matches:
            print(f"\n[EXACT MATCH] Serial: {exact_serial}")
            print(f"[RETURN] {len(exact_matches)} document(s)")
            print(f"{'='*70}\n")
            return exact_matches
    
    # Exact match สำหรับ Asset
    if patterns["assets"]:
        exact_asset = patterns["assets"][0]
        exact_matches = [d for d in keyword_docs if (d.metadata.get('asset_no') or '') == exact_asset]
        if exact_matches:
            print(f"\n[EXACT MATCH] Asset: {exact_asset}")
            print(f"[RETURN] {len(exact_matches)} document(s)")
            print(f"{'='*70}\n")
            return exact_matches
    
    # Specific Model
    if patterns["specific_model"]:
        specific_model = patterns["specific_model"].upper()
        print(f"\n[STEP 2] Specific Model Search: {specific_model}")
        
        if keyword_docs:
            print(f"[STEP 2 RESULT] Found {len(keyword_docs)} matches")
            print(f"[RETURN] {len(keyword_docs)} document(s)")
            print(f"{'='*70}\n")
            return keyword_docs
        else:
            print(f"[STEP 2] No matches found for {specific_model}")
            print(f"{'='*70}\n")
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
        semantic_docs = get_vectorstore().as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,
                "filter": {"source": "inventory"}
            }
        ).invoke(question)
        print(f"[STEP 4 RESULT] Found {len(semantic_docs)} docs from semantic search")
        
        if patterns["specific_model"]:
            specific_model = patterns["specific_model"].upper()
            before_filter = len(semantic_docs)
            semantic_docs = [
                d for d in semantic_docs
                if specific_model in (d.metadata.get('model') or '').upper()
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
# IMPROVED INTENT CLASSIFICATION (ปรับเฉพาะ ticket)
# ============================================================================

def classify_intent(question: str) -> str:
    """แยกว่าเป็นคำถามเกี่ยวกับ inventory, ticket, หรือ general"""
    q_lower = question.lower()
    
    # ============================================================
    # INVENTORY KEYWORDS (เดิม - ไม่แก้)
    # ============================================================
    inventory_keywords = [
        "serial", "s/n", "sn", "asset", "model", "รุ่น", "เครื่อง", "อุปกรณ์",
        "มี", "เหลือ", "กี่", "จำนวน", "spare", "obsolete", "ค้นหา", "หา",
        "thinkpad", "laptop", "switch", "router", "printer", "computer",
        "location", "ตำแหน่ง", "อยู่ที่", "sriracha", "ศรีราชา",
        "model no", "asset no", "serial number", "2930f", "2930m", "fr-4080"
    ]
    
    # ============================================================
    # IMPROVED TICKET KEYWORDS (ปรับปรุงใหม่)
    # ============================================================
    ticket_keywords = [
        # คำหลัก
        "ticket", "support", "issue", "problem", "ปัญหา", "คำถาม",
        "help", "assist", "request", "เคส", "แจ้ง", "bug", "error",
        
        # คำถามแบบ how-to
        "ทำยังไง", "วิธี", "how to", "how do", "how can", "ขั้นตอน",
        
        # คำที่บ่งบอกว่าเป็นปัญหา
        "แก้", "fix", "solve", "แก้ไข", "รีเซ็ต", "reset",
        "ไม่ทำงาน", "not working", "broken", "เสีย", "ไม่ได้",
        
        # Technical issues
        "vpn", "network", "connection", "password", "login", "เข้าสู่ระบบ",
        "install", "ติดตั้ง", "configure", "setup", "เชื่อมต่อ",
        "email", "อีเมล", "security", "ความปลอดภัย",
        
        # Priority/Queue related
        "urgent", "ด่วน", "priority", "สำคัญ", "high", "สูง",
        
        # Tags ที่มักเจอใน ticket
        "audio", "video", "เสียง", "ภาพ", "performance", "ประสิทธิภาพ",
        "compatibility", "เข้ากันได้", "automation", "อัตโนมัติ",
        
        # Support-specific
        "technical support", "it support", "ซัพพอร์ต", "helpdesk"
    ]
    
    # Check for serial/asset patterns (แน่นอนว่าเป็น inventory)
    if re.search(r'\b[A-Z0-9]{6,20}\b', question.upper()):
        return "inventory"
    
    # Count keyword matches
    inventory_score = sum(1 for k in inventory_keywords if k in q_lower)
    ticket_score = sum(1 for k in ticket_keywords if k in q_lower)
    
    print(f"[INTENT] Scores - Inventory: {inventory_score}, Ticket: {ticket_score}")
    
    # ถ้า ticket score สูงกว่าอย่างชัดเจน
    if ticket_score > inventory_score and ticket_score >= 2:
        print(f"[INTENT] Classified as: ticket")
        return "ticket"
    
    # ถ้า inventory score สูงกว่า
    if inventory_score > ticket_score:
        print(f"[INTENT] Classified as: inventory")
        return "inventory"
    
    # ถ้ามี ticket keyword อย่างน้อย 1 ตัว
    if ticket_score > 0:
        print(f"[INTENT] Classified as: ticket (has keywords)")
        return "ticket"
    
    # Default
    print(f"[INTENT] Classified as: general")
    return "general"

# ============================================================================
# ✨ NEW: IMPROVED TICKET RETRIEVAL
# ============================================================================

def retrieve_tickets(question: str) -> List[Document]:
    """
    ค้นหา support tickets ด้วยวิธีที่ดีขึ้น
    - เพิ่มจำนวนผลลัพธ์
    - ค้นหาทั้ง subject และ content
    """
    print(f"\n{'='*70}")
    print(f"[TICKET RETRIEVE] Starting ticket search")
    print(f"Question: {question}")
    print(f"{'='*70}")
    
    try:
        # ค้นหาด้วย semantic search (เพิ่มจาก 10 เป็น 20)
        semantic_docs = get_vectorstore().similarity_search(
            question,
            k=20,  # เพิ่มจำนวน
            filter={"source": "support_tickets"}
        )
        
        print(f"[TICKET RETRIEVE] Found {len(semantic_docs)} tickets from semantic search")
        
        # แสดง debug info
        if semantic_docs:
            print("\n[TICKET RETRIEVE] Top 5 results:")
            for i, doc in enumerate(semantic_docs[:5], 1):
                subject = doc.metadata.get('subject', 'N/A')
                typ = doc.metadata.get('type', 'N/A')
                tags = doc.metadata.get('tags', 'N/A')
                print(f"  {i}. Subject: {subject}")
                print(f"     Type: {typ}, Tags: {tags}")
        
        print(f"{'='*70}\n")
        return semantic_docs
        
    except Exception as e:
        print(f"[ERROR] Ticket retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return []

# ============================================================================
# PROMPT TEMPLATES (เดิม - ไม่แก้)
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
# CONTEXT FORMATTING (เดิม - ไม่แก้)
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

def clean_text_formatting(text: str) -> str:
    """
    ทำความสะอาด text โดยลบ HTML tags, underscores, formatting characters และอักขระแปลกปลอม
    """
    if not text:
        return ""
    
    import html
    import unicodedata
    
    # Decode HTML entities (เช่น &amp; → &)
    text = html.unescape(text)
    
    # ลบ HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # ลบ underscores ที่ใช้สำหรับ formatting (underline)
    text = re.sub(r'_{2,}', ' ', text)  # ลบ __ หรือ ___ เป็นต้นไป
    
    # ลบ markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold** → bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic* → italic
    text = re.sub(r'__([^_]+)__', r'\1', text)      # __underline__ → underline
    
    # ✅ ลบอักขระที่ไม่ใช่ภาษาไทย อังกฤษ ตัวเลข และเครื่องหมายพื้นฐาน
    # Keep: Thai (0E00-0E7F), English (0000-007F), digits, common punctuation
    allowed_chars = []
    for char in text:
        # Check if character is in allowed ranges
        code_point = ord(char)
        
        # Thai characters (U+0E00 to U+0E7F)
        # Basic Latin (U+0000 to U+007F) - English, numbers, punctuation
        # General Punctuation (U+2000 to U+206F)
        # Spaces and common symbols
        if (0x0E00 <= code_point <= 0x0E7F or  # Thai
            0x0020 <= code_point <= 0x007E or  # Basic Latin (printable)
            code_point in [0x000A, 0x000D] or  # Line breaks
            0x2000 <= code_point <= 0x206F):   # Punctuation
            allowed_chars.append(char)
        else:
            # แทนที่อักขระแปลกด้วยช่องว่าง
            allowed_chars.append(' ')
    
    text = ''.join(allowed_chars)
    
    # ลบช่องว่างซ้ำซ้อน
    text = re.sub(r'\s+', ' ', text)
    
    # ลบช่องว่างหน้าหลัง
    text = text.strip()
    
    return text

def format_ticket_context(docs: List[Document], max_docs: int = 5) -> str:
    """Format support ticket documents (with cleaned text)"""
    if not docs:
        return "ไม่พบ ticket ในระบบ"
    
    docs = docs[:max_docs]
    lines = [f"พบ {len(docs)} tickets ที่เกี่ยวข้อง:\n"]
    
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        
        lines.append(f"\n--- Ticket #{i} ---")
        
        # ✅ ทำความสะอาด subject
        subject = clean_text_formatting(meta.get('subject', 'N/A'))
        lines.append(f"Subject: {subject}")
        
        lines.append(f"Type: {meta.get('type', 'N/A')}")
        lines.append(f"Queue: {meta.get('queue', 'N/A')}")
        lines.append(f"Priority: {meta.get('priority', 'N/A')}")
        lines.append(f"Tags: {meta.get('tags', 'N/A')}")
        
        # ✅ ทำความสะอาด content
        content_preview = clean_text_formatting(doc.page_content[:300]) if doc.page_content else ""
        if len(doc.page_content) > 300:
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
# MAIN CHAT FUNCTION (ปรับเฉพาะส่วน ticket)
# ============================================================================

def chat_with_warehouse_system(
    session_id: str,
    question: str,
    image: bytes | None = None
) -> Generator[str, None, None]:
    """Main chat function with improved ticket handling"""
    
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
        # INVENTORY QUERIES (เดิม - ไม่แก้)
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
            
            total_docs = len(docs)
            print(f"[CHAT] Total documents: {total_docs}")
            
            display_limit = total_docs
            print(f"[CHAT] Will display: {display_limit} items")
            
            if total_docs == 1:
                header = "พบข้อมูล 1 รายการ:\n\n"
            else:
                header = f"พบข้อมูล {total_docs} รายการ:\n\n"
            
            yield header
            print(f"[CHAT] ✓ Sent header")
            
            full_response_parts = [header]
            
            for i, doc in enumerate(docs[:display_limit], 1):
                meta = doc.metadata
                print(f"[CHAT] Processing item {i}/{display_limit}")
                print(f"[CHAT] Metadata: {meta}")
                
                # ✅ Header ที่สวยขึ้น
                item_header = f"{'='*3}\n"
                item_header += f"รายการที่ {i}:\n"
                item_header += f"{'='*3}\n"
                yield item_header
                full_response_parts.append(item_header)
                print(f"[CHAT] ✓ Sent item {i} header")
                
                # Model
                model = meta.get('model', 'N/A')
                model_line = f"• Model: {model}\n"
                yield model_line
                full_response_parts.append(model_line)
                print(f"[CHAT] ✓ Sent model: {model}")
                
                # Model No
                model_no = meta.get('model_no', '')
                if model_no and model_no.strip() and model_no != 'N/A':
                    model_no_line = f"• Model No: {model_no}\n"
                    yield model_no_line
                    full_response_parts.append(model_no_line)
                    print(f"[CHAT] ✓ Sent model_no: {model_no}")
                
                # Serial Number
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
                
                # Asset Number
                asset_no = meta.get('asset_no', '')
                if asset_no and asset_no.strip() and asset_no != 'N/A':
                    asset_line = f"• Asset Number: {asset_no}\n"
                    yield asset_line
                    full_response_parts.append(asset_line)
                    print(f"[CHAT] ✓ Sent asset: {asset_no}")
                
                # Status
                status = meta.get('status', '')
                if status and status.strip() and status != 'N/A':
                    status_line = f"• Status: {status}\n"
                    yield status_line
                    full_response_parts.append(status_line)
                    print(f"[CHAT] ✓ Sent status: {status}")
                
                # Location
                location = meta.get('location', '')
                if location and location.strip() and location != 'N/A':
                    location_line = f"• Location: {location}\n"
                    yield location_line
                    full_response_parts.append(location_line)
                    print(f"[CHAT] ✓ Sent location: {location}")
                
                # ✅ เพิ่มช่องว่างระหว่างรายการ (2 บรรทัด)
                separator = "\n\n"
                yield separator
                full_response_parts.append(separator)
                print(f"[CHAT] ✓ Sent item {i} separator")
            
            full_response = "".join(full_response_parts)
            history.add_user_message(question)
            history.add_ai_message(full_response)
            
            print(f"[CHAT] ✅ Response complete: displayed {display_limit}/{total_docs} items")
            return
        
        # ============================================================
        # TICKET QUERIES (ปรับปรุงใหม่)
        # ============================================================
        elif intent == "ticket":
            print("[CHAT] Processing as TICKET query")
            
            # ใช้ฟังก์ชันใหม่ที่ปรับปรุงแล้ว
            ticket_docs = retrieve_tickets(question)
            
            if not ticket_docs:
                print("[CHAT] No tickets found")
                yield "🔍 ไม่พบ Support Tickets ที่เกี่ยวข้อง\n\n"
                yield "💡 ลองถามคำถามแบบนี้:\n"
                yield "• 'มีปัญหาเกี่ยวกับ VPN บ้างไหม'\n"
                yield "• 'ticket เกี่ยวกับ network มีอะไรบ้าง'\n"
                yield "• 'แก้ปัญหา login ยังไง'\n"
                yield "• 'วิธีติดตั้ง software'\n"
                yield "• 'ปัญหา printer มีอะไรบ้าง'\n"
                return

            # สร้าง context (เพิ่มจาก 5 เป็น 10)
            context = format_ticket_context(ticket_docs, max_docs=10)
            print(f"[CHAT] Ticket context prepared")
            
            # ส่งไป LLM
            chain = (
                {
                    "context": lambda _: context,
                    "question": RunnablePassthrough()
                }
                | SUPPORT_TICKET_PROMPT
                | llm
            )
            
            print("[CHAT] Starting LLM stream for tickets...")
            full_response = ""
            for chunk in chain.stream(question):
                content = getattr(chunk, "content", str(chunk))
                full_response += content
                yield content
            
            history.add_user_message(question)
            history.add_ai_message(full_response)
            print(f"[CHAT] ✅ Ticket response complete")
            return
        
        # ============================================================
        # GENERAL QUERIES (เดิม - ไม่แก้)
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