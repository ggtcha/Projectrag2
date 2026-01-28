from typing import Generator, List, Dict, Optional
import os
import re
from dotenv import load_dotenv
from functools import lru_cache
from datetime import datetime
from contextlib import contextmanager
import psycopg2

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

# Database settings
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DATABASE = os.getenv("PG_DATABASE")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:0.5b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = "http://localhost:11434"

# Response mode: "rule" uses pattern extraction, "llm" uses AI generation
TICKET_RESPONSE_MODE = os.getenv("TICKET_RESPONSE_MODE", "rule")

# Logging level: "debug", "info", "error", "none"
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# Database connection strings
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

# Search result limits
MAX_KEYWORD_RESULTS = 100
MAX_EXACT_MATCHES = 3
MAX_LOCATION_MATCHES = 5
MAX_SEMANTIC_RESULTS = 10
MAX_FINAL_RESULTS = 10

# Text processing constraints
MIN_SOLUTION_LENGTH = 15
MAX_SOLUTION_LENGTH = 150
MAX_SUBJECT_LENGTH = 100
MIN_SUBJECT_LENGTH = 10

#============================================================================
# GLOBAL INSTANCES
#============================================================================

_embeddings = None
_vectorstore = None
_retriever = None
_llm = None
_db_conn = None

#============================================================================
# LOGGING UTILITY
#============================================================================

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

def get_retriever():
    global _retriever
    if _retriever is None:
        log("[LLM] Creating retriever...", "debug")
        _vectorstore = get_vectorstore()
        _retriever = _vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    return _retriever

def get_llm():
    global _llm
    if _llm is None:
        log("[LLM] Connecting to LLM...", "debug")
        _llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.1,
            stream=False,
            base_url=OLLAMA_BASE_URL,
            num_ctx=2048,
            num_predict=128,
            repeat_penalty=1.1,
            top_p=0.9,
            top_k=40
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
        return
    
    import html
    
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'_{2,}', ' ', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    
    allowed_chars = []
    for char in text:
        code_point = ord(char)
        
        if (0x0E00 <= code_point <= 0x0E7F or
            0x0020 <= code_point <= 0x007E or
            code_point in [0x000A, 0x000D] or
            0x2000 <= code_point <= 0x206F):
            allowed_chars.append(char)
        else:
            allowed_chars.append(' ')
    
    text = ''.join(allowed_chars)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

#============================================================================
# INTENT CLASSIFICATION
#============================================================================

def classify_intent(question: str) -> str:
    q_lower = question.lower()
    
    # Hard inventory indicators
    if re.search(r'(serial|serial number|sn|asset|asset number|ขอserial|ขอ serial)', q_lower):
        log(f"[Intent] inventory (serial request)", "debug")
        return "inventory"
    
    # Code-like patterns
    if is_code_like(question):
        log(f"[Intent] inventory (code pattern)", "debug")
        return "inventory"
    
    # Strong ticket indicators
    strong_ticket_patterns = [
        r'ทำ(อย่างไร|ยังไง|ไง)',
        r'วิธี(การ|แก้|ใช้)',
        r'(how to|how do|how can)',
        r'(แก้|fix|solve|แก้ไข)',
        r'ปัญหา',
        r'(ไม่ทำงาน|not working|broken|เสีย)',
        r'(ลูกค้า|user|ผู้ใช้)(แจ้ง|report|ถาม)',
        r'(ticket|issue|problem|help|support)',
        r'(รีเซ็ต|reset|configure|setup|install|ติดตั้ง)',
        r'(vpn|network|connection|password|login|email)',
        r'(เชื่อมต่อ|เข้าสู่ระบบ|ความปลอดภัย)',
    ]
    
    for pattern in strong_ticket_patterns:
        if re.search(pattern, q_lower):
            log(f"[Intent] ticket (matched: {pattern})", "debug")
            return "ticket"
    
    # Pattern-based scoring
    patterns = extract_search_patterns(question)
    
    inventory_score = (
        len(patterns["serials"]) * 3 +
        len(patterns["assets"]) * 3 +
        len(patterns["model_nos"]) * 2 +
        len(patterns["models"]) * 2 +
        len(patterns["locations"]) * 1
    )
    
    # Keyword scoring
    ticket_keywords = [
        "ช่วย", "assist", "support", "help", "แจ้ง", "report",
        "ปัญหา", "problem", "issue", "แก้", "fix", "solve",
        "ติดตั้ง", "install", "configure", "setup"
    ]
    ticket_score = sum(1 for k in ticket_keywords if k in q_lower)
    
    log(f"[Intent] scores - Inventory: {inventory_score}, Ticket: {ticket_score}", "debug")
    
    if inventory_score >= 3:
        log(f"[Intent] inventory (strong patterns)", "debug")
        return "inventory"
    
    if ticket_score >= 2:
        log(f"[Intent] ticket (strong keywords)", "debug")
        return "ticket"
    
    if inventory_score > ticket_score:
        return "inventory"
    elif ticket_score > inventory_score:
        return "ticket"
    
    log(f"[Intent] general", "debug")
    return "general"

#============================================================================
# KEYWORD EXTRACTION
#============================================================================

def extract_search_patterns(question: str) -> Dict[str, list]:
    patterns = {
        "serials": [],
        "assets": [],
        "models": [],
        "model_nos": [],
        "locations": [],
        "keywords": [],
        "specific_model": None
    }
    
    device_words = {
        'PRINTER', 'MACBOOK', 'THINKPAD', 'LAPTOP', 'SWITCH', 
        'ROUTER', 'DESKTOP', 'MONITOR', 'KEYBOARD', 'SCANNER'
    }
    
    # Extract Serial Numbers (8-20 alphanumeric)
    potential_serials = re.findall(r'\b[A-Z0-9]{8,20}\b', question.upper())
    serials = [s for s in potential_serials if s not in device_words]
    patterns["serials"].extend(serials)
    
    # Extract Asset Numbers (7-10 digits)
    assets = re.findall(r'\b\d{7,10}\b', question)
    patterns["assets"].extend(assets)
    
    # Extract Model Numbers (H2-PNCN pattern)
    model_nos = re.findall(r'\b[A-Z0-9]{2,3}-[A-Z0-9-]+\b', question.upper())
    patterns["model_nos"].extend(model_nos)
    
    # Detect Specific Model
    q_lower = question.lower()
    q_upper = question.upper()
    
    model_pattern_checks = [
        r'\b([A-Z0-9]{2,3}-[A-Z0-9]+)\b',
        r'\b(FR-\d+)\b',
        r'\b(2930F|2930M)\b',
        r'\b(JL\d+[A-Z])\b',
        r'\b(THINKPAD\w*)\b',
        r'\b(ELITEBOOK\w*)\b',
    ]
    
    for pattern in model_pattern_checks:
        match = re.search(pattern, q_upper)
        if match:
            detected = match.group(1).strip()
            patterns["specific_model"] = detected
            patterns["models"].append(detected.lower())
            log(f"[Pattern] Specific model: {detected}", "debug")
            break
    
    # Use entire question as model if code-like
    if not patterns["specific_model"] and is_code_like(question):
        clean_q = re.sub(r'\s+', '', question.upper())
        patterns["specific_model"] = clean_q
        patterns["models"].append(clean_q.lower())
        log(f"[Pattern] Full query as model: {clean_q}", "debug")
    
    # Extract Model Keywords
    model_keywords = [
        "fr-4080", "2930f", "2930m", "h2-pncn",
        "thinkpad", "thinkcentre", "thinkstation", 
        "switch", "router", "beacon",
        "gateway", "access point", "ups",
        "elitebook", "optiplex", "prodesk",
    ]
    
    for mk in model_keywords:
        if mk in q_lower:
            if mk not in patterns["models"]:
                patterns["models"].append(mk)
    
    # Extract Location Keywords
    location_keywords = [
        "sriracha", "ศรีราชา", 
        "chonburi", "ชลบุรี",
        "custom", "customs",
        "server room", "building",
    ]
    
    for lk in location_keywords:
        if lk in q_lower:
            patterns["locations"].append(lk)
    
    # Extract Status Keywords
    if any(k in q_lower for k in ["spare", "พร้อมใช้", "สำรอง", "ว่าง"]):
        patterns["keywords"].append("spare")
    
    if any(k in q_lower for k in ["obsolete", "เลิกใช้", "เสื่อม"]):
        patterns["keywords"].append("obsolete")
    
    log(f"[Pattern] Extracted: {patterns}", "debug")
    return patterns

#============================================================================
# KEYWORD SEARCH WITH SQL
#============================================================================

def keyword_search_direct(patterns: Dict[str, list]) -> List[Document]:
    all_docs = []
    
    try:
        with get_db_cursor() as cursor:
            base_filter = "AND cmetadata->>'source' = 'inventory'"
            
            # Search by Specific Model
            if patterns.get("specific_model"):
                specific_model = patterns["specific_model"]
                log(f"[SQL] Searching specific model: {specific_model}", "debug")
                query = f"""
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE (
                    UPPER(cmetadata->>'model') LIKE UPPER(%s)
                    OR UPPER(cmetadata->>'model_no') LIKE UPPER(%s)
                )
                {base_filter}
                LIMIT {MAX_KEYWORD_RESULTS}
                """
                cursor.execute(query, (f'%{specific_model}%', f'%{specific_model}%'))
                rows = cursor.fetchall()
                log(f"[SQL] Found {len(rows)} matches", "debug")
                all_docs.extend([
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ])
            
            # Search by Model
            for model in patterns["models"]:
                log(f"[SQL] Searching model: {model}", "debug")
                query = f"""
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE (
                    LOWER(cmetadata->>'model') LIKE LOWER(%s)
                    OR LOWER(cmetadata->>'model_no') LIKE LOWER(%s)
                )
                {base_filter}
                LIMIT 20
                """
                cursor.execute(query, (f'%{model}%', f'%{model}%'))
                rows = cursor.fetchall()
                all_docs.extend([
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ])
            
            # Search by Serial Number
            for serial in patterns["serials"]:
                log(f"[SQL] Searching serial: {serial}", "debug")
                query = f"""
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE UPPER(cmetadata->>'serial') = UPPER(%s)
                {base_filter}
                LIMIT {MAX_EXACT_MATCHES}
                """
                cursor.execute(query, (serial,))
                rows = cursor.fetchall()
                all_docs.extend([
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ])
            
            # Search by Asset Number
            for asset in patterns["assets"]:
                log(f"[SQL] Searching asset: {asset}", "debug")
                query = f"""
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE cmetadata->>'asset_no' = %s
                {base_filter}
                LIMIT {MAX_EXACT_MATCHES}
                """
                cursor.execute(query, (asset,))
                rows = cursor.fetchall()
                all_docs.extend([
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ])
            
            # Search by Model No
            for model_no in patterns["model_nos"]:
                query = f"""
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE UPPER(cmetadata->>'model_no') LIKE UPPER(%s)
                {base_filter}
                LIMIT {MAX_EXACT_MATCHES}
                """
                cursor.execute(query, (f'%{model_no}%',))
                rows = cursor.fetchall()
                all_docs.extend([
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ])
            
            # Search by Location
            for loc in patterns["locations"]:
                query = f"""
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE LOWER(cmetadata->>'location') LIKE LOWER(%s)
                {base_filter}
                LIMIT {MAX_LOCATION_MATCHES}
                """
                cursor.execute(query, (f'%{loc}%',))
                rows = cursor.fetchall()
                all_docs.extend([
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ])
                
    except Exception as e:
        log(f"[ERROR] Keyword search failed: {e}", "error")
    
    return all_docs

#============================================================================
# HYBRID RETRIEVAL SYSTEM
#============================================================================

def hybrid_retrieve(question: str) -> List[Document]:
    log(f"[Retrieve] Query: {question}", "info")
    
    # Exact code match for model/serial/asset
    if is_code_like(question):
        compact = re.sub(r'\s+', '', question.upper())
        log(f"[Retrieve] Code pattern detected: {compact}", "debug")

        try:
            with get_db_cursor() as cursor:
                query = """
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE (
                    UPPER(cmetadata->>'serial') = %s
                    OR UPPER(cmetadata->>'asset_no') = %s
                    OR UPPER(cmetadata->>'model_no') = %s
                    OR UPPER(cmetadata->>'model') = %s
                )
                AND cmetadata->>'source' = 'inventory'
                LIMIT %s
                """
                
                cursor.execute(query, (compact, compact, compact, compact, MAX_EXACT_MATCHES))
                rows = cursor.fetchall()
                
                if rows:
                    log(f"[Retrieve] Found {len(rows)} exact matches", "info")
                    return [
                        Document(page_content=doc, metadata=meta or {})
                        for doc, meta in rows
                    ]
                
                # Fallback to LIKE search
                log(f"[Retrieve] Trying fuzzy match...", "debug")
                query = """
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE (
                    UPPER(cmetadata->>'model') LIKE %s
                    OR UPPER(cmetadata->>'model_no') LIKE %s
                )
                AND cmetadata->>'source' = 'inventory'
                LIMIT 50
                """
                cursor.execute(query, (f'%{compact}%', f'%{compact}%'))
                rows = cursor.fetchall()
                
                if not rows:
                    log("[Retrieve] No match found", "info")
                    return []
                
                log(f"[Retrieve] Found {len(rows)} fuzzy matches", "info")
                return [
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ]
        except Exception as e:
            log(f"[ERROR] Code search failed: {e}", "error")
            return []
    
    # Extract patterns
    patterns = extract_search_patterns(question)
    
    # Keyword search
    keyword_docs = keyword_search_direct(patterns)
    log(f"[Retrieve] Keyword search: {len(keyword_docs)} docs", "debug")
    
    # Deduplicate
    keyword_docs = deduplicate_documents(keyword_docs)
    log(f"[Retrieve] After dedup: {len(keyword_docs)} docs", "debug")
    
    # Exact match for Serial
    if patterns["serials"]:
        exact_serial = patterns["serials"][0].upper()
        exact_matches = [
            d for d in keyword_docs 
            if (d.metadata.get('serial') or '').upper() == exact_serial
        ]
        if exact_matches:
            log(f"[Retrieve] Exact serial match: {exact_serial}", "info")
            return exact_matches
    
    # Exact match for Asset
    if patterns["assets"]:
        exact_asset = patterns["assets"][0]
        exact_matches = [
            d for d in keyword_docs 
            if (d.metadata.get('asset_no') or '') == exact_asset
        ]
        if exact_matches:
            log(f"[Retrieve] Exact asset match: {exact_asset}", "info")
            return exact_matches
    
    # Specific Model
    if patterns["specific_model"]:
        return keyword_docs if keyword_docs else []
    
    # Return keyword results if found
    if keyword_docs:
        log(f"[Retrieve] Returning {len(keyword_docs[:MAX_FINAL_RESULTS])} results", "info")
        return keyword_docs[:MAX_FINAL_RESULTS]
    
    # Semantic search fallback
    log(f"[Retrieve] Trying semantic search...", "debug")
    try:
        semantic_docs = get_vectorstore().similarity_search(
            question,
            k=MAX_SEMANTIC_RESULTS,
            filter={"source": "inventory"}
        )
        log(f"[Retrieve] Semantic search: {len(semantic_docs)} docs", "info")
        
        # Filter by specific model if needed
        if patterns["specific_model"]:
            specific_model = patterns["specific_model"].upper()
            semantic_docs = [
                d for d in semantic_docs
                if specific_model in (d.metadata.get('model') or '').upper()
            ]
        
        return semantic_docs
        
    except Exception as e:
        log(f"[ERROR] Semantic search failed: {e}", "error")
        return []

#============================================================================
# TICKET RETRIEVAL
#============================================================================

def retrieve_tickets(question: str) -> List[Document]:
    log(f"[Ticket] Searching for: {question}", "info")
    
    try:
        semantic_docs = get_vectorstore().similarity_search(
            question,
            k=30,
            filter={"source": "support_tickets"}
        )
        
        log(f"[Ticket] Found {len(semantic_docs)} results", "info")
        
        if semantic_docs and LOG_LEVEL == "debug":
            log("\n[Ticket] Top 5 results:", "debug")
            for i, doc in enumerate(semantic_docs[:5], 1):
                subject = doc.metadata.get('subject', 'N/A')
                typ = doc.metadata.get('type', 'N/A')
                tags = doc.metadata.get('tags', 'N/A')
                log(f"  {i}. {subject}", "debug")
                log(f"     Type: {typ}, Tags: {tags}", "debug")
        
        return semantic_docs
        
    except Exception as e:
        log(f"[ERROR] Ticket retrieval failed: {e}", "error")
        return []

#============================================================================
# TICKET SOLUTION EXTRACTION
#============================================================================

def extract_solutions_from_tickets(docs: List[Document]) -> Dict[str, list]:
    solutions = []
    problem_subjects = []
    
    for doc in docs[:15]:
        content = doc.page_content
        subject = doc.metadata.get('subject', '')
        
        # Extract problem subjects
        if MIN_SUBJECT_LENGTH < len(subject) < MAX_SUBJECT_LENGTH:
            problem_subjects.append(subject)
        
        lines = content.split('\n')
        in_solution_section = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Solution section headers
            solution_headers = [
                'solution:', 'fix:', 'resolution:', 'วิธีแก้:', 'แก้ไข:',
                'how to fix:', 'troubleshooting:', 'steps to resolve:',
                'recommended action:', 'การแก้ไข:', 'ขั้นตอน:'
            ]
            
            if any(header in line_lower for header in solution_headers):
                in_solution_section = True
                continue
            
            # End markers
            end_markers = [
                'note:', 'warning:', 'status:', 'priority:', 'tags:',
                'created:', 'updated:', 'assigned:', 'หมายเหตุ:'
            ]
            
            if any(marker in line_lower for marker in end_markers):
                in_solution_section = False
                continue
            
            # Extract solution lines
            if in_solution_section and line.strip():
                noise_patterns = [
                    r'^[<>=/\-_]{3,}',
                    r'^\s*ticket\s*#?\d+',
                    r'^\s*subject:',
                    r'^\s*type:',
                    r'^\s*queue:',
                    r'problem|issue|error|failed|malfunction',
                ]
                
                is_noise = any(re.search(pattern, line_lower) for pattern in noise_patterns)
                
                if not is_noise:
                    clean_line = clean_text_formatting(line.strip())
                    
                    action_verbs = [
                        'restart', 'reset', 'check', 'verify', 'install', 'uninstall',
                        'update', 'configure', 'enable', 'disable', 'connect', 'disconnect',
                        'remove', 'add', 'replace', 'repair', 'contact', 'call',
                        'ตรวจสอบ', 'รีสตาร์ท', 'รีเซ็ต', 'ติดตั้ง', 'ถอนการติดตั้ง',
                        'อัปเดต', 'ตั้งค่า', 'เชื่อมต่อ', 'ลบ', 'เพิ่ม', 'เปลี่ยน', 'ติดต่อ'
                    ]
                    
                    has_action = any(verb in clean_line.lower() for verb in action_verbs)
                    
                    if has_action and MIN_SOLUTION_LENGTH <= len(clean_line) <= MAX_SOLUTION_LENGTH:
                        solutions.append(clean_line)
        
        # Extract numbered steps
        numbered_pattern = r'^\s*(?:\d+[\.\):]|-|\*|•)\s+(.+)'
        for line in lines:
            match = re.match(numbered_pattern, line)
            if match:
                step = match.group(1).strip()
                clean_step = clean_text_formatting(step)
                
                action_verbs = [
                    'restart', 'reset', 'check', 'verify', 'install', 'update',
                    'configure', 'enable', 'disable', 'connect', 'remove', 'replace',
                    'ตรวจสอบ', 'รีสตาร์ท', 'รีเซ็ต', 'ติดตั้ง', 'อัปเดต', 'ตั้งค่า'
                ]
                
                problem_words = ['problem', 'issue', 'error', 'failed', 'malfunction', 'ปัญหา']
                
                has_action = any(verb in clean_step.lower() for verb in action_verbs)
                is_not_problem = not any(word in clean_step.lower() for word in problem_words)
                
                if has_action and is_not_problem and MIN_SOLUTION_LENGTH <= len(clean_step) <= MAX_SOLUTION_LENGTH:
                    solutions.append(clean_step)
    
    # Deduplicate solutions
    unique_solutions = []
    seen = set()
    
    for sol in solutions:
        normalized = re.sub(r'\s+', ' ', sol.lower().strip())
        
        if normalized not in seen and len(normalized) > MIN_SOLUTION_LENGTH:
            seen.add(normalized)
            unique_solutions.append(sol)
    
    result = {
        'problems': list(set(problem_subjects))[:3],
        'solutions': unique_solutions[:10],
    }
    
    log(f"[Extract] {len(result['problems'])} problems, {len(result['solutions'])} solutions", "debug")
    return result

def format_solution_response(question: str, docs: List[Document]) -> str:
    if not docs:
        return "ไม่พบ Support Tickets ที่เกี่ยวข้อง"
    try:
        extracted = extract_solutions_from_tickets(docs)
        
        if not extracted or not isinstance(extracted, dict):
            extracted = {'problems': [], 'solutions': []}
        
    except Exception as e:
        log(f"[ERROR] Failed to extract solutions: {e}", "error")
        extracted = {'problems': [], 'solutions': []}
    
    response_parts = []
    response_parts.append("วิธีแก้ไขปัญหา\n")
    response_parts.append("="*50 + "\n")
    
    # Show common problems
    problems = extracted.get('problems', [])
    if problems:
        response_parts.append("\nปัญหาที่พบบ่อย:\n")
        for i, prob in enumerate(problems[:3], 1):
            clean_prob = clean_text_formatting(prob)
            if len(clean_prob) > MIN_SUBJECT_LENGTH:
                response_parts.append(f"  {i}. {clean_prob}\n")
    
    # Show solutions
    solutions = extracted.get('solutions', [])
    if solutions:
        response_parts.append("\nวิธีแก้ไขเบื้องต้น:\n")
        
        # Group similar solutions
        grouped_solutions = []
        used_indices = set()
        
        for i, sol in enumerate(solutions):
            if i in used_indices:
                continue
                
            similar = [sol]
            for j, other_sol in enumerate(solutions):
                if i != j and j not in used_indices:
                    sol_words = set(sol.lower().split())
                    other_words = set(other_sol.lower().split())
                    if len(sol_words) > 0 and len(other_words) > 0:
                        overlap = len(sol_words & other_words) / max(len(sol_words), len(other_words))
                        
                        if overlap > 0.6:
                            similar.append(other_sol)
                            used_indices.add(j)
            
            best_solution = min(similar, key=len)
            grouped_solutions.append(best_solution)
            used_indices.add(i)
        
        for i, sol in enumerate(grouped_solutions[:8], 1):
            response_parts.append(f"  {i}. {sol}\n")
        
    else:
        response_parts.append("\nคำแนะนำทั่วไป:\n")
        response_parts.append("  - ตรวจสอบการเชื่อมต่อและสายเคเบิล\n")
        response_parts.append("  - ลองรีสตาร์ทอุปกรณ์\n")
        response_parts.append("  - อัปเดต drivers และ software ให้เป็นเวอร์ชันล่าสุด\n")
        response_parts.append("  - ตรวจสอบ error logs หรือข้อความแสดงข้อผิดพลาด\n")
    
    # Show related tickets
    response_parts.append("\nTickets ที่เกี่ยวข้อง:\n")
    shown_tickets = 0
    for doc in docs[:10]:
        subject = clean_text_formatting(doc.metadata.get('subject', ''))
        if subject and len(subject) > MIN_SUBJECT_LENGTH and shown_tickets < 5:
            shown_tickets += 1
            response_parts.append(f"  {shown_tickets}. {subject}\n")
    
    response_parts.append("\nหมายเหตุ: หากวิธีแก้ไขข้างต้นไม่ได้ผล กรุณาติดต่อทีม IT Support\n")
    
    return "".join(response_parts)
#============================================================================
# CONTEXT FORMATTING
#============================================================================

def format_inventory_context(docs: List[Document], max_docs: int = 3) -> str:
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
        
        # Fallback: extract serial from content if missing
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
    return result

def format_ticket_context(docs: List[Document], max_docs: int = 10) -> str:
    if not docs:
        return "ไม่พบ ticket ในระบบ"
    
    docs = docs[:max_docs]
    lines = [f"พบ {len(docs)} tickets ที่เกี่ยวข้อง:\n"]
    
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        
        lines.append(f"\n--- Ticket #{i} ---")
        
        subject = clean_text_formatting(meta.get('subject', 'N/A'))
        lines.append(f"Subject: {subject}")
        
        lines.append(f"Type: {meta.get('type', 'N/A')}")
        lines.append(f"Queue: {meta.get('queue', 'N/A')}")
        lines.append(f"Priority: {meta.get('priority', 'N/A')}")
        lines.append(f"Tags: {meta.get('tags', 'N/A')}")
        
        content_preview = clean_text_formatting(doc.page_content[:500]) if doc.page_content else ""
        if len(doc.page_content) > 500:
            content_preview += "..."
        lines.append(f"Details:\n{content_preview}")
    
    return "\n".join(lines)

#============================================================================
# PROMPT TEMPLATES
#============================================================================

IT_ASSET_PROMPT = ChatPromptTemplate.from_template("""
You are an IT Asset Database assistant.

LANGUAGE RULE:
- Detect the language ONLY from the user's question.
- If the question is in Thai, answer in correct Thai.
- If the question is in English, answer in correct English.
- Do NOT mix languages.
- Use proper grammar and correct spacing for the selected language.

DATA:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. For EACH asset found, you MUST include ALL fields below:
   - Model (from MODEL)
   - Serial Number (from SERIAL_NUMBER)
   - Asset Number (from ASSET_NUMBER)
   - Status (from STATUS)
   - Location (from LOCATION)

2. Use this exact format (do not add decoration or symbols):

   Model: [MODEL]
   Serial Number: [SERIAL_NUMBER]
   Asset Number: [ASSET_NUMBER]
   Status: [STATUS]
   Location: [LOCATION]

3. NEVER omit the Serial Number.
4. If multiple assets exist, list ALL of them.
5. Use clear, simple language with correct spacing.

ANSWER:
""")

SUPPORT_TICKET_PROMPT = ChatPromptTemplate.from_template("""
You are an IT Support Expert.

LANGUAGE RULE:
- Detect the language ONLY from the user's question.
- Respond in the SAME language.
- Use grammatically correct sentences and proper spacing.
- Do NOT mix Thai and English.

RETRIEVED SUPPORT TICKETS:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Focus ONLY on solutions and troubleshooting steps.
2. Ignore problem descriptions unless needed for clarity.
3. Present the answer as clear, ordered steps.
4. Use actionable and professional language.
5. If multiple tickets describe the same issue, merge their solutions into one.

ANSWER:
""")

GENERAL_PROMPT = ChatPromptTemplate.from_template("""
You are a friendly and professional IT Support Assistant.

LANGUAGE RULE:
- Detect the language from the user's question.
- Reply in the SAME language.
- Use correct grammar and natural sentence structure.
- Ensure proper spacing and readability.

USER QUESTION:
{question}

Provide a clear, helpful, and professional response:
""")

#============================================================================
# CHAT HISTORY
#============================================================================

@lru_cache(maxsize=10)
def get_session_history(session_id: str):
    return PostgresChatMessageHistory(
        connection_string=PSYCOPG_CONN_INFO,
        session_id=session_id
    )

#============================================================================
# MAIN CHAT FUNCTION
#============================================================================

def chat_with_warehouse_system(
    session_id: str,
    question: str,
    image: bytes | None = None
) -> Generator[str, None, None]:
    
    log(f"[Chat] Session: {session_id} | Query: {question}", "info")
    
    try:
        llm = get_llm()
        history = get_session_history(session_id)
        
        # Classify intent
        intent = classify_intent(question)
        force_inventory = is_code_like(question)

        log(f"[Chat] Intent: {intent}", "info")

        if force_inventory:
            log("[Chat] Forcing inventory mode (code detected)", "debug")
            intent = "inventory"        
        
        # Retrieve documents
        docs = hybrid_retrieve(question)

        #--------------------------------------------------------------------
        # INVENTORY QUERY HANDLING
        #--------------------------------------------------------------------
        if intent == "inventory":
            log("[Chat] Processing inventory query", "debug")
            
            if not docs:
                yield "ไม่พบข้อมูลในระบบ\n\n"
                yield "ลองตรวจสอบ:\n"
                yield "- Serial Number ถูกต้องหรือไม่\n"
                yield "- ค้นหาด้วย Model หรือ Asset Number แทน\n"
                return
            
            total_docs = len(docs)
            
            if total_docs == 1:
                header = "พบข้อมูล 1 รายการ:\n\n"
            else:
                header = f"พบข้อมูล {total_docs} รายการ:\n\n"
            
            yield header
            
            full_response_parts = [header]
            
            # Display each item
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                
                item_header = f"{'='*3}\nรายการที่ {i}:\n{'='*3}\n"
                yield item_header
                full_response_parts.append(item_header)
                
                model = meta.get('model', 'N/A')
                model_line = f"- Model: {model}\n"
                yield model_line
                full_response_parts.append(model_line)
                
                model_no = meta.get('model_no', '')
                if model_no and model_no.strip() and model_no != 'N/A':
                    model_no_line = f"- Model No: {model_no}\n"
                    yield model_no_line
                    full_response_parts.append(model_no_line)
                
                serial = meta.get('serial', '')
                if serial and serial.strip() and serial != 'N/A':
                    serial_line = f"- Serial Number: {serial}\n"
                    yield serial_line
                    full_response_parts.append(serial_line)
                else:
                    no_serial_line = f"- Serial Number: ไม่มีข้อมูล\n"
                    yield no_serial_line
                    full_response_parts.append(no_serial_line)
                
                asset_no = meta.get('asset_no', '')
                if asset_no and asset_no.strip() and asset_no != 'N/A':
                    asset_line = f"- Asset Number: {asset_no}\n"
                    yield asset_line
                    full_response_parts.append(asset_line)
                
                status = meta.get('status', '')
                if status and status.strip() and status != 'N/A':
                    status_line = f"- Status: {status}\n"
                    yield status_line
                    full_response_parts.append(status_line)
                
                location = meta.get('location', '')
                if location and location.strip() and location != 'N/A':
                    location_line = f"- Location: {location}\n"
                    yield location_line
                    full_response_parts.append(location_line)
                
                separator = "\n\n"
                yield separator
                full_response_parts.append(separator)
            
            # Save to history
            full_response = "".join(full_response_parts)
            history.add_user_message(question)
            history.add_ai_message(full_response)
            return
        
        #--------------------------------------------------------------------
        # TICKET QUERY HANDLING
        #--------------------------------------------------------------------
        elif intent == "ticket":
            log("[Chat] Processing ticket query", "debug")
            
            ticket_docs = retrieve_tickets(question)
            
            if not ticket_docs:
                yield "ไม่พบ Support Tickets ที่เกี่ยวข้อง\n\n"
                yield "ลองถามคำถามแบบนี้:\n"
                yield "- มีปัญหาเกี่ยวกับ VPN บ้างไหม\n"
                yield "- ticket เกี่ยวกับ network มีอะไรบ้าง\n"
                yield "- แก้ปัญหา login ยังไง\n"
                return

            log(f"[Chat] Response mode: {TICKET_RESPONSE_MODE}", "debug")
            
            if TICKET_RESPONSE_MODE == "rule":
                log("[Chat] Using rule-based extraction", "debug")
                response = format_solution_response(question, ticket_docs)
                
                for char in response:
                    yield char
                
                history.add_user_message(question)
                history.add_ai_message(response)
                
            else:
                log("[Chat] Using LLM-based response", "debug")
                context = format_ticket_context(ticket_docs, max_docs=10)
                
                chain = (
                    {
                        "context": lambda _: context,
                        "question": RunnablePassthrough()
                    }
                    | SUPPORT_TICKET_PROMPT
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
        
        #--------------------------------------------------------------------
        # GENERAL QUERY HANDLING
        #--------------------------------------------------------------------
        else:
            log("[Chat] Processing general query", "debug")
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
        log(f"[ERROR] Chat failed: {e}", "error")
        import traceback
        traceback.print_exc()
        yield f"เกิดข้อผิดพลาด: {str(e)}\n"

#============================================================================
# CLEANUP FUNCTIONS
#============================================================================

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