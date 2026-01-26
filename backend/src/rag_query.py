# 8
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

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Database Configuration
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DATABASE = os.getenv("PG_DATABASE")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:0.5b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = "http://localhost:11434"

# Response Mode
TICKET_RESPONSE_MODE = os.getenv("TICKET_RESPONSE_MODE", "rule")

# Connection Strings
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

# Search Configuration
MAX_KEYWORD_RESULTS = 100
MAX_EXACT_MATCHES = 3
MAX_LOCATION_MATCHES = 5
MAX_SEMANTIC_RESULTS = 10
MAX_FINAL_RESULTS = 10

# Text Cleaning Configuration
MIN_SOLUTION_LENGTH = 15
MAX_SOLUTION_LENGTH = 150
MAX_SUBJECT_LENGTH = 100
MIN_SUBJECT_LENGTH = 10

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

_embeddings = None
_vectorstore = None
_retriever = None
_llm = None
_db_conn = None

# ============================================================================
# DATABASE CONNECTION MANAGEMENT
# ============================================================================

def get_db_connection():
    """Get or create database connection with singleton pattern"""
    global _db_conn
    try:
        if _db_conn is None or _db_conn.closed:
            print("[INIT] Connecting to database...")
            _db_conn = psycopg2.connect(PSYCOPG_CONN_INFO)
        return _db_conn
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        raise

@contextmanager
def get_db_cursor():
    """Context manager for database cursor - prevents connection leaks"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Database operation failed: {e}")
        raise e
    finally:
        cursor.close()

# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def get_embeddings():
    """Initialize embeddings model"""
    global _embeddings
    if _embeddings is None:
        print("[INIT] Creating embeddings...")
        _embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL
        )
    return _embeddings

def get_vectorstore():
    """Initialize vector store"""
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
    """Initialize retriever"""
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
    """Initialize LLM"""
    global _llm
    if _llm is None:
        print("[INIT] Connecting to LLM...")
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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_code_like(text: str) -> bool:
    """
    Detect if text is a model/serial/asset code
    Examples: H2-PNCN, ThinkPad13, JL253A, 2930F
    """
    if not text:
        return False

    compact = re.sub(r'\s+', '', text.upper())

    return bool(
        re.fullmatch(r'[A-Z0-9\-]{4,}', compact) or
        re.search(r'(THINKPAD|ELITEBOOK|OPTIPLEX)\d*', compact)
    )

def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """Remove duplicate documents based on unique identifiers"""
    seen = set()
    unique_docs = []
    
    for doc in docs:
        serial = (doc.metadata.get('serial') or '').strip().upper()
        asset = (doc.metadata.get('asset_no') or '').strip()
        
        # Create unique key
        if serial:
            unique_key = f"serial_{serial}"
        elif asset:
            unique_key = f"asset_{asset}"
        else:
            # Fallback to content-based key
            unique_key = f"row_{doc.metadata.get('row', '')}_{doc.metadata.get('model', '')}"
        
        if unique_key not in seen:
            seen.add(unique_key)
            unique_docs.append(doc)
    
    return unique_docs

def clean_text_formatting(text: str) -> str:
    """Clean and normalize text formatting"""
    if not text:
        return ""
    
    import html
    
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove markdown formatting
    text = re.sub(r'_{2,}', ' ', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    
    # Filter allowed characters
    allowed_chars = []
    for char in text:
        code_point = ord(char)
        
        if (0x0E00 <= code_point <= 0x0E7F or  # Thai
            0x0020 <= code_point <= 0x007E or  # ASCII
            code_point in [0x000A, 0x000D] or  # Newlines
            0x2000 <= code_point <= 0x206F):   # Punctuation
            allowed_chars.append(char)
        else:
            allowed_chars.append(' ')
    
    text = ''.join(allowed_chars)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================

def classify_intent(question: str) -> str:
    """
    Classify question intent into: inventory, ticket, or general
    
    Args:
        question: User's question
        
    Returns:
        Intent type: 'inventory', 'ticket', or 'general'
    """
    q_lower = question.lower()
    
    print(f"\n[INTENT DEBUG] Analyzing: {question}")
    
    # Rule 1: Hard inventory indicators
    if re.search(r'(serial|serial number|sn|asset|asset number|ขอserial|ขอ serial)', q_lower):
        print("[INTENT DEBUG] Hard rule: inventory (serial request)")
        return "inventory"
    
    # Rule 2: Check if code-like (model/serial/asset)
    if is_code_like(question):
        print("[INTENT DEBUG] Code-like pattern detected: inventory")
        return "inventory"
    
    # Rule 3: Strong ticket indicators
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
            print(f"[INTENT DEBUG] Matched ticket indicator: {pattern}")
            return "ticket"
    
    # Rule 4: Pattern-based scoring
    patterns = extract_search_patterns(question)
    
    inventory_score = (
        len(patterns["serials"]) * 3 +
        len(patterns["assets"]) * 3 +
        len(patterns["model_nos"]) * 2 +
        len(patterns["models"]) * 2 +
        len(patterns["locations"]) * 1
    )
    
    # Rule 5: Keyword scoring
    ticket_keywords = [
        "ช่วย", "assist", "support", "help", "แจ้ง", "report",
        "ปัญหา", "problem", "issue", "แก้", "fix", "solve",
        "ติดตั้ง", "install", "configure", "setup"
    ]
    ticket_score = sum(1 for k in ticket_keywords if k in q_lower)
    
    print(f"[INTENT DEBUG] Scores - Inventory: {inventory_score}, Ticket: {ticket_score}")
    
    # Decision logic
    if inventory_score >= 3:
        print(f"[INTENT DEBUG] Classified as: inventory (strong patterns)")
        return "inventory"
    
    if ticket_score >= 2:
        print(f"[INTENT DEBUG] Classified as: ticket (strong keywords)")
        return "ticket"
    
    if inventory_score > ticket_score:
        return "inventory"
    elif ticket_score > inventory_score:
        return "ticket"
    
    print(f"[INTENT DEBUG] Classified as: general")
    return "general"

# ============================================================================
# KEYWORD EXTRACTION
# ============================================================================

def extract_search_patterns(question: str) -> Dict[str, list]:
    """
    Extract search patterns from question
    
    Returns:
        Dictionary containing serials, assets, models, locations, etc.
    """
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
    
    # Extract Serial Numbers
    potential_serials = re.findall(r'\b[A-Z0-9]{8,20}\b', question.upper())
    serials = [s for s in potential_serials if s not in device_words]
    patterns["serials"].extend(serials)
    
    # Extract Asset Numbers
    assets = re.findall(r'\b\d{7,10}\b', question)
    patterns["assets"].extend(assets)
    
    # Extract Model Numbers (including H2-PNCN pattern)
    model_nos = re.findall(r'\b[A-Z0-9]{2,3}-[A-Z0-9-]+\b', question.upper())
    patterns["model_nos"].extend(model_nos)
    
    # Detect Specific Model (IMPROVED - detect H2-PNCN pattern)
    q_lower = question.lower()
    q_upper = question.upper()
    
    # Check for model-like patterns in the question
    model_pattern_checks = [
        r'\b([A-Z0-9]{2,3}-[A-Z0-9]+)\b',  # H2-PNCN, FR-4080, etc.
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
            print(f"[PATTERN MATCH] Detected specific model: {detected}")
            break
    
    # Alternative: if question is just a code-like string, use it as model
    if not patterns["specific_model"] and is_code_like(question):
        clean_q = re.sub(r'\s+', '', question.upper())
        patterns["specific_model"] = clean_q
        patterns["models"].append(clean_q.lower())
        print(f"[PATTERN MATCH] Using entire question as model: {clean_q}")
    
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
    
    print(f"[SEARCH PATTERNS] {patterns}")
    return patterns

# ============================================================================
# KEYWORD SEARCH WITH SQL
# ============================================================================

def keyword_search_direct(patterns: Dict[str, list]) -> List[Document]:
    """
    Direct SQL search using metadata - IMPROVED with context manager
    """
    all_docs = []
    
    try:
        with get_db_cursor() as cursor:
            base_filter = "AND cmetadata->>'source' = 'inventory'"
            
            # Search by Specific Model
            if patterns.get("specific_model"):
                specific_model = patterns["specific_model"]
                print(f"[SQL SEARCH] Specific Model: {specific_model}")
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
                print(f"[SQL RESULT] Found {len(rows)} matches")
                all_docs.extend([
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ])
            
            # Search by Model (general model search)
            for model in patterns["models"]:
                print(f"[SQL SEARCH] Model: {model}")
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
                print(f"[SQL RESULT] Found {len(rows)} matches")
                all_docs.extend([
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ])
            
            # Search by Serial Number
            for serial in patterns["serials"]:
                print(f"[SQL SEARCH] Serial: {serial}")
                query = f"""
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE UPPER(cmetadata->>'serial') = UPPER(%s)
                {base_filter}
                LIMIT {MAX_EXACT_MATCHES}
                """
                cursor.execute(query, (serial,))
                rows = cursor.fetchall()
                print(f"[SQL RESULT] Found {len(rows)} matches")
                all_docs.extend([
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ])
            
            # Search by Asset Number
            for asset in patterns["assets"]:
                print(f"[SQL SEARCH] Asset: {asset}")
                query = f"""
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE cmetadata->>'asset_no' = %s
                {base_filter}
                LIMIT {MAX_EXACT_MATCHES}
                """
                cursor.execute(query, (asset,))
                rows = cursor.fetchall()
                print(f"[SQL RESULT] Found {len(rows)} matches")
                all_docs.extend([
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ])
            
            # Search by Model No
            for model_no in patterns["model_nos"]:
                print(f"[SQL SEARCH] Model No: {model_no}")
                query = f"""
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE UPPER(cmetadata->>'model_no') LIKE UPPER(%s)
                {base_filter}
                LIMIT {MAX_EXACT_MATCHES}
                """
                cursor.execute(query, (f'%{model_no}%',))
                rows = cursor.fetchall()
                print(f"[SQL RESULT] Found {len(rows)} matches")
                all_docs.extend([
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ])
            
            # Search by Location
            for loc in patterns["locations"]:
                print(f"[SQL SEARCH] Location: {loc}")
                query = f"""
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE LOWER(cmetadata->>'location') LIKE LOWER(%s)
                {base_filter}
                LIMIT {MAX_LOCATION_MATCHES}
                """
                cursor.execute(query, (f'%{loc}%',))
                rows = cursor.fetchall()
                print(f"[SQL RESULT] Found {len(rows)} matches")
                all_docs.extend([
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ])
                
    except Exception as e:
        print(f"[ERROR] Keyword search failed: {e}")
    
    return all_docs

# ============================================================================
# HYBRID RETRIEVAL SYSTEM
# ============================================================================

def hybrid_retrieve(question: str) -> List[Document]:
    """
    Hybrid retrieval combining keyword search and semantic search
    """
    print(f"\n{'='*70}")
    print("[RETRIEVE START]")
    print(f"Question: {question}")
    print(f"{'='*70}")
    
    # HARD GUARD: Exact code match for model/serial/asset
    if is_code_like(question):
        compact = re.sub(r'\s+', '', question.upper())
        print(f"[HARD RETRIEVE GUARD] Code-like query: {compact}")

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
                    print(f"[HARD RETRIEVE GUARD] Found {len(rows)} exact matches")
                    return [
                        Document(page_content=doc, metadata=meta or {})
                        for doc, meta in rows
                    ]
                
                # If no exact match, try LIKE search for model field
                print(f"[HARD RETRIEVE GUARD] No exact match, trying LIKE search")
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
                    print("[HARD RETRIEVE GUARD] No match found")
                    return []
                
                print(f"[HARD RETRIEVE GUARD] Found {len(rows)} matches via LIKE")
                return [
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in rows
                ]
        except Exception as e:
            print(f"[ERROR] Hard guard query failed: {e}")
            return []
    
    # STEP 1: Extract patterns
    patterns = extract_search_patterns(question)
    
    # STEP 2: Keyword search
    print("\n[STEP 2] Keyword Search")
    keyword_docs = keyword_search_direct(patterns)
    print(f"[STEP 2 RESULT] Found {len(keyword_docs)} docs")
    
    # STEP 3: Deduplicate
    keyword_docs = deduplicate_documents(keyword_docs)
    print(f"[STEP 3 RESULT] After dedup: {len(keyword_docs)} unique docs")
    
    # STEP 4: Exact match for Serial
    if patterns["serials"]:
        exact_serial = patterns["serials"][0].upper()
        exact_matches = [
            d for d in keyword_docs 
            if (d.metadata.get('serial') or '').upper() == exact_serial
        ]
        if exact_matches:
            print(f"\n[EXACT MATCH] Serial: {exact_serial}")
            return exact_matches
    
    # STEP 5: Exact match for Asset
    if patterns["assets"]:
        exact_asset = patterns["assets"][0]
        exact_matches = [
            d for d in keyword_docs 
            if (d.metadata.get('asset_no') or '') == exact_asset
        ]
        if exact_matches:
            print(f"\n[EXACT MATCH] Asset: {exact_asset}")
            return exact_matches
    
    # STEP 6: Specific Model
    if patterns["specific_model"]:
        return keyword_docs if keyword_docs else []
    
    # STEP 7: Return keyword results if found
    if keyword_docs:
        return keyword_docs[:MAX_FINAL_RESULTS]
    
    # STEP 8: Semantic search fallback
    print(f"\n[STEP 8] Semantic Search (fallback)")
    try:
        semantic_docs = get_vectorstore().similarity_search(
            question,
            k=MAX_SEMANTIC_RESULTS,
            filter={"source": "inventory"}
        )
        print(f"[STEP 8 RESULT] Found {len(semantic_docs)} docs")
        
        # Filter by specific model if needed
        if patterns["specific_model"]:
            specific_model = patterns["specific_model"].upper()
            semantic_docs = [
                d for d in semantic_docs
                if specific_model in (d.metadata.get('model') or '').upper()
            ]
        
        return semantic_docs
        
    except Exception as e:
        print(f"[ERROR] Semantic search failed: {e}")
        return []

# ============================================================================
# TICKET RETRIEVAL
# ============================================================================

def retrieve_tickets(question: str) -> List[Document]:
    """
    Search for support tickets using semantic search
    
    Args:
        question: User's question
        
    Returns:
        List of relevant ticket documents
    """
    print(f"\n{'='*70}")
    print(f"[TICKET RETRIEVE] Starting ticket search")
    print(f"Question: {question}")
    print(f"{'='*70}")
    
    try:
        semantic_docs = get_vectorstore().similarity_search(
            question,
            k=30,
            filter={"source": "support_tickets"}
        )
        
        print(f"[TICKET RETRIEVE] Found {len(semantic_docs)} tickets")
        
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
        return []

# ============================================================================
# TICKET SOLUTION EXTRACTION
# ============================================================================

def extract_solutions_from_tickets(docs: List[Document]) -> Dict[str, list]:
    """
    Extract solutions from ticket documents
    
    Args:
        docs: List of ticket documents
        
    Returns:
        Dictionary containing problems and solutions
    """
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
        numbered_pattern = r'^\s*(?:\d+[\.\):]|-|\*|•)\s+(.+)$'
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
    
    print(f"[EXTRACT] Found {len(result['problems'])} problems, {len(result['solutions'])} solutions")
    return result

def format_solution_response(question: str, docs: List[Document]) -> str:
    """
    Format response with solution-focused approach
    
    Args:
        question: User's question
        docs: List of ticket documents
        
    Returns:
        Formatted solution response
    """
    if not docs:
        return "🔍 ไม่พบ Support Tickets ที่เกี่ยวข้อง"
    
    try:
        extracted = extract_solutions_from_tickets(docs)
        
        if not extracted or not isinstance(extracted, dict):
            extracted = {'problems': [], 'solutions': []}
        
    except Exception as e:
        print(f"[ERROR] Failed to extract solutions: {e}")
        extracted = {'problems': [], 'solutions': []}
    
    response_parts = []
    response_parts.append("## 🔧 วิธีแก้ไขปัญหา\n")
    
    # Show common problems
    problems = extracted.get('problems', [])
    if problems:
        response_parts.append("**ปัญหาที่พบบ่อย:**")
        for i, prob in enumerate(problems[:3], 1):
            clean_prob = clean_text_formatting(prob)
            if len(clean_prob) > MIN_SUBJECT_LENGTH:
                response_parts.append(f"  {i}. {clean_prob}")
        response_parts.append("")
    
    # Show solutions
    solutions = extracted.get('solutions', [])
    if solutions:
        response_parts.append("**วิธีแก้ไขเบื้องต้น:**")
        
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
            response_parts.append(f"  {i}. {sol}")
        
        response_parts.append("")
        
    else:
        response_parts.append("**คำแนะนำทั่วไป:**")
        response_parts.append("  • ตรวจสอบการเชื่อมต่อและสายเคเบิล")
        response_parts.append("  • ลองรีสตาร์ทอุปกรณ์")
        response_parts.append("  • อัปเดต drivers และ software ให้เป็นเวอร์ชันล่าสุด")
        response_parts.append("  • ตรวจสอบ error logs หรือข้อความแสดงข้อผิดพลาด")
        response_parts.append("")
    
    # Show related tickets
    response_parts.append("**Tickets ที่เกี่ยวข้อง:**")
    shown_tickets = 0
    for doc in docs[:10]:
        subject = clean_text_formatting(doc.metadata.get('subject', ''))
        if subject and len(subject) > MIN_SUBJECT_LENGTH and shown_tickets < 5:
            shown_tickets += 1
            response_parts.append(f"  {shown_tickets}. {subject}")
    
    response_parts.append("\n💡 **หมายเหตุ:** หากวิธีแก้ไขข้างต้นไม่ได้ผล กรุณาติดต่อทีม IT Support")
    
    return "\n".join(response_parts)

# ============================================================================
# CONTEXT FORMATTING
# ============================================================================

def format_inventory_context(docs: List[Document], max_docs: int = 3) -> str:
    """
    Format inventory documents for LLM context
    
    Args:
        docs: List of inventory documents
        max_docs: Maximum number of documents to include
        
    Returns:
        Formatted context string
    """
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
    """
    Format support ticket documents for LLM context
    
    Args:
        docs: List of ticket documents
        max_docs: Maximum number of documents to include
        
    Returns:
        Formatted context string
    """
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

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

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

# ============================================================================
# CHAT HISTORY
# ============================================================================

@lru_cache(maxsize=10)
def get_session_history(session_id: str):
    """Get or create chat history for session"""
    return PostgresChatMessageHistory(
        connection_string=PSYCOPG_CONN_INFO,
        session_id=session_id
    )

# ============================================================================
# MAIN CHAT FUNCTION
# ============================================================================

def chat_with_warehouse_system(
    session_id: str,
    question: str,
    image: bytes | None = None
) -> Generator[str, None, None]:
    """
    Main chat function with streaming response
    
    Args:
        session_id: Unique session identifier
        question: User's question
        image: Optional image input (not implemented)
        
    Yields:
        Response chunks
    """
    
    print("\n" + "="*70)
    print(f"[CHAT START] Session: {session_id}")
    print(f"[CHAT START] Question: {question}")
    print("="*70)
    
    try:
        llm = get_llm()
        history = get_session_history(session_id)
        
        # Classify intent
        intent = classify_intent(question)
        force_inventory = is_code_like(question)

        print(f"\n[INTENT] {intent}")

        if force_inventory:
            print("[GUARD] Force inventory (model/serial detected)")
            intent = "inventory"        
        
        # Retrieve documents
        docs = hybrid_retrieve(question)

        if intent == "inventory":
            print("[CHAT] Processing as INVENTORY query")
            
            if not docs:
                yield "ไม่พบข้อมูลในระบบ\n\n"
                yield "ลองตรวจสอบ:\n"
                yield "• Serial Number ถูกต้องหรือไม่\n"
                yield "• ค้นหาด้วย Model หรือ Asset Number แทน\n"
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
                model_line = f"• Model: {model}\n"
                yield model_line
                full_response_parts.append(model_line)
                
                model_no = meta.get('model_no', '')
                if model_no and model_no.strip() and model_no != 'N/A':
                    model_no_line = f"• Model No: {model_no}\n"
                    yield model_no_line
                    full_response_parts.append(model_no_line)
                
                serial = meta.get('serial', '')
                if serial and serial.strip() and serial != 'N/A':
                    serial_line = f"• Serial Number: {serial}\n"
                    yield serial_line
                    full_response_parts.append(serial_line)
                else:
                    no_serial_line = f"• Serial Number: ไม่มีข้อมูล\n"
                    yield no_serial_line
                    full_response_parts.append(no_serial_line)
                
                asset_no = meta.get('asset_no', '')
                if asset_no and asset_no.strip() and asset_no != 'N/A':
                    asset_line = f"• Asset Number: {asset_no}\n"
                    yield asset_line
                    full_response_parts.append(asset_line)
                
                status = meta.get('status', '')
                if status and status.strip() and status != 'N/A':
                    status_line = f"• Status: {status}\n"
                    yield status_line
                    full_response_parts.append(status_line)
                
                location = meta.get('location', '')
                if location and location.strip() and location != 'N/A':
                    location_line = f"• Location: {location}\n"
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
        
        elif intent == "ticket":
            print("[CHAT] Processing as TICKET query")
            
            ticket_docs = retrieve_tickets(question)
            
            if not ticket_docs:
                yield "🔍 ไม่พบ Support Tickets ที่เกี่ยวข้อง\n\n"
                yield "💡 ลองถามคำถามแบบนี้:\n"
                yield "• 'มีปัญหาเกี่ยวกับ VPN บ้างไหม'\n"
                yield "• 'ticket เกี่ยวกับ network มีอะไรบ้าง'\n"
                yield "• 'แก้ปัญหา login ยังไง'\n"
                return

            print(f"[CHAT] Ticket response mode: {TICKET_RESPONSE_MODE}")
            
            if TICKET_RESPONSE_MODE == "rule":
                print("[CHAT] Using RULE-based solution extraction")
                response = format_solution_response(question, ticket_docs)
                
                for char in response:
                    yield char
                
                history.add_user_message(question)
                history.add_ai_message(response)
                
            else:
                print("[CHAT] Using LLM-based response")
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
        yield f"เกิดข้อผิดพลาด: {str(e)}\n"

# ============================================================================
# CLEANUP FUNCTIONS
# ============================================================================

def clear_session_history(session_id: str):
    """Clear chat history for a session"""
    history = get_session_history(session_id)
    history.clear()
    get_session_history.cache_clear()

def cleanup_resources():
    """Clean up and close all resources"""
    global _vectorstore, _embeddings, _llm, _retriever, _db_conn
    
    if _db_conn and not _db_conn.closed:
        _db_conn.close()
        print("[CLEANUP] Database connection closed")
    
    _vectorstore = None
    _embeddings = None
    _llm = None
    _retriever = None
    _db_conn = None
    
    get_session_history.cache_clear()
    print("[CLEANUP] Resources cleaned up")