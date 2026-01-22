# 8
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

TICKET_RESPONSE_MODE = os.getenv("TICKET_RESPONSE_MODE", "rule")

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
# GLOBAL VARIABLES
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
# INTENT CLASSIFICATION
# ============================================================================

def classify_intent(question: str) -> str:
    """แยกประเภทคำถาม: inventory, ticket, หรือ general"""
    q_lower = question.lower()
    
    print(f"\n[INTENT DEBUG] Analyzing: {question}")
    
    # Strong ticket indicators
    strong_ticket_indicators = [
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
    
    for pattern in strong_ticket_indicators:
        if re.search(pattern, q_lower):
            print(f"[INTENT DEBUG] Matched strong ticket indicator: {pattern}")
            return "ticket"
    
    # Serial Number patterns
    serial_pattern = r'\b[A-Z0-9]{8,20}\b'
    serial_matches = re.findall(serial_pattern, question.upper())
    common_words = {'PRINTER', 'MACBOOK', 'THINKPAD', 'LAPTOP', 'SWITCH', 'ROUTER'}
    real_serials = [s for s in serial_matches if s not in common_words]
    
    if real_serials:
        print(f"[INTENT DEBUG] Found real serial numbers: {real_serials}")
        return "inventory"
    
    # Asset Number
    if re.search(r'\b\d{7,10}\b', question):
        print(f"[INTENT DEBUG] Found asset number pattern")
        return "inventory"
    
    # Model No patterns
    if re.search(r'\b[A-Z]{2,3}-[A-Z0-9]{3,}\b', question.upper()):
        print(f"[INTENT DEBUG] Found model number pattern")
        return "inventory"
    
    # Keyword scoring
    inventory_keywords = [
        "serial", "s/n", "sn", "asset", "รุ่น", "เครื่อง", "อุปกรณ์",
        "มี", "เหลือ", "กี่", "จำนวน", "spare", "obsolete",
        "ตำแหน่ง", "อยู่ที่", "location",
        "model no", "asset no", "serial number",
        "ค้นหา", "หา", "ดู", "เช็ค", "ตรวจสอบ",
    ]
    
    ticket_keywords = [
        "ช่วย", "assist", "support", "help",
        "แจ้ง", "report", "แจ้งเตือน",
        "ปัญหา", "problem", "issue", "bug", "error",
        "ด่วน", "urgent", "priority", "สำคัญ",
        "vpn", "network", "printer", "printing",
        "software", "application", "system",
        "account", "access", "permission",
        "แก้", "fix", "solve", "troubleshoot",
        "ติดตั้ง", "install", "configure", "setup",
        "รีเซ็ต", "reset", "restart", "reboot",
    ]
    
    inventory_score = sum(1 for k in inventory_keywords if k in q_lower)
    ticket_score = sum(1 for k in ticket_keywords if k in q_lower)
    
    print(f"[INTENT DEBUG] Scores - Inventory: {inventory_score}, Ticket: {ticket_score}")
    
    if ticket_score > inventory_score and ticket_score >= 1:
        print(f"[INTENT DEBUG] Classified as: ticket (score-based)")
        return "ticket"
    
    if inventory_score > ticket_score and inventory_score >= 2:
        print(f"[INTENT DEBUG] Classified as: inventory (score-based)")
        return "inventory"
    
    if ticket_score > 0:
        print(f"[INTENT DEBUG] Classified as: ticket (default with keywords)")
        return "ticket"
    
    print(f"[INTENT DEBUG] Classified as: general")
    return "general"

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
    
    device_words = {
        'PRINTER', 'MACBOOK', 'THINKPAD', 'LAPTOP', 'SWITCH', 
        'ROUTER', 'DESKTOP', 'MONITOR', 'KEYBOARD', 'SCANNER'
    }
    
    # Serial patterns
    potential_serials = re.findall(r'\b[A-Z0-9]{8,20}\b', question.upper())
    serials = [s for s in potential_serials if s not in device_words]
    patterns["serials"].extend(serials)
    
    # Asset Number
    assets = re.findall(r'\b\d{7,10}\b', question)
    patterns["assets"].extend(assets)
    
    # Model No
    model_nos = re.findall(r'\b[A-Z]{2,3}-[A-Z0-9-]+\b', question.upper())
    patterns["model_nos"].extend(model_nos)
    
    # Specific Model
    q_lower = question.lower()
    has_inventory_context = any(k in q_lower for k in ['serial', 'asset', 'จำนวน', 'มี', 'เหลือ'])
    
    if has_inventory_context:
        specific_model_patterns = [
            r'\b(FR-\d+)\b',
            r'\b(2930F)\b',
            r'\b(2930M)\b',
            r'\b(JL\d+[A-Z])\b',
        ]
        
        for pattern in specific_model_patterns:
            match = re.search(pattern, question.upper())
            if match:
                patterns["specific_model"] = match.group(1).strip()
                print(f"[PATTERN MATCH] Detected specific model: {patterns['specific_model']}")
                break
    
    # Model keywords
    if has_inventory_context:
        model_keywords = [
            "fr-4080", "2930f", "2930m",
            "thinkpad", "thinkcentre", "thinkstation", 
            "switch", "router", "beacon",
            "gateway", "access point", "ups",
            "elitebook", "optiplex", "prodesk",
        ]
        
        for mk in model_keywords:
            if mk in q_lower:
                patterns["models"].append(mk)
    
    # Location keywords
    location_keywords = [
        "sriracha", "ศรีราชา", 
        "chonburi", "ชลบุรี",
        "custom", "customs",
        "server room", "building",
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
    """ค้นหาด้วย SQL โดยตรงจาก metadata"""
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
            print(f"[SQL RESULT] Found {len(rows)} matches")
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
            print(f"[SQL RESULT] Found {len(rows)} matches")
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
            print(f"[SQL RESULT] Found {len(rows)} matches")
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
            print(f"[SQL RESULT] Found {len(rows)} matches")
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
            print(f"[SQL RESULT] Found {len(rows)} matches")
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
    
    print("\n[STEP 1] Keyword Search")
    keyword_docs = keyword_search_direct(patterns)
    print(f"[STEP 1 RESULT] Found {len(keyword_docs)} docs")
    
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
    
    keyword_docs = unique_keyword_docs
    print(f"[STEP 1 RESULT] After dedup: {len(keyword_docs)} unique docs")
    
    # Exact match for Serial
    if patterns["serials"]:
        exact_serial = patterns["serials"][0].upper()
        exact_matches = [d for d in keyword_docs if (d.metadata.get('serial') or '').upper() == exact_serial]
        if exact_matches:
            print(f"\n[EXACT MATCH] Serial: {exact_serial}")
            return exact_matches
    
    # Exact match for Asset
    if patterns["assets"]:
        exact_asset = patterns["assets"][0]
        exact_matches = [d for d in keyword_docs if (d.metadata.get('asset_no') or '') == exact_asset]
        if exact_matches:
            print(f"\n[EXACT MATCH] Asset: {exact_asset}")
            return exact_matches
    
    # Specific Model
    if patterns["specific_model"]:
        if keyword_docs:
            return keyword_docs
        else:
            return []
    
    # Use keyword results
    if keyword_docs:
        return keyword_docs[:10]
    
    # Semantic Search fallback
    print(f"\n[STEP 4] Semantic Search (fallback)")
    try:
        semantic_docs = get_vectorstore().as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,
                "filter": {"source": "inventory"}
            }
        ).invoke(question)
        print(f"[STEP 4 RESULT] Found {len(semantic_docs)} docs")
        
        if patterns["specific_model"]:
            specific_model = patterns["specific_model"].upper()
            semantic_docs = [
                d for d in semantic_docs
                if specific_model in (d.metadata.get('model') or '').upper()
            ]
        
    except Exception as e:
        print(f"[ERROR] Semantic search failed: {e}")
        semantic_docs = []
    
    # Combine results
    combined = keyword_docs + semantic_docs
    
    # Remove duplicates
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
    
    return unique_docs[:10]

# ============================================================================
# TICKET SOLUTION EXTRACTION
# ============================================================================

def clean_text_formatting(text: str) -> str:
    """ทำความสะอาด text"""
    if not text:
        return ""
    
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

def extract_solutions_from_tickets(docs: List[Document]) -> dict:
    """Extract solutions จาก ticket documents"""
    solutions = []
    problem_subjects = []
    
    for doc in docs[:15]:
        content = doc.page_content
        subject = doc.metadata.get('subject', '')
        
        if 10 < len(subject) < 100:
            problem_subjects.append(subject)
        
        lines = content.split('\n')
        in_solution_section = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            solution_headers = [
                'solution:', 'fix:', 'resolution:', 'วิธีแก้:', 'แก้ไข:',
                'how to fix:', 'troubleshooting:', 'steps to resolve:',
                'recommended action:', 'การแก้ไข:', 'ขั้นตอน:'
            ]
            
            if any(header in line_lower for header in solution_headers):
                in_solution_section = True
                continue
            
            end_markers = [
                'note:', 'warning:', 'status:', 'priority:', 'tags:',
                'created:', 'updated:', 'assigned:', 'หมายเหตุ:'
            ]
            
            if any(marker in line_lower for marker in end_markers):
                in_solution_section = False
                continue
            
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
                    
                    if has_action and 15 <= len(clean_line) <= 150:
                        solutions.append(clean_line)
        
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
                
                if has_action and is_not_problem and 15 <= len(clean_step) <= 150:
                    solutions.append(clean_step)
    
    unique_solutions = []
    seen = set()
    
    for sol in solutions:
        normalized = re.sub(r'\s+', ' ', sol.lower().strip())
        
        if normalized not in seen and len(normalized) > 15:
            seen.add(normalized)
            unique_solutions.append(sol)
    
    result = {
        'problems': list(set(problem_subjects))[:3],
        'solutions': unique_solutions[:10],
    }
    
    print(f"[EXTRACT] Found {len(result['problems'])} problems, {len(result['solutions'])} solutions")
    return result

def format_solution_response(question: str, docs: List[Document]) -> str:
    """Format response แบบ solution-focused"""
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
    
    problems = extracted.get('problems', [])
    if problems:
        response_parts.append("**ปัญหาที่พบบ่อย:**")
        for i, prob in enumerate(problems[:3], 1):
            clean_prob = clean_text_formatting(prob)
            if len(clean_prob) > 10:
                response_parts.append(f"  {i}. {clean_prob}")
        response_parts.append("")
    
    solutions = extracted.get('solutions', [])
    if solutions:
        response_parts.append("**วิธีแก้ไขเบื้องต้น:**")
        
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
    
    response_parts.append("**Tickets ที่เกี่ยวข้อง:**")
    shown_tickets = 0
    for doc in docs[:10]:
        subject = clean_text_formatting(doc.metadata.get('subject', ''))
        if subject and len(subject) > 10 and shown_tickets < 5:
            shown_tickets += 1
            response_parts.append(f"  {shown_tickets}. {subject}")
    
    response_parts.append("\n💡 **หมายเหตุ:** หากวิธีแก้ไขข้างต้นไม่ได้ผล กรุณาติดต่อทีม IT Support")
    
    return "\n".join(response_parts)

def retrieve_tickets(question: str) -> List[Document]:
    """ค้นหา support tickets"""
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
You are an IT Support Expert providing solutions to technical problems.
Answer in the SAME LANGUAGE as the question (Thai or English).

## Retrieved Support Tickets:
{context}

## User Question:
{question}

## Instructions:
1. Focus on SOLUTIONS, not just problem descriptions
2. Extract and present ONLY the troubleshooting steps and fixes from the tickets
3. Format your answer with clear steps
4. Use actionable language
5. If multiple tickets have similar problems, combine their solutions

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
    """Format inventory documents"""
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
    return result

def format_ticket_context(docs: List[Document], max_docs: int = 10) -> str:
    """Format support ticket documents"""
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
# CHAT HISTORY
# ============================================================================

@lru_cache(maxsize=10)
def get_session_history(session_id: str):
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
    """Main chat function"""
    
    print("\n" + "="*70)
    print(f"[CHAT START] Session: {session_id}")
    print(f"[CHAT START] Question: {question}")
    print("="*70)
    
    try:
        llm = get_llm()
        history = get_session_history(session_id)
        
        intent = classify_intent(question)
        print(f"\n[INTENT] {intent}")
        
        if intent == "inventory":
            print("[CHAT] Processing as INVENTORY query")
            docs = hybrid_retrieve(question)
            
            if not docs:
                yield "🔍 ไม่พบข้อมูลในระบบ\n\n"
                yield "💡 ลองตรวจสอบ:\n"
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