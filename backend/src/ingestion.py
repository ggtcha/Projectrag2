import os
import gc
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import PGVector
# ลบ RecursiveCharacterTextSplitter ออกเพราะข้อมูลตารางไม่ควรโดนหั่น

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# ============================================================================
# Configuration
# ============================================================================
DB_URL = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
COLLECTION = os.getenv("COLLECTION_NAME")
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

INVENTORY_FILE = "data/data_inventory.xlsx"

TARGET_SHEETS = [
    ("Spare", "อะไหล่/อุปกรณ์สำรอง"),
    ("Obsolete", "อุปกรณ์เลิกใช้งาน"),
]

# ============================================================================
# TEXT CLEANING
# ============================================================================
USELESS_VALUES = {"nan", "none", "null", "n/a", "", "-", "ไม่มี", "ไม่มีข้อมูล", "N/A"}

def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in USELESS_VALUES or text == "-":
        return None
    return text

# ============================================================================
# DEVICE CATEGORY DETECTION
# ============================================================================
def detect_device_category(model: str) -> str:
    if not model: return "อุปกรณ์ IT อื่นๆ"
    m = model.lower()
    if any(x in m for x in ["thinkpad", "laptop", "elitebook", "notebook"]): return "Laptop/Notebook"
    if any(x in m for x in ["thinkcentre", "optiplex", "prodesk", "desktop"]): return "Desktop PC"
    if "printer" in m: return "Printer"
    if "switch" in m: return "Network Switch"
    return "อุปกรณ์ IT"

# ============================================================================
# IMPROVED CONTENT BUILDER (หัวใจสำคัญที่ทำให้ค้นหาแม่นยำขึ้น)
# ============================================================================
def build_inventory_content(row: Dict, sheet_label: str) -> str:
    """สร้างเนื้อหาที่เน้น Keyword ให้ครอบคลุมทั้ง Serial และสถานะการใช้งาน"""
    
    model = clean_text(row.get('Model', ''))
    serial = clean_text(row.get('Serial', ''))
    status = clean_text(row.get('Status', ''))
    asset_no = clean_text(row.get('Asset No', ''))
    loc = clean_text(row.get('Locations', ''))
    
    device_cat = detect_device_category(model)
    
    # วิเคราะห์สถานะให้เป็นภาษาไทยที่ AI เข้าใจง่าย
    availability = "ว่าง/พร้อมใช้งาน (In Stock)" if "spare" in status.lower() else "ถูกใช้งานอยู่ (In Use)"
    
    # สร้างก้อนข้อมูลที่เน้น Keyword สำคัญ
    parts = [
        f"ข้อมูลอุปกรณ์: {device_cat} รุ่น {model}",
        f"สถานะปัจจุบัน: {status} - {availability}",
        f"ตำแหน่งที่เก็บ: {loc}",
        f"หมายเลขซีเรียล (Serial Number): {serial}",
        f"หมายเลขทรัพย์สิน (Asset No): {asset_no}",
        f"หมวดหมู่ในระบบ: {sheet_label}",
        f"--- สำหรับค้นหา ---",
        f"keyword: {model} {serial} {asset_no} {loc} {device_cat} สต็อกคอม ว่าง"
    ]
    
    return "\n".join(parts)

# ============================================================================
# DOCUMENT LOADER (ปรับให้ 1 Row = 1 Doc)
# ============================================================================
def load_inventory_documents(file_path: str, sheet_configs: List[tuple]) -> List[Document]:
    all_docs = []
    for sheet_name, label in sheet_configs:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str).dropna(how="all")
            df.columns = [str(c).strip() for c in df.columns]
            
            for idx, row in df.iterrows():
                data = row.to_dict()
                if not clean_text(data.get('Model')): continue
                
                content = build_inventory_content(data, label)
                
                # เก็บ Metadata สำคัญเพื่อใช้ทำ Filtering ในอนาคต
                metadata = {
                    "source": "inventory",
                    "sheet": sheet_name,
                    "model": str(data.get('Model', '')).upper(),
                    "serial": str(data.get('Serial', '')).upper(),
                    "status": str(data.get('Status', '')).lower(),
                    "row": int(idx)
                }
                
                all_docs.append(Document(page_content=content, metadata=metadata))
            del df
        except Exception as e:
            print(f"Error loading {sheet_name}: {e}")
    return all_docs

# ============================================================================
# MAIN INGESTION (ตัดส่วนการ Split ออก)
# ============================================================================
def ingest_real_inventory():
    print("="*70)
    print(" IT SUPPORT KNOWLEDGE BASE - IMPROVED INGESTION")
    print("="*70)
    
    if not os.path.exists(INVENTORY_FILE):
        print(f"ไม่พบไฟล์: {INVENTORY_FILE}")
        return

    all_docs = load_inventory_documents(INVENTORY_FILE, TARGET_SHEETS)
    if not all_docs:
        print("ไม่พบข้อมูลที่ใช้งานได้")
        return

    # หมายเหตุ: เราไม่ใช้ RecursiveCharacterTextSplitter แล้ว 
    # เพื่อป้องกันข้อมูล Serial หรือ Status ของเครื่องเดียวกันถูกแยกออกจากกัน
    chunks = all_docs 
    
    print(f"จำนวนอุปกรณ์ที่พบ: {len(chunks)} เครื่อง (1 เครื่องต่อ 1 ก้อนข้อมูล)")
    
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    
    try:
        PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION,
            connection_string=DB_URL,
            pre_delete_collection=True, # ล้างข้อมูลเก่าเพื่อป้องกันข้อมูลซ้ำซ้อน
        )
        print("\n" + "="*70)
        print(" INGESTION สำเร็จ! ระบบพร้อมรองรับการค้นหา Serial และเช็คสต็อก")
        print("="*70)
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    ingest_real_inventory()