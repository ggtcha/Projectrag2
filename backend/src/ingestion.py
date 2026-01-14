import os
import gc
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import time

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import PGVector
import psycopg2

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configuration
DB_URL = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
COLLECTION = os.getenv("COLLECTION_NAME")
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

INVENTORY_FILE = "data/data_inventory.xlsx"
SUPPORT_TICKETS_FILE = "data/support_tickets.xlsx"

TARGET_SHEETS = [
    ("Spare", "อะไหล่/อุปกรณ์สำรอง"),
    ("Obsolete", "อุปกรณ์เลิกใช้งาน"),
]

# Text Cleaning
USELESS_VALUES = {"nan", "none", "null", "n/a", "", "-", "ไม่มี", "ไม่มีข้อมูล", "N/A"}

def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in USELESS_VALUES or text == "-":
        return None
    return text

# Device Category Detection
def detect_device_category(model: str) -> str:
    if not model: return "อุปกรณ์ IT อื่นๆ"
    m = model.lower()
    if any(x in m for x in ["thinkpad", "laptop", "elitebook", "notebook"]): return "Laptop/Notebook"
    if any(x in m for x in ["thinkcentre", "optiplex", "prodesk", "desktop"]): return "Desktop PC"
    if "printer" in m: return "Printer"
    if "switch" in m: return "Network Switch"
    return "อุปกรณ์ IT"

# Inventory Content Builder
def build_inventory_content(row: Dict, sheet_label: str) -> str:
    model = clean_text(row.get('Model', ''))
    serial = clean_text(row.get('Serial', ''))
    status = clean_text(row.get('Status', ''))
    asset_no = clean_text(row.get('Asset No', ''))
    loc = clean_text(row.get('Locations', ''))
    
    device_cat = detect_device_category(model)
    availability = "ว่าง/พร้อมใช้งาน (In Stock)" if status and "spare" in status.lower() else "ถูกใช้งานอยู่ (In Use)"
    
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

# Support Tickets Content Builder
def build_ticket_content(row: Dict) -> str:
    subject = clean_text(row.get('subject', '')) or ''
    body = clean_text(row.get('body', '')) or ''
    type_ticket = clean_text(row.get('type', '')) or ''
    queue = clean_text(row.get('queue', '')) or ''
    priority = clean_text(row.get('priority', '')) or ''
    lang = clean_text(row.get('language_version', '')) or ''
    
    tag_1 = clean_text(row.get('tag_1', '')) or ''
    tag_2 = clean_text(row.get('tag_2', '')) or ''
    tag_3 = clean_text(row.get('tag_3', '')) or ''
    tag_4 = clean_text(row.get('tag_4', '')) or ''
    
    priority_th = {
        'high': 'สูง',
        'medium': 'กลาง', 
        'low': 'ต่ำ'
    }.get(priority.lower() if priority else '', priority)
    
    type_mapping = {
        'technical': 'ปัญหาทางเทคนิค',
        'service': 'บริการ',
        'billing': 'เรื่องบิล/การเงิน',
        'product': 'เรื่องผลิตภัณฑ์',
        'customer': 'ลูกค้าทั่วไป',
        'returns': 'เรื่องการคืนสินค้า',
        'request': 'คำขอ',
        'incident': 'เหตุการณ์/ปัญหา',
        'thank you': 'ขอบคุณ',
        'change': 'เปลี่ยนแปลง'
    }
    
    type_th = 'อื่นๆ'
    if type_ticket:
        type_lower = type_ticket.lower()
        for key, val in type_mapping.items():
            if key in type_lower:
                type_th = val
                break
    
    queue_mapping = {
        'it': 'ไอที',
        'tech support': 'ฝ่ายซัพพอร์ตเทคนิค',
        'sales': 'ฝ่ายขาย',
        'support': 'ฝ่ายสนับสนุน',
        'documentation': 'เอกสาร',
        'feedback': 'ความคิดเห็น',
        'bug': 'บั๊ก',
        'feature': 'ฟีเจอร์ใหม่',
        'security': 'ความปลอดภัย',
        'network': 'เครือข่าย',
        'hardware': 'ฮาร์ดแวร์',
        'performance': 'ประสิทธิภาพ'
    }
    
    queue_th = queue
    if queue:
        queue_lower = queue.lower()
        for key, val in queue_mapping.items():
            if key in queue_lower:
                queue_th = val
                break
    
    tag_mapping = {
        'security': 'ความปลอดภัย',
        'outage': 'ระบบล่ม',
        'disruption': 'การหยุดชะงัก',
        'data breach': 'ข้อมูลรั่วไหล',
        'it': 'ไอที',
        'tech support': 'ซัพพอร์ตเทคนิค',
        'documentation': 'เอกสาร',
        'feedback': 'ความคิดเห็น',
        'feature': 'ฟีเจอร์',
        'bug': 'บั๊ก',
        'network': 'เครือข่าย',
        'hardware': 'ฮาร์ดแวร์',
        'performance': 'ประสิทธิภาพ',
        'compatibility': 'ความเข้ากันได้',
        'automation': 'ระบบอัตโนมัติ',
        'vpn': 'วีพีเอ็น',
        'audio': 'เสียง',
        'video': 'วิดีโอ'
    }
    
    def translate_tag(tag):
        if not tag:
            return ''
        tag_lower = tag.lower()
        for key, val in tag_mapping.items():
            if key in tag_lower:
                return f"{val} ({tag})"
        return tag
    
    tags_th = ', '.join(filter(None, [translate_tag(tag_1), translate_tag(tag_2), 
                                       translate_tag(tag_3), translate_tag(tag_4)]))
    
    parts = [
        f"หัวข้อคำถาม: {subject}",
        f"รายละเอียด: {body}",
        f"ประเภท: {type_th} ({type_ticket})",
        f"แผนกที่รับผิดชอบ: {queue_th}",
        f"ระดับความสำคัญ: {priority_th}",
        f"แท็ก/หมวดหมู่: {tags_th}" if tags_th else "",
        f"ภาษา: {lang}",
        f"--- สำหรับค้นหา ---",
        f"keyword: ticket support help คำถาม ปัญหา แก้ไข {subject} {body} {type_ticket} {queue} {type_th} {queue_th} {tag_1} {tag_2} {tag_3} {tag_4}"
    ]
    
    return "\n".join([p for p in parts if p])

# Document Loaders
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
            print(f"  Sheet '{sheet_name}': {len([d for d in all_docs if d.metadata['sheet'] == sheet_name])} รายการ")
        except Exception as e:
            print(f"  Error loading {sheet_name}: {e}")
    return all_docs

def load_support_tickets(file_path: str, sheet_name: str = "Sheet1") -> List[Document]:
    docs = []
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str).dropna(how="all")
        df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
        
        print(f"  พบคอลัมน์: {', '.join(df.columns.tolist())}")
        
        for idx, row in df.iterrows():
            data = row.to_dict()
            if not clean_text(data.get('subject')): 
                continue
            
            content = build_ticket_content(data)
            
            metadata = {
                "source": "support_tickets",
                "type": str(data.get('type', '')),
                "priority": str(data.get('priority', '')),
                "queue": str(data.get('queue', '')),
                "row": int(idx)
            }
            
            docs.append(Document(page_content=content, metadata=metadata))
        
        del df
        print(f"  โหลดสำเร็จ: {len(docs)} tickets")
    except Exception as e:
        print(f"  Error loading support tickets: {e}")
    
    return docs

# Batch Embedding with Progress
def batch_embed_documents(docs: List[Document], embeddings, batch_size: int = 50):
    print(f"\nเริ่มต้น Embedding {len(docs)} เอกสาร (Batch size: {batch_size})")
    
    all_embedded = []
    total_batches = (len(docs) + batch_size - 1) // batch_size
    
    with tqdm(total=len(docs), desc="Embedding Documents", unit="doc", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                texts = [doc.page_content for doc in batch]
                embedded_texts = embeddings.embed_documents(texts)
                
                for doc, embedding in zip(batch, embedded_texts):
                    all_embedded.append((doc, embedding))
                
                pbar.update(len(batch))
                pbar.set_postfix({
                    "Batch": f"{batch_num}/{total_batches}",
                    "Completed": f"{len(all_embedded)}/{len(docs)}"
                })
                
            except Exception as e:
                print(f"\nError in batch {batch_num}: {e}")
                pbar.update(len(batch))
    
    print(f"Embedding เสร็จสิ้น: {len(all_embedded)}/{len(docs)} เอกสาร\n")
    return all_embedded

# Save to Database with Progress
def save_to_database_with_progress(docs: List[Document], embeddings, 
                                   collection_name: str, connection_string: str,
                                   batch_size: int = 100):
    print(f"บันทึก {len(docs)} เอกสารลง PostgreSQL...")
    print(f"   Collection: {collection_name}")
    print(f"   Batch size: {batch_size}\n")
    
    try:
        conn = psycopg2.connect(connection_string.replace('postgresql+psycopg2://', 'postgresql://'))
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;")
        cursor.execute(f"DROP TABLE IF EXISTS langchain_pg_collection CASCADE;")
        conn.commit()
        cursor.close()
        conn.close()
        print("ล้างข้อมูลเก่าสำเร็จ\n")
    except Exception as e:
        print(f"Warning: {e}\n")
    
    total_batches = (len(docs) + batch_size - 1) // batch_size
    saved_count = 0
    
    with tqdm(total=len(docs), desc="Saving to Database", unit="doc",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                if i == 0:
                    vectorstore = PGVector.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        collection_name=collection_name,
                        connection_string=connection_string,
                        pre_delete_collection=False
                    )
                else:
                    vectorstore = PGVector(
                        connection_string=connection_string,
                        collection_name=collection_name,
                        embedding_function=embeddings
                    )
                    vectorstore.add_documents(batch)
                
                saved_count += len(batch)
                
                pbar.update(len(batch))
                pbar.set_postfix({
                    "Batch": f"{batch_num}/{total_batches}",
                    "Saved": f"{saved_count}/{len(docs)}",
                    "Progress": f"{(saved_count/len(docs)*100):.1f}%"
                })
                
            except Exception as e:
                print(f"\nError in batch {batch_num}: {e}")
                pbar.update(len(batch))
    
    print(f"\nบันทึกเสร็จสิ้น: {saved_count}/{len(docs)} เอกสาร")
    return saved_count

# Main Ingestion
def ingest_real_inventory():
    start_time = time.time()
    
    print("="*70)
    print(" IT SUPPORT KNOWLEDGE BASE - FULL INGESTION")
    print("="*70)
    
    all_docs = []
    
    print("\n[1/2] กำลังโหลด IT Inventory...")
    if os.path.exists(INVENTORY_FILE):
        inventory_docs = load_inventory_documents(INVENTORY_FILE, TARGET_SHEETS)
        all_docs.extend(inventory_docs)
        print(f"  รวม Inventory: {len(inventory_docs)} รายการ")
    else:
        print(f"  ไม่พบไฟล์: {INVENTORY_FILE}")
    
    print("\n[2/2] กำลังโหลด Support Tickets...")
    if os.path.exists(SUPPORT_TICKETS_FILE):
        ticket_docs = load_support_tickets(SUPPORT_TICKETS_FILE)
        all_docs.extend(ticket_docs)
        print(f"  รวม Support Tickets: {len(ticket_docs)} รายการ")
    else:
        print(f"  ไม่พบไฟล์: {SUPPORT_TICKETS_FILE}")
    
    if not all_docs:
        print("\nไม่พบข้อมูลที่ใช้งานได้")
        return

    print("\n" + "="*70)
    print(f"สรุปข้อมูลที่โหลด")
    print("="*70)
    print(f"  • Inventory Items: {len([d for d in all_docs if d.metadata.get('source') == 'inventory'])}")
    print(f"  • Support Tickets: {len([d for d in all_docs if d.metadata.get('source') == 'support_tickets'])}")
    print(f"  • รวมทั้งหมด: {len(all_docs)} เอกสาร")
    print("="*70)
    
    print("\nเตรียม Embedding Model...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    print(f"  ใช้ Model: {EMBED_MODEL}\n")
    
    try:
        embedded_docs = batch_embed_documents(all_docs, embeddings, batch_size=50)
        
        saved_count = save_to_database_with_progress(
            docs=all_docs,
            embeddings=embeddings,
            collection_name=COLLECTION,
            connection_string=DB_URL,
            batch_size=100
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*70)
        print(" INGESTION สำเร็จ!")
        print(" ระบบพร้อมตอบคำถามภาษาไทยจากข้อมูลภาษาอังกฤษ")
        print("="*70)
        print(f"\nรายละเอียด:")
        print(f"  • Collection: {COLLECTION}")
        print(f"  • จำนวนเอกสารที่บันทึก: {saved_count}/{len(all_docs)}")
        print(f"  • Embedding Model: {EMBED_MODEL}")
        print(f"  • Database: {DB_URL.split('@')[1]}")
        print(f"  • เวลาที่ใช้: {elapsed_time:.2f} วินาที ({elapsed_time/60:.2f} นาที)")
        print("="*70)
        
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    ingest_real_inventory()