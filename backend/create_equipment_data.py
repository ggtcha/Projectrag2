"""
สคริปต์สร้างไฟล์ CSV ตัวอย่างสำหรับข้อมูลอุปกรณ์คอมพิวเตอร์
รันแล้วจะได้ไฟล์ data/computer_equipment.csv
"""
import pandas as pd
import os

# สร้างโฟลเดอร์ data ถ้ายังไม่มี
os.makedirs('data', exist_ok=True)

# ข้อมูลตัวอย่าง - คุณสามารถแก้ไขหรือเพิ่มได้
equipment_data = [
    # === LAPTOPS ===
    {
        'name': 'ThinkPad X1 Carbon Gen 11',
        'brand': 'Lenovo',
        'price': '45,900 บาท',
        'category': 'Laptop',
        'specs': 'Intel Core i7-1355U, 16GB LPDDR5 RAM, 512GB NVMe SSD, 14" WUXGA Display, Win 11 Pro',
        'description': 'Business ultrabook น้ำหนักเบา ทนทาน ทำงานได้ยาวนาน เหมาะสำหรับ executive และ mobile workers',
        'availability': 'In Stock',
        'url': 'https://www.lenovo.com/th/en/p/laptops/thinkpad/thinkpadx1/thinkpad-x1-carbon-gen-11'
    },
    {
        'name': 'ThinkPad T14 Gen 4',
        'brand': 'Lenovo',
        'price': '35,900 บาท',
        'category': 'Laptop',
        'specs': 'Intel Core i5-1335U, 16GB DDR4, 256GB SSD, 14" FHD Display, Win 11 Pro',
        'description': 'Mainstream business laptop ประสิทธิภาพดี ราคาเหมาะสม เหมาะสำหรับงานทั่วไป',
        'availability': 'In Stock',
        'url': 'https://www.lenovo.com/th/en/p/laptops/thinkpad/thinkpadt/thinkpad-t14-gen-4'
    },
    {
        'name': 'Dell Latitude 5440',
        'brand': 'Dell',
        'price': '38,500 บาท',
        'category': 'Laptop',
        'specs': 'Intel Core i7-1355U, 16GB DDR4, 512GB SSD, 14" FHD, Win 11 Pro',
        'description': 'Enterprise laptop ที่มีความปลอดภัยสูง มี Dell Optimizer และ ExpressConnect',
        'availability': 'In Stock',
        'url': 'https://www.dell.com/en-us/shop/laptops/latitude-5440/spd/latitude-14-5440-laptop'
    },
    {
        'name': 'HP EliteBook 840 G10',
        'brand': 'HP',
        'price': '42,900 บาท',
        'category': 'Laptop',
        'specs': 'Intel Core i7-1355U, 16GB DDR5, 512GB SSD, 14" WUXGA, Win 11 Pro',
        'description': 'Premium business laptop พร้อม HP Sure Sense และ HP Wolf Security',
        'availability': 'In Stock',
        'url': 'https://www.hp.com/us-en/shop/pdp/hp-elitebook-840-g10'
    },
    {
        'name': 'ASUS VivoBook 15',
        'brand': 'ASUS',
        'price': '18,900 บาท',
        'category': 'Laptop',
        'specs': 'Intel Core i5-1235U, 8GB DDR4, 512GB SSD, 15.6" FHD, Win 11 Home',
        'description': 'Everyday laptop ราคาประหยัด เหมาะสำหรับงานทั่วไป',
        'availability': 'In Stock',
        'url': 'https://www.asus.com/laptops/for-home/vivobook/vivobook-15/'
    },
    
    # === DESKTOPS ===
    {
        'name': 'ThinkCentre M70q Gen 4',
        'brand': 'Lenovo',
        'price': '22,900 บาท',
        'category': 'Desktop',
        'specs': 'Intel Core i5-13400, 8GB DDR4, 256GB SSD, Tiny Form Factor, Win 11 Pro',
        'description': 'Compact desktop ประหยัดพื้นที่ ประสิทธิภาพดี เหมาะสำหรับออฟฟิศ',
        'availability': 'In Stock',
        'url': 'https://www.lenovo.com/us/en/p/desktops/thinkcentre/m-series-tiny/thinkcentre-m70q-gen-4'
    },
    {
        'name': 'Dell OptiPlex 7090 SFF',
        'brand': 'Dell',
        'price': '28,500 บาท',
        'category': 'Desktop',
        'specs': 'Intel Core i7-11700, 16GB DDR4, 512GB SSD, Small Form Factor, Win 11 Pro',
        'description': 'Business desktop ที่มีความปลอดภัยสูง รองรับ Dell Optimizer',
        'availability': 'In Stock',
        'url': 'https://www.dell.com/en-us/shop/desktop-computers/optiplex-7090-small-form-factor/spd/optiplex-7090-desktop'
    },
    {
        'name': 'HP ProDesk 400 G9 SFF',
        'brand': 'HP',
        'price': '24,900 บาท',
        'category': 'Desktop',
        'specs': 'Intel Core i5-12500, 8GB DDR4, 256GB SSD, Small Form Factor, Win 11 Pro',
        'description': 'Reliable business desktop ราคาเหมาะสม เหมาะสำหรับ SMB',
        'availability': 'In Stock',
        'url': 'https://www.hp.com/us-en/shop/pdp/hp-prodesk-400-g9-small-form-factor-pc'
    },
    
    # === MONITORS ===
    {
        'name': 'ThinkVision T27hv-30',
        'brand': 'Lenovo',
        'price': '12,900 บาท',
        'category': 'Monitor',
        'specs': '27" QHD (2560x1440), IPS Panel, 75Hz, USB-C Hub, Built-in Webcam',
        'description': 'Monitor พร้อม webcam และ USB-C hub เหมาะสำหรับ WFH และ video conference',
        'availability': 'In Stock',
        'url': 'https://www.lenovo.com/us/en/p/accessories-and-software/monitors/professional/62afuat1us'
    },
    {
        'name': 'Dell UltraSharp U2723DE',
        'brand': 'Dell',
        'price': '14,900 บาท',
        'category': 'Monitor',
        'specs': '27" QHD (2560x1440), IPS Black, USB-C 90W, RJ45 Ethernet Hub',
        'description': 'Premium monitor สีสันสดใส contrast สูง พร้อม USB-C docking',
        'availability': 'In Stock',
        'url': 'https://www.dell.com/en-us/shop/dell-ultrasharp-27-usb-c-hub-monitor-u2723de/apd/210-bdiv'
    },
    
    # === PRINTERS ===
    {
        'name': 'HP LaserJet Pro M404dn',
        'brand': 'HP',
        'price': '12,500 บาท',
        'category': 'Printer',
        'specs': 'Monochrome Laser, 38 ppm, Duplex, Network, 350-sheet Input',
        'description': 'Fast reliable laser printer เหมาะสำหรับ workgroup ขนาดกลาง',
        'availability': 'In Stock',
        'url': 'https://www.hp.com/us-en/shop/pdp/hp-laserjet-pro-m404dn'
    },
    {
        'name': 'HP Color LaserJet Pro M454dw',
        'brand': 'HP',
        'price': '18,900 บาท',
        'category': 'Printer',
        'specs': 'Color Laser, 28 ppm, Duplex, WiFi, Network, 300-sheet Input',
        'description': 'Color laser printer คุณภาพสูง เหมาะสำหรับพิมพ์เอกสารสี',
        'availability': 'In Stock',
        'url': 'https://www.hp.com/us-en/shop/pdp/hp-color-laserjet-pro-m454dw'
    },
    {
        'name': 'Canon imageRUNNER 2425i',
        'brand': 'Canon',
        'price': '35,900 บาท',
        'category': 'MFP',
        'specs': 'Monochrome MFP, 25 ppm, Print/Copy/Scan, Network, Duplex, ADF',
        'description': 'Multifunction printer เหมาะสำหรับออฟฟิศขนาดกลาง',
        'availability': 'In Stock',
        'url': 'https://www.canon.co.th/en/business-products/office-copiers-printers/imagerunner-2425i'
    },
    
    # === NETWORKING ===
    {
        'name': 'HP Aruba 2930F 48G 4SFP+ Switch',
        'brand': 'HP',
        'price': '45,900 บาท',
        'category': 'Network Switch',
        'specs': '48-port Gigabit, 4x SFP+ 10G, Layer 3, Stackable, PoE+ (370W)',
        'description': 'Enterprise switch รองรับ Layer 3 routing และ PoE+ สำหรับ access points',
        'availability': 'In Stock',
        'url': 'https://www.hpe.com/us/en/product-catalog/networking/networking-switches/pip.aruba-2930f-switch-series.1008995367.html'
    },
    {
        'name': 'Cisco Catalyst 9200L 48-port',
        'brand': 'Cisco',
        'price': '52,900 บาท',
        'category': 'Network Switch',
        'specs': '48-port Gigabit, 4x 10G SFP+, Layer 3, Stackable, PoE+ (740W)',
        'description': 'Enterprise-grade switch พร้อม Cisco DNA และ security features',
        'availability': 'In Stock',
        'url': 'https://www.cisco.com/c/en/us/products/switches/catalyst-9200-series-switches/'
    },
    {
        'name': 'Ubiquiti UniFi Dream Machine Pro',
        'brand': 'Ubiquiti',
        'price': '18,900 บาท',
        'category': 'Network Gateway',
        'specs': 'All-in-one Gateway, 8-port Gigabit, 2x SFP+, UniFi Controller, IPS/IDS',
        'description': 'Complete network solution สำหรับ SMB พร้อม management และ security',
        'availability': 'In Stock',
        'url': 'https://store.ui.com/us/en/products/udm-pro'
    },
    
    # === STORAGE ===
    {
        'name': 'Synology DS923+',
        'brand': 'Synology',
        'price': '24,900 บาท',
        'category': 'NAS',
        'specs': '4-Bay NAS, AMD Ryzen R1600, 4GB DDR4 (expandable to 32GB), Dual 1GbE',
        'description': 'Professional NAS พร้อม snapshot replication และ Synology Drive',
        'availability': 'In Stock',
        'url': 'https://www.synology.com/en-global/products/DS923+'
    },
    {
        'name': 'QNAP TS-464-8G',
        'brand': 'QNAP',
        'price': '22,900 บาท',
        'category': 'NAS',
        'specs': '4-Bay NAS, Intel Celeron N5105, 8GB DDR4, Dual 2.5GbE, PCIe Gen3 slot',
        'description': 'Versatile NAS พร้อม 2.5GbE และ PCIe expansion',
        'availability': 'In Stock',
        'url': 'https://www.qnap.com/en-us/product/ts-464'
    },
    
    # === ACCESSORIES ===
    {
        'name': 'Logitech MX Keys for Business',
        'brand': 'Logitech',
        'price': '4,290 บาท',
        'category': 'Keyboard',
        'specs': 'Wireless Keyboard, Backlit, USB-C Rechargeable, Multi-device',
        'description': 'Premium wireless keyboard สำหรับ productivity สูง',
        'availability': 'In Stock',
        'url': 'https://www.logitech.com/en-us/products/keyboards/mx-keys-for-business.html'
    },
    {
        'name': 'Logitech MX Master 3S',
        'brand': 'Logitech',
        'price': '3,490 บาท',
        'category': 'Mouse',
        'specs': 'Wireless Mouse, 8000 DPI, Quiet Clicks, USB-C Rechargeable, Multi-device',
        'description': 'Flagship wireless mouse พร้อม precision tracking',
        'availability': 'In Stock',
        'url': 'https://www.logitech.com/en-us/products/mice/mx-master-3s.html'
    },
    {
        'name': 'APC Back-UPS Pro 1500VA',
        'brand': 'APC',
        'price': '8,900 บาท',
        'category': 'UPS',
        'specs': '1500VA/900W, LCD Display, 10 Outlets, USB Monitoring, AVR',
        'description': 'Line-interactive UPS ป้องกันไฟกระชาก เหมาะสำหรับ workstation',
        'availability': 'In Stock',
        'url': 'https://www.apc.com/us/en/product/BR1500MS/apc-backups-pro-1500va-10-outlet-tower-ups/'
    },
]

# สร้าง DataFrame
df = pd.DataFrame(equipment_data)

# บันทึกเป็น CSV
output_file = 'data/computer_equipment.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print("="*70)
print(" ✅ สร้างไฟล์สำเร็จ!")
print("="*70)
print(f"ไฟล์: {output_file}")
print(f"จำนวนรายการ: {len(df)}")
print(f"\nตัวอย่างข้อมูล:")
print(df[['name', 'brand', 'price', 'category']].head(10).to_string())
print("\n" + "="*70)
print("คุณสามารถ:")
print("1. เปิดไฟล์ CSV ด้วย Excel/Google Sheets เพื่อแก้ไข/เพิ่มข้อมูล")
print("2. รัน ingestion.py เพื่อโหลดข้อมูลเข้า database")
print("="*70)