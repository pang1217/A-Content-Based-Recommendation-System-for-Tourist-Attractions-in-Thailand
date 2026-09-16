# A-Content-Based-Recommendation-System-for-Tourist-Attractions-in-Thailand

ระบบแนะนำสถานที่ท่องเที่ยวในไทย (content-based recommendation) จากคำอธิบายสถานที่ภาษาไทย โดยรองรับโมเดล 4 แบบ ได้แก่ **TF-IDF**, **Thai2Vec (Word2Vec)**, **FastText** และ **Sentence-BERT** สามารถแนะนำสถานที่ที่คล้ายกันจาก `place_id`, ชื่อสถานที่ หรือข้อความค้นหาอิสระ (free-text query) ได้

## คุณสมบัติหลัก

- ทำความสะอาดข้อมูล (drop ค่าว่าง/placeholder, ลบ HTML tag, ลบข้อมูลซ้ำ)
- Preprocess และ tokenize ข้อความภาษาไทย (ใช้ `pythainlp` ถ้ามี ไม่มีก็ fallback เป็น split แบบง่าย)
- คำนวณความคล้ายกันด้วย Cosine Similarity
- แนะนำสถานที่ได้ 3 รูปแบบ: จาก `--place-id`, จาก `--name`, หรือจาก `--text`
- รันเปรียบเทียบทุกโมเดลพร้อมกันได้ด้วย `--model all` (สรุปคะแนน, จัดอันดับ, เปรียบเทียบผลแบบเคียงข้างกัน)
- บันทึกผลลัพธ์เป็นไฟล์ `.csv` หรือ `.xlsx` (พร้อมชีตสรุป/เปรียบเทียบเมื่อใช้ `--model all`)
- แสดง token และ embedding vector ของ query/ผลลัพธ์ได้ (`--show-tokens`, `--show-vector`)

## โมเดลที่รองรับ

| ค่า `--model` | โมเดล            | หมายเหตุ |
|----------------|-------------------|----------|
| `tfidf`        | TF-IDF            | ค่าเริ่มต้น ใช้ character n-gram (3-5) |
| `thai2vec`     | Word2Vec/Thai2Vec | ใช้โมเดลสำเร็จรูปจาก `pythainlp` ถ้าโหลดไม่ได้จะเทรน Word2Vec เองจากข้อมูล |
| `fasttext`     | FastText          | เทรนจากข้อมูลในไฟล์ CSV โดยตรง |
| `berta`        | Sentence-BERT     | ใช้โมเดล `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (ต้องติดตั้ง `sentence-transformers`) |
| `all`          | ทุกโมเดิน         | รันและเปรียบเทียบทั้ง 4 โมเดลในครั้งเดียว |

## การติดตั้ง

ต้องใช้ Python 3.10 ขึ้นไป (มีการใช้ `X | Y` type hint และ `from __future__ import annotations`)

```bash
pip install pandas numpy beautifulsoup4 scikit-learn gensim openpyxl
```

ไลบรารีเสริม (ไม่บังคับ แต่แนะนำให้ติดตั้งเพื่อผลลัพธ์ที่ดีกว่า):

```bash
# สำหรับ tokenizer ภาษาไทยที่แม่นยำขึ้น และโมเดล Thai2Vec สำเร็จรูป
pip install pythainlp

# สำหรับโมเดล Sentence-BERT (--model berta)
pip install sentence-transformers
```

หากไม่ได้ติดตั้ง `pythainlp` สคริปต์จะ fallback ไปใช้การตัดคำแบบ whitespace-split อย่างง่ายโดยอัตโนมัติ

## ข้อมูลนำเข้า (CSV)

ค่าเริ่มต้นสคริปต์จะมองหาไฟล์ `attraction.csv` ในโฟลเดอร์เดียวกับสคริปต์ (เปลี่ยนได้ด้วย `--csv`) โดยไฟล์ CSV ต้นฉบับต้องมีคอลัมน์ต่อไปนี้ ซึ่งจะถูก map เป็นชื่อคอลัมน์ใหม่โดยอัตโนมัติ:

| คอลัมน์เดิมใน CSV     | คอลัมน์หลัง map     |
|------------------------|----------------------|
| `ATT_ID`               | `place_id`           |
| `ATT_NAME_TH`          | `name`               |
| `ATT_CATEGORY_LABEL`   | `category`           |
| `ATT_DETAIL_TH`        | `description_th`     |
| `ATT_DETAIL_EN`        | `description_en`     |
| `PROVINCE_NAME_TH`     | `location`           |

ระหว่างโหลดข้อมูล สคริปต์จะ:
1. ตัดแถวที่ `ATT_DETAIL_TH` เป็นค่าว่างหรือเป็น `-` / `*`
2. ลบ HTML markup ออกจาก `ATT_DETAIL_TH` และ `ATT_DETAIL_EN`
3. ตัดแถวที่ซ้ำกัน (พิจารณาจากชื่อ + คำอธิบาย)

## วิธีใช้งาน

```bash
python code.py [OPTIONS]
```

รันแบบไม่ใส่อาร์กิวเมนต์ใด ๆ จะใช้โมเดล TF-IDF และแสดงตัวอย่างการแนะนำ 3 แบบ (ตามชื่อสถานที่, ตามข้อความค้นหา, ตามชื่อสถานที่อีกแห่ง)

### อาร์กิวเมนต์หลัก

| อาร์กิวเมนต์      | คำอธิบาย |
|--------------------|----------|
| `--csv PATH`        | พาธไฟล์ CSV (ถ้าไม่ระบุ ใช้ `attraction.csv` ข้าง ๆ สคริปต์) |
| `--model {tfidf,thai2vec,fasttext,berta,all}` | เลือกโมเดล (ค่าเริ่มต้น `tfidf`) |
| `--analyzer {char,word}` | รูปแบบการตัดคำ: `char` (n-gram ตัวอักษร) หรือ `word` (แนะนำสำหรับ Thai2Vec) |
| `--top-n N`          | จำนวนผลลัพธ์ที่แนะนำ (ค่าเริ่มต้น 5) |
| `--place-id ID`      | แนะนำสถานที่ที่คล้ายกับ `place_id` นี้ |
| `--name "ชื่อ"`      | แนะนำสถานที่ที่คล้ายกับชื่อนี้ |
| `--text "คำค้นหา"`   | แนะนำสถานที่จากข้อความค้นหาอิสระ |
| `--save-csv PATH`    | บันทึกผลลัพธ์เป็น CSV |
| `--save-excel PATH`  | บันทึกผลลัพธ์เป็น Excel (.xlsx) |
| `--show-tokens`      | แสดง token ของคำอธิบาย/คำค้นหา |
| `--show-vector`      | แสดง embedding vector (20 มิติแรก) ของคำอธิบาย/คำค้นหา |

> หมายเหตุ: `--place-id`, `--name`, `--text` ใช้ได้ทีละอย่างเท่านั้น (mutually exclusive)

### ตัวอย่างการใช้งาน

แนะนำสถานที่คล้ายกับ place_id ด้วย TF-IDF:
```bash
python code.py --place-id 123 --top-n 10
```

แนะนำจากชื่อสถานที่ด้วย Thai2Vec:
```bash
python code.py --model thai2vec --analyzer word --name "อุทยานแห่งชาติดอยอินทนนท์"
```

ค้นหาด้วยข้อความอิสระ และบันทึกผลเป็น Excel:
```bash
python code.py --model berta --text "ทะเล น้ำใส ดำน้ำ พักผ่อน" --save-excel results.xlsx
```

เปรียบเทียบทุกโมเดลพร้อมกัน พร้อมดู token และ vector:
```bash
python code.py --model all --name "วัดสองพี่น้อง" --show-tokens --show-vector --save-csv all_results.csv
```

## ผลลัพธ์ที่ได้

ผลลัพธ์เป็นตาราง (DataFrame) ที่มีคอลัมน์:

- `source_name` — ชื่อสถานที่ต้นทาง (เมื่อค้นหาแบบ similar-items เช่นจาก `place_id`/`name`)
- `similarity_score` — คะแนนความคล้าย (cosine similarity)
- `place_id`, `name`, `category`, `location`, `description_th` — ข้อมูลสถานที่ที่แนะนำ

เมื่อใช้ `--model all` จะได้เพิ่มเติม:
- **Side-by-side comparison** — เปรียบเทียบผล Top-N ของแต่ละโมเดลในตารางเดียว
- **Model comparison summary** — เวลา fit/query และคะแนน top-1 ของแต่ละโมเดล
- **Model Ranking** — จัดอันดับโมเดลตามคะแนน top-1

และเมื่อบันทึกเป็น Excel (`--save-excel`) พร้อม `--model all` จะมีชีต `summary`, `model_ranking`, `comparison` และชีตผลลัพธ์แยกตามโมเดล

## โครงสร้างโค้ดโดยสรุป

- `clean_dataset`, `clean_html`, `clean_thai_text` — ทำความสะอาดข้อมูลดิบและข้อความ
- `get_thai_word_tokenizer`, `get_char_ngram_tokenizer` — ตัวตัดคำ
- `BaseRecommender` — abstract base class กำหนด interface ร่วมของโมเดลแนะนำ
- `TourismTFIDFRecommender`, `TourismThai2VecRecommender`, `TourismFastTextRecommender`, `TourismSentenceBERTRecommender` — โมเดลแนะนำแต่ละแบบ
- `build_recommender`, `run_all_models` — factory และตัวรันเปรียบเทียบโมเดล
- `build_parser`, `main` — CLI entry point

## ข้อจำกัด / หมายเหตุ

- การเทรน Word2Vec/FastText ในสคริปต์นี้เป็นการเทรนจากข้อมูลชุดเล็กในไฟล์ CSV เอง (ไม่ใช่ pretrained ขนาดใหญ่) ผลลัพธ์จึงขึ้นกับขนาด/คุณภาพของข้อมูล
- โมเดล Sentence-BERT ต้องเชื่อมต่ออินเทอร์เน็ต (หรือมีโมเดลอยู่ใน cache ของ Hugging Face แล้ว) เพื่อดาวน์โหลดโมเดลครั้งแรก
