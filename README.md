# PDF Component Segregator (YOLO DocLayNet + pytesseract)

This repo contains a **simple, modular Python application** that:

- Takes a **PDF** as input
- Uses **YOLO DocLayNet** to detect document layout components (pictures, captions, text blocks, tables, etc.)
- Uses **pytesseract** to extract text inside each detected region
- Saves:
  - Cropped **images** for all `"Picture"` regions
  - A single `components.json` file describing all detected components and their extracted text

This directly matches the assignment requirement:  
> *“Take input as PDF and segregate its components using YOLO DocLayNet for image extraction and pytesseract for text extraction.”*

---

## 1. Installation

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv/Scripts/activate
pip install -r requirements.txt
```



## 2. Pytesseract Installation
a. Download from: https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe
b. install it and add the path to system environment variables

## 3. Run Application
```bash
python src/code/main.py 
  --pdf <pdf_path>
  --model  <src/code/model/yolov11l-doclaynet.pt>
  --output outputs
```
