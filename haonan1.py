import fitz
import os
import json
import re
import spacy
import tiktoken
from PIL import Image
import pytesseract

# Load spaCy model for sentence splitting
nlp = spacy.load("en_core_web_sm")
# Load toke nizer for token counting
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(tokenizer.encode(text))

def split_sentences_spacy(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def split_long_sentence_safely(sentence, max_tokens=500):
    sub_sentences = re.split(r"[\u3001,;:!\-\u2013\u2014\(\)\[\]\{\}]+", sentence)
    chunks = []
    temp_chunk = ""
    temp_tokens = 0

    for sub in sub_sentences:
        sub = sub.strip()
        if not sub:
            continue
        sub_tokens = count_tokens(sub)

        if sub_tokens > max_tokens:
            token_ids = tokenizer.encode(sub)
            for i in range(0, len(token_ids), max_tokens):
                token_chunk = tokenizer.decode(token_ids[i:i + max_tokens])
                chunks.append(token_chunk.strip())
            continue

        if temp_tokens + sub_tokens <= max_tokens:
            temp_chunk += " " + sub
            temp_tokens += sub_tokens
        else:
            chunks.append(temp_chunk.strip())
            temp_chunk = sub
            temp_tokens = sub_tokens

    if temp_chunk:
        chunks.append(temp_chunk.strip())

    return chunks

def chunk_text(text, max_tokens=500):
    sentences = split_sentences_spacy(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)
        if sent_tokens > max_tokens:
            # Sentence too long, split it further
            sub_chunks = split_long_sentence_safely(sent, max_tokens)
            chunks.extend(sub_chunks)
            continue

        if current_tokens + sent_tokens <= max_tokens:
            current_chunk += " " + sent
            current_tokens += sent_tokens
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent
            current_tokens = sent_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def extract_with_pymupdf(pdf_path, output_dir="output_json"):
    doc = fitz.open(pdf_path)
    doc_title = os.path.splitext(os.path.basename(pdf_path))[0]
    doc_metadata = doc.metadata

    structured_data = {
        "document_title": doc_title,
        "metadata": {
            "source": os.path.basename(pdf_path),
            "page_count": len(doc),
            "title": doc_metadata.get("title"),
            "author": doc_metadata.get("author"),
            "creationDate": doc_metadata.get("creationDate"),
            "modDate": doc_metadata.get("modDate"),
            "subject": doc_metadata.get("subject")
        },
        "content": []
    }

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        sections = []
        tables = []
        images = []

        # Process text blocks
        for block in blocks:
            if block["type"] != 0:
                continue  # skip non-text blocks
            text = " ".join([span["text"] for line in block["lines"] for span in line["spans"]]).strip()
            if not text:
                continue
            font_size = block["lines"][0]["spans"][0].get("size", 12)
            if font_size >= 14:
                # Likely a heading
                sections.append({
                    "heading": text,
                    "text": "",
                    "chunks": []
                })
            else:
                # Body text: attach to last heading if exists without text, otherwise new section
                if sections and not sections[-1]["text"]:
                    sections[-1]["text"] = text
                    sections[-1]["chunks"] = chunk_text(text)
                else:
                    sections.append({
                        "heading": "Untitled Section",
                        "text": text,
                        "chunks": chunk_text(text)
                    })

        # Process tables if any
        try:
            for table in page.find_tables():
                tables.append({
                    "bbox": table.bbox,
                    "page_number": page_num,
                    "data": table.extract()
                })
        except Exception as e:
            print(f"Table parsing failed on page {page_num}: {e}")

        # Extract images (figures) if any
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            image_info = doc.extract_image(xref)
            image_bytes = image_info["image"]
            image_ext = image_info["ext"]
            image_name = f"{doc_title}_page_{page_num}_img_{img_index}.{image_ext}"
            image_path = os.path.join(output_dir, image_name)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            images.append({
                "page_number": page_num,
                "file": image_name,
                "bbox": None
            })

        structured_data["content"].append({
            "page_number": page_num,
            "sections": sections,
            "tables": tables,
            "images": images
        })

    return structured_data

def extract_from_image(image_path):
    """
    Extract text from an image (single page) and structure it similar to extract_with_pymupdf output.
    """
    doc_title = os.path.splitext(os.path.basename(image_path))[0]
    structured_data = {
        "document_title": doc_title,
        "metadata": {
            "source": os.path.basename(image_path),
            "page_count": 1,
            "title": None,
            "author": None,
            "creationDate": None,
            "modDate": None,
            "subject": None
        },
        "content": []
    }
    # Perform OCR on the image
    img = Image.open(image_path)
    ocr_text = pytesseract.image_to_string(img)
    text = ocr_text.strip()
    sections = []
    tables = []
    images = []
    if text:
        # Split into paragraphs by blank lines to create sections
        paragraphs = re.split(r"\n\s*\n", text)
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Replace line breaks within a paragraph with space
            para_text = para.replace("\n", " ")
            sections.append({
                "heading": "Untitled Section",
                "text": para_text,
                "chunks": chunk_text(para_text)
            })
    structured_data["content"].append({
        "page_number": 1,
        "sections": sections,
        "tables": tables,
        "images": images
    })
    return structured_data

def convert_pdf_to_json(pdf_path, base_output_dir="output_json"):
    doc_title = os.path.splitext(os.path.basename(pdf_path))[0]
    doc_output_dir = os.path.join(base_output_dir, doc_title)
    os.makedirs(doc_output_dir, exist_ok=True)

    structured = extract_with_pymupdf(pdf_path, doc_output_dir)
    json_path = os.path.join(doc_output_dir, f"{doc_title}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=4, ensure_ascii=False)
    print(f"Saved structured JSON to {json_path}")

if __name__ == "__main__":
    input_folder = "input_pdfs"     # <-- change this line if needed
    output_folder = "output_pdfs"   # <-- and this line
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            input_pdf_path = os.path.join(input_folder, filename)
            convert_pdf_to_json(input_pdf_path, output_folder)
