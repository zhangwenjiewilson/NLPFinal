"""
Merge the single-page PDFs in base_dataset/<split>/pdfs/
into one multi-page PDF per original document.

Usage:
    python scripts/stitch_doclaynet_base.py \
           --root base_dataset/train \
           --dst  input_pdfs/doclaynet_base_train
"""
import json, fitz, argparse
from pathlib import Path
from tqdm import tqdm

def stitch(split_root: Path, dst: Path):
    ann_dir  = split_root / "annotations"
    pdf_dir  = split_root / "pdfs"
    dst.mkdir(parents=True, exist_ok=True)

    # 1) bucket page-PDFs by original document
    buckets = {}   # {doc_name: [(page_no, one_page_pdf)]}
    # --- 1) bucket page-PDFs by original document ---------------------------
    # --- 1) bucket page-PDFs by original document ---------------------------
    for ann_file in ann_dir.rglob("*.json"):
        meta = json.loads(ann_file.read_text())

        # DocLayNet-base 2023 format
        doc = (
                meta.get("metadata", {}).get("original_filename")  # << main field
                or meta.get("doc_name")  # fallback
        )
        if doc is None:
            print(f"⚠️  no document id in {ann_file.name}, skipped")
            continue

        page = meta.get("metadata", {}).get("page_no", meta.get("page_no", 0))
        guid = ann_file.stem  # hash – matches 1-page PDF name
        one_page_pdf = pdf_dir / f"{guid}.pdf"
        if not one_page_pdf.exists():
            print(f"⚠️  missing {one_page_pdf.name}, skipped")
            continue

        buckets.setdefault(doc, []).append((page, one_page_pdf))

    # 2) merge the buckets
    for doc, pages in tqdm(buckets.items(), desc=f"Stitching {split_root.name}"):
        pages.sort()                       # by page_no
        out_file = dst / doc
        if out_file.exists():
            continue
        doc_out = fitz.open()
        for _, p_file in pages:
            doc_out.insert_pdf(fitz.open(p_file))
        doc_out.save(out_file, deflate=True)
        doc_out.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="train/  val/  or test/ directory that "
                         "contains annotations/, pdfs/")
    ap.add_argument("--dst",  required=True,
                    help="folder for stitched multi-page PDFs")
    args = ap.parse_args()
    stitch(Path(args.root), Path(args.dst))
