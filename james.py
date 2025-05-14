from PIL import Image
if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.BILINEAR

import os
import cv2
import json
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import layoutparser as lp

def setup_model():
    weights_url = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/model_final?dl=1"
    if "?" in weights_url:
        weights_url = weights_url.split("?")[0]
    print("Using weights:", weights_url)
    model = lp.models.Detectron2LayoutModel(
        config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8,
            "MODEL.WEIGHTS", weights_url
        ]
    )
    return model

def process_page(image, model):
    layout = model.detect(image)
    regions = []
    for block in layout:
        x1, y1, x2, y2 = map(int, block.coordinates)
        crop = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(crop)
        region_info = {
            "bbox": [x1, y1, x2, y2],
            "label": block.type,
            "score": float(block.score),
            "text": text.strip()
        }
        regions.append(region_info)
    return regions

def process_pdf(pdf_path, model):
    pages = convert_from_path(pdf_path, dpi=300)
    document_json = {"document": os.path.basename(pdf_path), "pages": []}
    for idx, page in enumerate(pages):
        image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        regions = process_page(image, model)
        page_json = {"page_number": idx + 1, "regions": regions}
        document_json["pages"].append(page_json)
    return document_json

def process_image(image_path, model):
    """
    Process a single-page image file and return JSON with detected layout regions and OCR text.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    regions = process_page(img, model)
    document_json = {"document": os.path.basename(image_path), "pages": []}
    page_json = {"page_number": 1, "regions": regions}
    document_json["pages"].append(page_json)
    return document_json

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="PDF-to-JSON Conversion using LayoutParser with a pre-trained PubLayNet model"
    )
    parser.add_argument("--pdf", type=str, required=True, help="Path to the academic paper PDF")
    parser.add_argument("--output", type=str, required=True, help="Path for the output JSON file")
    args = parser.parse_args()
    layout_model = setup_model()
    output_json = process_pdf(args.pdf, layout_model)
    with open(args.output, "w") as f:
        json.dump(output_json, f, indent=4)
    print(f"Conversion complete. Output saved to {args.output}")
