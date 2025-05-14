import os, json, argparse
from james import setup_model, process_pdf   # imports from james.py

def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # load the PubLayNet model once
    model = setup_model()

    for fname in os.listdir(in_dir):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(in_dir, fname)
            print(f"Processing {fname} …")
            doc_json = process_pdf(pdf_path, model)

            out_name = f"{os.path.splitext(fname)[0]}_layout.json"
            out_path = os.path.join(out_dir, out_name)
            with open(out_path, "w") as f:
                json.dump(doc_json, f, indent=4)
            print(f"  ↳ saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",  default="input_pdfs")   # your existing folder
    parser.add_argument("--out_dir", default="james_output") # <── changed here
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)
