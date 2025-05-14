import json
import difflib
import math
from collections import Counter
from typing import List, Dict, Any



# Helper function for tokenization (simple whitespace splitter for now)
def tokenize(text: str) -> List[str]:
    # Basic tokenization: split on whitespace and punctuation.
    # For more robust tokenization, consider regex or nltk word_tokenize.
    tokens = []
    word = ''
    for ch in text:
        if ch.isalnum():
            word += ch
        else:
            if word:
                tokens.append(word)
                word = ''
            if ch.isspace():
                continue
            # treat punctuation as separate token if needed
            # (could skip punctuation by not appending it, depending on needs)
            tokens.append(ch)
    if word:
        tokens.append(word)
    return tokens


class ContentEvaluator:
    def __init__(self):
        self.name = "content_fidelity"

    def evaluate(self, docA: Dict, docB: Dict) -> Dict:
        """Compare the textual content of two documents and compute overlap metrics."""

        # Combine all text from the documents. Assuming doc structure has 'content' or similar.
        # We attempt to get the full text in reading order from each JSON.
        def get_full_text(doc: Dict) -> str:
            text_parts = []
            if "content" in doc and isinstance(doc["content"], list):
                # If content is organized by page or sections
                for page in doc["content"]:
                    if isinstance(page, dict) and "sections" in page:
                        for section in page["sections"]:
                            # Each section might have 'text' field
                            if "text" in section:
                                text_parts.append(section["text"])
                    elif isinstance(page, dict) and "text" in page:
                        # Some outputs might just have text per page
                        text_parts.append(page["text"])
                    elif isinstance(page, str):
                        # content list could also be strings already
                        text_parts.append(page)
            elif "text" in doc:
                # If the doc JSON directly has a full text field
                text_parts.append(doc["text"])
            else:
                # Fallback: if we have chunks or paragraphs
                if "chunks" in doc:
                    # doc might itself represent content
                    text_parts.extend(doc["chunks"])
            return "\n".join(text_parts)

        textA = get_full_text(docA)
        textB = get_full_text(docB)
        # Tokenize texts
        tokensA = tokenize(textA)
        tokensB = tokenize(textB)

        total_A = len(tokensA)
        total_B = len(tokensB)

        # Use SequenceMatcher for sequence alignment
        matcher = difflib.SequenceMatcher(a=tokensA, b=tokensB)
        opcodes = matcher.get_opcodes()
        matched_tokens = 0
        # We'll also record differences for optional output
        diff_list = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                matched_tokens += (i2 - i1)
            elif tag == 'replace':
                # Tokens from A (i1:i2) replaced by tokens from B (j1:j2)
                ref_text = " ".join(tokensA[i1:i2])
                new_text = " ".join(tokensB[j1:j2])
                diff_list.append({
                    "type": "replace",
                    "ref_text": ref_text,
                    "new_text": new_text
                })
            elif tag == 'delete':
                # Tokens from A were deleted (not present in B)
                ref_text = " ".join(tokensA[i1:i2])
                diff_list.append({
                    "type": "delete",
                    "ref_text": ref_text
                })
            elif tag == 'insert':
                # Tokens from B were inserted (not in A)
                new_text = " ".join(tokensB[j1:j2])
                diff_list.append({
                    "type": "insert",
                    "new_text": new_text
                })
        # Precision/Recall/F1
        precision_B = matched_tokens / total_B if total_B > 0 else 0.0
        recall_B = matched_tokens / total_A if total_A > 0 else 0.0
        f1_B = (2 * precision_B * recall_B / (precision_B + recall_B)) if (precision_B + recall_B) > 0 else 0.0
        # Also compute A vs B (essentially swap roles)
        precision_A = matched_tokens / total_A if total_A > 0 else 0.0  # actually recall_B
        recall_A = matched_tokens / total_B if total_B > 0 else 0.0  # actually precision_B
        f1_A = f1_B  # F1 is symmetrical for swap of P and R values (should be same as f1_B)

        # Compute BLEU-4 using a simple approach (unigram to 4-gram overlap)
        # We'll implement a simple BLEU calculation for demonstration.
        def compute_bleu(reference_tokens: List[str], candidate_tokens: List[str], max_n: int = 4) -> float:
            # count n-grams in ref and candidate
            ref_ngrams = {}
            cand_ngrams = {}
            for n in range(1, max_n + 1):
                for i in range(len(reference_tokens) - n + 1):
                    ngram = tuple(reference_tokens[i:i + n])
                    ref_ngrams.setdefault(n, Counter()).update([ngram])
                for j in range(len(candidate_tokens) - n + 1):
                    ngram = tuple(candidate_tokens[j:j + n])
                    cand_ngrams.setdefault(n, Counter()).update([ngram])
            # calculate overlaps
            weights = [0.25, 0.25, 0.25, 0.25]  # uniform weights for BLEU-4
            score_components = []
            for n in range(1, max_n + 1):
                if n not in cand_ngrams:
                    score_components.append(0.0)
                    continue
                overlap = 0
                total = 0
                for ngram, count in cand_ngrams[n].items():
                    total += count
                    if n in ref_ngrams:
                        overlap += min(count, ref_ngrams[n][ngram])
                precision_n = overlap / total if total > 0 else 0.0
                score_components.append(precision_n)
            # geometric mean of precisions
            geom_mean = 1.0
            for p in score_components:
                if p > 0:
                    geom_mean *= (p ** weights[score_components.index(p)])
                else:
                    geom_mean *= 0.0
            # brevity penalty
            ref_len = len(reference_tokens)
            cand_len = len(candidate_tokens)
            if cand_len == 0:
                bleu = 0.0
            else:
                bp = 1.0
                if cand_len < ref_len:
                    bp = math.exp(1 - ref_len / cand_len)
                bleu = bp * geom_mean
            return bleu

        bleu_B = compute_bleu(tokensA, tokensB, max_n=4)  # A as reference, B as candidate
        bleu_A = compute_bleu(tokensB, tokensA, max_n=4)  # B as reference, A as candidate

        # Bag-of-words overlap for order analysis
        counterA = Counter(tokensA)
        counterB = Counter(tokensB)
        # intersection count of tokens (considering frequency)
        common_tokens = sum((counterA & counterB).values())
        order_preservation = (
                    matched_tokens / common_tokens) if common_tokens > 0 else 1.0  # if no common, irrelevant; if identical sets, ratio indicates order kept

        # Prepare output dictionary
        result = {
            "content_fidelity": {
                "precision": precision_B,
                "recall": recall_B,
                "F1": f1_B,
                "BLEU-4": bleu_B,
                "aligned_tokens": matched_tokens
            },
            # Also include the reverse (A vs B) if needed for symmetry
            "content_fidelity_A_vs_B": {
                "precision": precision_A,
                "recall": recall_A,
                "F1": f1_A,
                "BLEU-4": bleu_A
            },
            "diff_summary": diff_list  # detailed differences (optional in final output)
        }
        # We can include order_preservation here or in coherence module; include here for convenience
        result["content_fidelity"]["order_preservation"] = order_preservation
        return result


class TableEvaluator:
    def __init__(self):
        self.name = "table_fidelity"

    def evaluate(self, docA: Dict, docB: Dict) -> Dict:
        """Evaluate table detection and structure preservation between two docs."""
        tablesA = []
        tablesB = []
        # If the docs have an explicit table list or table sections:
        if "tables" in docA:
            # Assuming docA["tables"] could be a list of table objects or entries
            tablesA = docA["tables"]
        else:
            # Try to find tables in content by keyword if not explicitly listed
            tablesA = self._find_tables_in_text(docA)
        if "tables" in docB:
            tablesB = docB["tables"]
        else:
            tablesB = self._find_tables_in_text(docB)
        countA = len(tablesA)
        countB = len(tablesB)
        matched = 0
        cell_match_total = 0.0
        cell_match_count = 0
        table_structure_scores = []
        # Try to match tables one-to-one (by index or caption similarity)
        usedB = set()
        for i, tableA in enumerate(tablesA):
            # Find best match in B for this table
            best_j = None
            best_overlap = 0.0
            contentA = self._get_table_text(tableA)
            for j, tableB in enumerate(tablesB):
                if j in usedB:
                    continue
                contentB = self._get_table_text(tableB)
                # simple overlap on text content as match criterion
                overlap = 0.0
                if contentA and contentB:
                    setA = set(tokenize(contentA))
                    setB = set(tokenize(contentB))
                    overlap = len(setA & setB) / len(setA | setB) if len(setA | setB) > 0 else 0.0
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_j = j
            if best_j is not None and best_overlap > 0.3:  # overlap threshold to count as same table
                matched += 1
                usedB.add(best_j)
                # If structured (e.g., tableA has cells property), compare structure
                struct_score = self._compare_table_structure(tableA, tablesB[best_j])
                if struct_score is not None:
                    table_structure_scores.append(struct_score)
        # If both lists empty, matched = 0 but that's fine (no tables present in either).
        precision = matched / countB if countB > 0 else (1.0 if countA == 0 else 0.0)
        recall = matched / countA if countA > 0 else (1.0 if countB == 0 else 0.0)
        # Average cell match (if we have any structured comparison done)
        avg_cell_match = sum(table_structure_scores) / len(table_structure_scores) if table_structure_scores else None

        result = {
            "table_fidelity": {
                "tables_count_A": countA,
                "tables_count_B": countB,
                "tables_matched": matched,
                "precision": precision,
                "recall": recall
            }
        }
        if avg_cell_match is not None:
            result["table_fidelity"]["avg_cell_match_rate"] = avg_cell_match
        return result

    def _find_tables_in_text(self, doc: Dict) -> List[str]:
        """Heuristic: find table captions or indications in the doc text."""
        full_text = ""
        if "content" in doc:
            for page in doc["content"]:
                if isinstance(page, dict):
                    if "sections" in page:
                        for sec in page["sections"]:
                            if "text" in sec:
                                full_text += sec["text"] + "\n"
                    elif "text" in page:
                        full_text += page["text"] + "\n"
        elif "text" in doc:
            full_text = doc["text"]
        # Look for lines starting with "Table" or "TABLE"
        tables = []
        for line in full_text.splitlines():
            if line.strip().lower().startswith("table"):
                # consider this line (caption) as a table indicator
                tables.append({"caption": line.strip()})
        return tables

    def _get_table_text(self, table_obj: Any) -> str:
        """Extract all text from a table object (either structured or just caption)."""
        if table_obj is None:
            return ""
        if isinstance(table_obj, dict):
            text_parts = []
            # If it has a caption or text content
            if "text" in table_obj:
                text_parts.append(table_obj["text"])
            if "caption" in table_obj:
                text_parts.append(table_obj["caption"])
            # If it has cells (assuming structure like list of rows)
            if "cells" in table_obj:
                for row in table_obj["cells"]:
                    # row might be list of cell texts
                    if isinstance(row, list):
                        text_parts.append(" ".join(str(cell) for cell in row))
            # Merge all text parts
            return " ".join(text_parts)
        elif isinstance(table_obj, str):
            return table_obj
        return ""

    def _compare_table_structure(self, tableA: Any, tableB: Any) -> float:
        """Compare structured cells of tableA vs tableB if possible. Returns cell match rate or None if not applicable."""
        # If tables have a 'cells' grid
        cellsA = tableA.get("cells") if isinstance(tableA, dict) else None
        cellsB = tableB.get("cells") if isinstance(tableB, dict) else None
        if cellsA and cellsB:
            rowsA = len(cellsA)
            colsA = len(cellsA[0]) if rowsA > 0 and isinstance(cellsA[0], list) else 0
            rowsB = len(cellsB)
            colsB = len(cellsB[0]) if rowsB > 0 and isinstance(cellsB[0], list) else 0
            # If dimensions differ significantly, return a low score
            if rowsA == 0 or rowsB == 0:
                return 0.0
            matched_cells = 0
            total_cells = max(rowsA, rowsB) * max(colsA, colsB)
            # Compare overlapping region
            min_rows = min(rowsA, rowsB)
            min_cols = min(colsA, colsB)
            for r in range(min_rows):
                for c in range(min_cols):
                    cellA_text = str(cellsA[r][c])
                    cellB_text = str(cellsB[r][c])
                    if cellA_text.strip() == cellB_text.strip():
                        matched_cells += 1
            return matched_cells / total_cells
        # If no structured cells available, return None (use text comparison only)
        return None


class HeadingEvaluator:
    def __init__(self):
        self.name = "headings"

    def evaluate(self, docA: Dict, docB: Dict) -> Dict:
        """Compare section headings and hierarchy."""
        headingsA = self._extract_headings(docA)
        headingsB = self._extract_headings(docB)
        countA = len(headingsA)
        countB = len(headingsB)
        # Match headings by text (case-insensitive, ignoring minor whitespace)
        matched = 0
        matched_headings = []
        headingsB_lower = {h.lower(): h for h in headingsB}
        for h in headingsA:
            h_norm = h.lower().strip()
            if h_norm in headingsB_lower:
                matched += 1
                matched_headings.append(h)
        heading_recall = matched / countA if countA > 0 else 1.0
        heading_precision = matched / countB if countB > 0 else 1.0
        # (If countA=0 and countB=0, define both as 1 meaning no headings needed and none found)
        # Hierarchy comparison (if hierarchy info present, e.g., numbers like 1.1 etc.)
        # For simplicity, let's attempt to detect numbering scheme and compare the sequence.
        hierarchy_match = None
        # (One could implement a proper hierarchy check by parsing numbers and comparing relative ordering)

        result = {
            "structure": {
                "headings_count": {
                    "A": countA,
                    "B": countB
                },
                "headings_percent": (heading_recall * 100.0) if countA > 0 else (100.0 if countB == 0 else 0.0),
                "headings_matched": matched,
                "heading_recall": heading_recall,
                "heading_precision": heading_precision
            }
        }
        if hierarchy_match is not None:
            result["structure"]["hierarchy_match_score"] = hierarchy_match
        return result

    def _extract_headings(self, doc: Dict) -> List[str]:
        """Extract heading texts from doc if available in structured form, else heuristically."""
        headings = []
        # If sections have a 'heading' field that is not a generic placeholder:
        if "content" in doc:
            for page in doc["content"]:
                if isinstance(page, dict) and "sections" in page:
                    for sec in page["sections"]:
                        if "heading" in sec:
                            heading_text = sec["heading"]
                            # ignore placeholder headings like "Untitled Section" or empty
                            if heading_text and heading_text.strip() and heading_text.lower() not in (
                            "untitled section", "abstract"):
                                headings.append(heading_text.strip())
        # If the doc might have metadata title (which could be considered heading level 0)
        if "metadata" in doc:
            title = doc["metadata"].get("title")
            if title:
                headings.insert(0, title.strip())
        # If no explicit headings found, we could try pattern: lines that are all caps or have numbering at start.
        # (omitted for brevity)
        return headings


class FigureEvaluator:
    def __init__(self):
        self.name = "figures"

    def evaluate(self, docA: Dict, docB: Dict) -> Dict:
        """Compare figure (image) detection and caption text."""
        figsA = self._extract_figures(docA)
        figsB = self._extract_figures(docB)
        countA = len(figsA)
        countB = len(figsB)
        # Match figures by caption text (if available)
        matched = 0
        usedB = set()
        for i, figA in enumerate(figsA):
            captionA = figA.get("caption", "").strip().lower()
            if not captionA:
                continue
            # find a figure in B with a very similar caption (exact match in lowercase for simplicity)
            for j, figB in enumerate(figsB):
                if j in usedB:
                    continue
                captionB = figB.get("caption", "").strip().lower()
                if captionB == captionA and captionB != "":
                    matched += 1
                    usedB.add(j)
                    break
        # Also check captions present in text:
        captionsA = [fig["caption"] for fig in figsA if "caption" in fig]
        captionsB = [fig["caption"] for fig in figsB if "caption" in fig]
        # If pipeline didn't explicitly mark figures, we find captions in plain text
        if not captionsA:
            captionsA = self._find_captions_in_text(docA)
        if not captionsB:
            captionsB = self._find_captions_in_text(docB)
        # Caption recall: how many captions from A appear in B text
        caption_match_count = 0
        textB_full = "\n".join(captionsB) if isinstance(captionsB, list) else str(captionsB)
        for cap in captionsA:
            if cap and cap in textB_full:
                caption_match_count += 1
        caption_recall = caption_match_count / len(captionsA) if captionsA else 1.0

        result = {
            "structure": {  # we can output under structure or separate "figures"
                "figures_count": {
                    "A": countA,
                    "B": countB
                },
                "figures_matched": matched,
                "caption_recall": caption_recall
            }
        }
        return result

    def _extract_figures(self, doc: Dict) -> List[Dict]:
        """Extract figure info (captions or image markers) from doc JSON."""
        figures = []
        # If doc has an 'images' or 'figures' list (like pipeline A does per page)
        if "content" in doc:
            for page in doc["content"]:
                if isinstance(page, dict):
                    # Some outputs might list images separate from text sections
                    if "images" in page:
                        for img in page["images"]:
                            # img might have 'file' and possibly no caption text
                            fig = {"page": img.get("page_number")}
                            # If pipeline stored caption in sections text (common), we need to capture that from text
                            # We assume caption text appears in the page's sections text somewhere labeled e.g. "Fig. X"
                            # We'll handle caption detection separately if not explicitly linked.
                            figures.append(fig)
        # Also search for lines starting with "Fig" or "Figure" in the text as potential captions
        captions = self._find_captions_in_text(doc)
        for cap in captions:
            figures.append({"caption": cap})
        return figures

    def _find_captions_in_text(self, doc: Dict) -> List[str]:
        """Find lines in the document text that look like figure captions (start with Figure/Fig)."""
        full_text = ""
        if "content" in doc:
            for page in doc["content"]:
                if isinstance(page, dict):
                    if "sections" in page:
                        for sec in page["sections"]:
                            if "text" in sec:
                                full_text += sec["text"] + "\n"
                    elif "text" in page:
                        full_text += page["text"] + "\n"
        elif "text" in doc:
            full_text = doc["text"]
        captions = []
        for line in full_text.splitlines():
            line_strip = line.strip()
            if line_strip.lower().startswith("fig") or line_strip.lower().startswith("figure"):
                captions.append(line_strip)
        return captions


class MetadataEvaluator:
    def __init__(self):
        self.name = "metadata"

    def evaluate(self, docA: Dict, docB: Dict) -> Dict:
        """Compare metadata fields (title, authors, date) between two docs."""
        metaA = docA.get("metadata", {})
        metaB = docB.get("metadata", {})
        titleA = (metaA.get("title") or "").strip()
        titleB = (metaB.get("title") or "").strip()
        authorA = (metaA.get("author") or "").strip()
        authorB = (metaB.get("author") or "").strip()
        dateA = metaA.get("creationDate") or metaA.get("modDate") or ""
        dateB = metaB.get("creationDate") or metaB.get("modDate") or ""
        # Simple string comparison for title and author
        title_match = (titleA.lower() == titleB.lower()) if titleA and titleB else (titleA == titleB)
        # If one is empty and the other not, it's a mismatch.
        authors_match = None
        authors_overlap = None
        if authorA or authorB:
            # Compare author lists by splitting by comma or semicolon (assuming a string of names)
            listA = [a.strip() for a in re.split(r'[;,]', authorA) if a.strip()] if authorA else []
            listB = [a.strip() for a in re.split(r'[;,]', authorB) if a.strip()] if authorB else []
            if listA and listB:
                setA = set([a.lower() for a in listA])
                setB = set([b.lower() for b in listB])
                common = setA & setB
                # authors_match True if all names match (or perhaps if one is subset of the other)
                authors_match = (setA == setB)
                authors_overlap = len(common) / max(len(setA), len(setB))
            else:
                authors_match = (listA == listB)
                authors_overlap = 1.0 if authors_match else 0.0
        # Date match: compare year or full string
        date_match = False
        if dateA and dateB:
            # date strings might be in format "D:YYYYMMDD..." from PDF metadata.
            # We'll compare year and month if possible.
            yearA = dateA[2:6] if dateA.startswith("D:") else dateA[:4]
            yearB = dateB[2:6] if dateB.startswith("D:") else dateB[:4]
            date_match = (yearA == yearB)
        elif not dateA and not dateB:
            date_match = True  # neither has date (treat as not disagreeing)

        result = {
            "metadata": {
                "title_A": titleA,
                "title_B": titleB,
                "title_match": bool(title_match),
                "author_A": authorA,
                "author_B": authorB,
                "authors_match": bool(authors_match) if authors_match is not None else None,
                "authors_overlap": authors_overlap,
                "date_A": dateA,
                "date_B": dateB,
                "date_match": bool(date_match)
            }
        }
        return result


class OrderCoherenceEvaluator:
    def __init__(self):
        self.name = "order_coherence"

    def evaluate(self, docA: Dict, docB: Dict) -> Dict:
        """Evaluate reading order coherence within and between outputs (builds on content evaluator results)."""
        # This uses results from ContentEvaluator (like order_preservation and chunk stats) if already computed.
        # In practice, we might integrate this in ContentEvaluator or compute again.
        # Here, assume we can recompute needed pieces quickly.
        textA = ""
        textB = ""
        if "content" in docA:
            for page in docA["content"]:
                if isinstance(page, dict):
                    if "sections" in page:
                        for sec in page["sections"]:
                            textA += sec.get("text", "") + " "
                    elif "text" in page:
                        textA += page["text"] + " "
        if "content" in docB:
            for page in docB["content"]:
                if isinstance(page, dict):
                    if "sections" in page:
                        for sec in page["sections"]:
                            textB += sec.get("text", "") + " "
                    elif "text" in page:
                        textB += page["text"] + " "
        tokensA = tokenize(textA)
        tokensB = tokenize(textB)

        # Chunk end at sentence calculation for each pipeline:
        def chunk_stats(doc: Dict) -> Dict:
            chunks = []
            if "content" in doc:
                for page in doc["content"]:
                    if isinstance(page, dict) and "sections" in page:
                        for sec in page["sections"]:
                            if "chunks" in sec:
                                for ch in sec["chunks"]:
                                    chunks.append(ch)
            # Fallback: if chunks not directly given, use paragraphs (sections text) as chunks
            if not chunks and "content" in doc:
                for page in doc["content"]:
                    if isinstance(page, dict) and "sections" in page:
                        for sec in page["sections"]:
                            if "text" in sec:
                                chunks.append(sec["text"])
            if not chunks:
                # if still empty, perhaps treat whole text as one chunk
                full_text = doc.get("text", "")
                if full_text:
                    chunks = [full_text]
            # Now analyze chunks:
            import nltk  # using nltk to detect sentence end if available (assuming it's installed)
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
            total_chunks = len(chunks)
            chunks_end_sentence = 0
            sentences_split = 0
            total_sentences = 0
            for ch in chunks:
                sents = sent_tokenize(ch)
                if len(sents) > 0:
                    total_sentences += len(sents)
                    # if the chunk text ends with a punctuation that indicates sentence end
                    if ch.strip()[-1] in '.!?':
                        chunks_end_sentence += 1
                    # if chunk contains more than one sentence, it means multiple sentences in one chunk (which is fine),
                    # but if chunk has partial sentence, we might detect it if chunk doesn't end with sentence end.
                else:
                    # if chunk is very short or no punctuation, consider it one sentence for count
                    total_sentences += 1
            # sentences_split: how many sentences are spread across chunk boundaries?
            # For simplicity: total_sentences - chunks_end_sentence will estimate sentences that didn't end at boundary
            # Actually, if a sentence was split, it means one sentence spanned two chunks, which increments this count.
            sentences_split = max(0, total_sentences - chunks_end_sentence)
            avg_tokens = 0
            std_tokens = 0
            max_tokens = 0
            token_counts = []
            for ch in chunks:
                token_counts.append(len(tokenize(ch)))
            if token_counts:
                avg_tokens = sum(token_counts) / len(token_counts)
                if len(token_counts) > 1:
                    mean = avg_tokens
                    variance = sum((x - mean) ** 2 for x in token_counts) / len(token_counts)
                    std_tokens = math.sqrt(variance)
                max_tokens = max(token_counts)
            return {
                "pct_chunks_end_at_sentence": (chunks_end_sentence / total_chunks * 100.0) if total_chunks > 0 else 0.0,
                "pct_sentences_split": (sentences_split / total_sentences * 100.0) if total_sentences > 0 else 0.0,
                "length_stats": {
                    "mean_tokens": avg_tokens,
                    "std_tokens": std_tokens,
                    "max_tokens": max_tokens
                }
            }

        statsA = chunk_stats(docA)
        statsB = chunk_stats(docB)
        # Order preservation between the two (recompute or use previous calculation)
        matcher = difflib.SequenceMatcher(a=tokensA, b=tokensB)
        matched_tokens = sum(triple.size for triple in matcher.get_matching_blocks())
        common_tokens = sum((Counter(tokensA) & Counter(tokensB)).values())
        order_preservation = (matched_tokens / common_tokens) if common_tokens > 0 else 1.0

        result = {
            "chunk_coherence": {
                "pipelineA": statsA,
                "pipelineB": statsB,
                "order_preservation": order_preservation
            }
        }
        return result


# Main evaluator that ties all modules together
class PDFComparisonEvaluator:
    def __init__(self):
        self.modules = [
            ContentEvaluator(),
            TableEvaluator(),
            HeadingEvaluator(),
            FigureEvaluator(),
            MetadataEvaluator(),
            OrderCoherenceEvaluator()
        ]

    def compare_documents(self, docA: Dict, docB: Dict) -> Dict:
        """Compare two JSON document outputs and return combined evaluation results."""
        result = {"document": docA.get("document_title") or docA.get("metadata", {}).get("source", "")}
        # Run each module
        for module in self.modules:
            res = module.evaluate(docA, docB)
            # Merge results dict
            # If there's a key overlap (like 'structure' used by multiple modules), merge sub-keys
            for key, val in res.items():
                if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                    result[key].update(val)
                else:
                    result[key] = val
        return result

    def compare_batch(self, docsA: Dict[str, Dict], docsB: Dict[str, Dict]) -> List[Dict]:
        """
        Compare multiple documents. docsA and docsB are dicts mapping document ID to JSON data.
        Returns list of result dicts for each document and also writes summary CSV/JSON.
        """
        results = []
        for doc_id, docA in docsA.items():
            if doc_id not in docsB:
                continue  # skip if not in both
            docB = docsB[doc_id]
            res = self.compare_documents(docA, docB)
            results.append(res)
        # After all, output aggregate CSV
        self._save_summary_csv(results, "comparison_summary.csv")
        return results

    def _save_summary_csv(self, results: List[Dict], csv_path: str):
        if not results:
            return
        # Define CSV columns of interest
        # We'll pick some representative metrics to include
        columns = [
            "document",
            "precision_A", "recall_A", "F1_A", "BLEU4_A",
            "precision_B", "recall_B", "F1_B", "BLEU4_B",
            "headings_count_A", "headings_count_B", "headings_percent",
            "figures_count_A", "figures_count_B", "caption_recall",
            "tables_count_A", "tables_count_B", "tables_matched",
            "order_preservation"
        ]
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for res in results:
                # Extract values safely from nested structure
                precision_B = res.get("content_fidelity", {}).get("precision", 0)
                recall_B = res.get("content_fidelity", {}).get("recall", 0)
                f1_B = res.get("content_fidelity", {}).get("F1", 0)
                bleu_B = res.get("content_fidelity", {}).get("BLEU-4", 0)
                precision_A = res.get("content_fidelity_A_vs_B", {}).get("precision", 0)
                recall_A = res.get("content_fidelity_A_vs_B", {}).get("recall", 0)
                f1_A = res.get("content_fidelity_A_vs_B", {}).get("F1", 0)
                bleu_A = res.get("content_fidelity_A_vs_B", {}).get("BLEU-4", 0)
                headings_count_A = res.get("structure", {}).get("headings_count", {}).get("A", 0)
                headings_count_B = res.get("structure", {}).get("headings_count", {}).get("B", 0)
                headings_pct = res.get("structure", {}).get("headings_percent", 0)
                figures_count_A = res.get("structure", {}).get("figures_count", {}).get("A", 0)
                figures_count_B = res.get("structure", {}).get("figures_count", {}).get("B", 0)
                caption_recall = res.get("structure", {}).get("caption_recall", 0)
                tables_count_A = res.get("table_fidelity", {}).get("tables_count_A", 0)
                tables_count_B = res.get("table_fidelity", {}).get("tables_count_B", 0)
                tables_matched = res.get("table_fidelity", {}).get("tables_matched", 0)
                order_pres = res.get("chunk_coherence", {}).get("order_preservation",
                                                                res.get("content_fidelity", {}).get(
                                                                    "order_preservation", 1.0))
                row = [
                    res.get("document", ""),
                    precision_A, recall_A, f1_A, bleu_A,
                    precision_B, recall_B, f1_B, bleu_B,
                    headings_count_A, headings_count_B, headings_pct,
                    figures_count_A, figures_count_B, caption_recall,
                    tables_count_A, tables_count_B, tables_matched,
                    order_pres
                ]
                writer.writerow(row)

    def generate_html_diff(self, docA: Dict, docB: Dict, output_path: str):
        """Generate an HTML file highlighting differences between docA and docB text."""
        textA = []
        textB = []
        # We'll use sections or paragraphs as lines for diff context
        if "content" in docA:
            for page in docA["content"]:
                if isinstance(page, dict) and "sections" in page:
                    for sec in page["sections"]:
                        if sec.get("text"):
                            textA.append(sec["text"])
        if "content" in docB:
            for page in docB["content"]:
                if isinstance(page, dict) and "sections" in page:
                    for sec in page["sections"]:
                        if sec.get("text"):
                            textB.append(sec["text"])
        # Use difflib.HtmlDiff to generate side-by-side diff
        diff = difflib.HtmlDiff(tabsize=2, wrapcolumn=80)
        html = diff.make_file(textA, textB, fromdesc='Pipeline A', todesc='Pipeline B', context=True, numlines=3)
        with open(output_path, 'w') as f:
            f.write(html)

# Example usage:
# Assume we have two dicts of documents, or lists of file paths to their JSON outputs.
# docsA = {"doc1": json.load(open("haonan_output/doc1.json")), ...}
# docsB = {"doc1": json.load(open("james_output/doc1.json")), ...}
# evaluator = PDFComparisonEvaluator()
# results = evaluator.compare_batch(docsA, docsB)
# # results now holds per-doc dictionaries of metrics, and "comparison_summary.csv" is written.
# # We can also generate an HTML diff for a particular doc if needed:
# evaluator.generate_html_diff(docsA["doc1"], docsB["doc1"], "doc1_diff.html")
