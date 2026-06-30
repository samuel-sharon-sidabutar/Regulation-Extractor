import os
import glob
import time
import shutil
import sys
import json
from datetime import datetime
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from markitdown import MarkItDown
from openai import OpenAI

CONFIG = {
    "MODEL": "deepseek",  # Switch to "minimax" to change model
    "WORK_DIR": "regulations",
    "FAILED_DIR": os.path.join("regulations", "failed_files"),
    "CONTINUOUS_EXECUTION": True,
    "FORCED_OVERWRITE": True,
    "ROUTER_IP": "https://api.tokenrouter.com/v1",
    "ROUTER_KEY": os.environ.get("TOKENROUTER_API_KEY"),
    "API_RETRY_MAX": 4,
    "API_RETRY_BASE_DELAY_SEC": 30,
    "MAX_LOOPS": 30,
    "LOG_DIR": "",
    "MODEL_ID": ""
}

MODEL_PRESETS = {
    "deepseek": { # The model this script was optimized on
        "LOG_DIR": os.path.join("regulations", "deepseek_logs"),
        "MODEL_ID": "deepseek/deepseek-v4-pro",
        "TEMPERATURE": 0.1,
    },
    "minimax": {
        "LOG_DIR": os.path.join("regulations", "minimax_logs"),
        "MODEL_ID": "MiniMax-M3",
        "TEMPERATURE": 0.1,
    },
}

def resolve_model_config():
    """Apply the active MODEL preset into CONFIG so everything downstream uses CONFIG directly."""
    preset = CONFIG["MODEL"]
    if preset not in MODEL_PRESETS:
        print(f"[CRITICAL ERROR] Unknown MODEL '{preset}'. Valid options: {list(MODEL_PRESETS.keys())}")
        sys.exit(1)
    CONFIG.update(MODEL_PRESETS[preset])

# Used to structure data integrity errors dynamically
MalformedRow = namedtuple("MalformedRow", ["row_num", "raw_fragment"])
DocumentMetadata = namedtuple("DocumentMetadata", ["regulasi", "tipe", "tanggal"])

class APIQuotaExhaustedError(Exception):
    """Raised when the API retry loop exhausts all attempts due to quota/credit limits."""

# PROMPTS
METADATA_PROMPT = """Act as an expert data extractor. Look at the provided Indonesian banking regulation (POJK).
Extract EXACTLY ONE line of text containing these 3 values separated by pipes (|). Do not include a header row.

Format: Regulasi|Tipe|Tanggal

1. Regulasi: The full title of the regulation document word-for-word.
2. Tipe: "POJK" (or based on the document type).
3. Tanggal: The date the regulation was enacted (Diundangkan/Ditetapkan).

Output only the data row. Do not include any conversational preamble.
"""

MASTER_PROMPT = """Act as an expert data extractor and compliance analyst. Your task is to extract hierarchical regulation data from Indonesian banking regulations (POJK) and output it strictly as a JSON array of flat objects.

To minimize token usage, you must use minified keys. Output exactly ONE valid JSON object per line. Do not pretty-print with extra indentation.

**MINIFIED JSON SCHEMA (7 KEYS):**
Each line must be a single flat object with these exact keys:
* `sec`: Bagian (Section) -> Evaluate the semantic intent of the article dynamically. See STRICT CATEGORIZATION RULES.
* `ref_item`: Item Ref -> The parent reference. If the article contains numbered paragraphs (Ayat), you MUST include the Ayat number here (e.g., "Pasal 2 ayat 1"). ONLY use "Pasal 2" if the article has no numbered paragraphs. NEVER include "huruf" or "angka" tags here.
* `item`: Item -> Main text of the Item. Strip numbering prefixes.
* `desc`: Item Description -> Explanation scoped strictly to this specific Item Ref (Pasal or Ayat) from the "PENJELASAN" section. See PENJELASAN SCOPING RULE and ANTI-LAZINESS RULE.
* `k_sub`: Kode Sub Item -> Incremental integer (1, 2, 3...) for sub-items under this parent. Use null if none.
* `ref_sub`: Sub Item Ref -> Letter or number of the list item (e.g., "a", "b", "1"). Use "" if none.
* `sub`: Sub Item -> Actual text of the sub-item. Strip its prefix. See 3-LEVEL NESTING RULE.

**INDONESIAN LEGAL HIERARCHY RULES:**
* **Pasal** (Article): Top-level grouping.
* **Ayat** (Paragraph): Numbered lists inside a Pasal, e.g., (1), (2). An 'Ayat' is ALWAYS an Item. 
* **List Elements** (Huruf/Angka): Lists inside an Ayat or Pasal (a, b, c or 1, 2, 3). These are ALWAYS Sub-items.

**STRICT 3-LEVEL NESTING RULE (CRITICAL):**
Your output schema ONLY supports 2 levels (Item and Sub-item). If the document goes 3 levels deep (e.g., Pasal 46 ayat 3 -> huruf a -> angka 1, 2, 3), you MUST flatten the 3rd level into the 2nd level.
* NEVER promote a "huruf" to an Item. `ref_item` MUST NEVER contain the word "huruf" or "angka".
* Keep the Ayat as the `ref_item`. Keep the huruf as the `ref_sub`.
* Concatenate the 3rd-level list directly into the `sub` string. 
* Example for Pasal 46 ayat 3 huruf a: `ref_item` is "Pasal 46 ayat 3", `ref_sub` is "a", and `sub` is "penerapan prinsip kehati-hatian berupa: 1. penilaian... 2. pemenuhan... 3. batas..."

**PENJELASAN SCOPING RULE (CRITICAL):**
The `desc` field must contain ONLY the explanation for the specific Item (Pasal or Ayat) being extracted. The Penjelasan section explains each part separately — you must match them one-to-one.
* Find the explanation for the exact Pasal or Ayat in the Penjelasan section and use only that text.
* NEVER concatenate explanations from multiple Ayat into one `desc` field.
* Strip the Ayat header label (e.g., "Ayat (1)") from the start of the extracted text — output only the explanation body.
* If multiple rows share the same `ref_item` (e.g., huruf a, b, c under the same Ayat), repeat the same scoped `desc` on each row.

BAD EXAMPLE (dumps entire Pasal Penjelasan into one Ayat):
{"ref_item": "Pasal 6 ayat 1", "desc": "Ayat (1) Modal disetor bagi BPR... Ayat (2) Modal disetor bagi BPR Syariah... Ayat (6) Cukup jelas."}

GOOD EXAMPLE (scoped to the correct Ayat, header stripped):
{"ref_item": "Pasal 6 ayat 1", "desc": "Modal disetor bagi BPR berbentuk badan hukum Koperasi yaitu simpanan pokok dan simpanan wajib sesuai dengan Undang-Undang mengenai perkoperasian."}

**ANTI-LAZINESS RULE FOR "PENJELASAN" (CRITICAL):**
You must extract the ENTIRE verbatim text of the matching Penjelasan for the `desc` field, even if it is very long or contains lists. ONLY output "Cukup jelas." if the Penjelasan for that specific Pasal/Ayat literally contains only those two words.

BAD EXAMPLE (lazy):
{"desc": "Cukup jelas."} // When the Penjelasan for that Ayat actually has a paragraph of explanation.

GOOD EXAMPLE:
{"desc": "Yang dimaksud dengan 'kualitas aset' adalah penilaian terhadap kondisi keuangan debitur dan prospek usaha..."}

**STRICT CATEGORIZATION RULES (`sec` field):**
Do not just blindly copy the Chapter number (e.g., "Bab III"). You must evaluate the semantic content of the specific Pasal. Overwrite the `sec` value EXACTLY as follows:
* **"Ketentuan"**: If the article defines terms (usually Pasal 1).
* **"Sanksi"**: If the article dictates administrative penalties, fines, or "Sanksi".
* **"Regulasi Lain"**: If the article is about "Ketentuan Peralihan" or "Ketentuan Penutup".
* For all standard operational rules, keep the original Chapter title (e.g., "Bab II").

**FLATTENING LISTS REQUIREMENT:**
You must separate parent text from sub-item text. If an Item has multiple Sub-items, output a distinct JSON object line for EACH Sub-item. Repeat the parent keys (`sec`, `ref_item`, `item`, `desc`) identically on every row.

**OUTPUT FORMAT:**
Start your response with `[` on its own line. Output each object on its own line followed by a comma. When completely finished, output `[END_OF_DOCUMENT]` on a new line.
"""

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def setup_environment():
    resolve_model_config()
    
    if not CONFIG["ROUTER_KEY"]:
        print("\n[CRITICAL ERROR] API key environment variable is missing. Exiting...")
        sys.exit(1)
        
    for folder in [CONFIG["WORK_DIR"], CONFIG["FAILED_DIR"], CONFIG["LOG_DIR"]]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(CONFIG["LOG_DIR"], f"run_{TIMESTAMP}.txt")

    if not glob.glob(os.path.join(CONFIG["WORK_DIR"], "*.pdf")) and not glob.glob(os.path.join(CONFIG["WORK_DIR"], "*.md")):
        print(f"Directory '{CONFIG['WORK_DIR']}' created. Drop your PDFs/MDs in there and run again!")
        sys.exit(0)

    return log_file

def append_to_log(log_file, text_block):
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(text_block)

def prompt_overwrite(filepath):
    if CONFIG["FORCED_OVERWRITE"]:
        if os.path.exists(filepath):
            print(f"  [!] Overwriting: '{os.path.basename(filepath)}'")
        return 'y'

    if not os.path.exists(filepath):
        return 'y'

    while True:
        choice = input(f"  [!] '{os.path.basename(filepath)}' exists. Overwrite? (y/n / c): ").strip().lower()
        if choice in ['y', 'n', 'c']:
            return choice

def is_document_ended(text):
    signals = [
        "[END_OF_DOCUMENT]", "[END_OF_DOCUMENT", "END_OF_DOCUMENT",
        "[END_OF_DOC", "END_OF_DOC", "[END_OF", "[END"
    ]
    return any(signal in text.upper() for signal in signals)

def format_time(seconds):
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"

# ==========================================
# DATA PROCESSING & SANITIZATION (JSON -> CSV)
# ==========================================
def convert_json_stream_to_csv(raw_json_stream, doc_meta):
    lines = raw_json_stream.split('\n')
    cleaned_lines = []
    malformed_errors = []
    data_row_count = 0

    global_item_counter = 0
    last_seen_ref = ""

    db_header = "Regulasi|Tipe|Tanggal|Status|Properti Regulasi|Bagian (Section)|Kode Item|Item Ref|Item|Item Description|Property|Kode Sub Item|Sub Item Ref|Sub Item"
    cleaned_lines.append(db_header)

    for i, line in enumerate(lines):
        cleaned_line = line.strip().rstrip(',')
        
        if not cleaned_line or cleaned_line in ['[', ']', '[END_OF_DOCUMENT]'] or is_document_ended(cleaned_line):
            continue

        try:
            obj = json.loads(cleaned_line)
            current_ref = str(obj.get("ref_item", "")).strip()
            
            if current_ref and current_ref != last_seen_ref:
                global_item_counter += 1
                last_seen_ref = current_ref
                
            final_cols = [
                doc_meta.regulasi, doc_meta.tipe, doc_meta.tanggal, "", "", 
                str(obj.get("sec", "")), 
                str(global_item_counter) if global_item_counter > 0 else "", 
                current_ref, str(obj.get("item", "")), str(obj.get("desc", "")), "", 
                str(obj.get("k_sub", "")) if obj.get("k_sub") is not None else "", 
                str(obj.get("ref_sub", "")), str(obj.get("sub", ""))
            ]
            
            final_cols = [col.replace('|', ' ').replace('\n', ' ').replace('\r', ' ') for col in final_cols]
            cleaned_lines.append("|".join(final_cols))
            data_row_count += 1
            
        except Exception:
            malformed_errors.append(MalformedRow(row_num=i + 1, raw_fragment=cleaned_line[:50]))
            cleaned_lines.append(f"{doc_meta.regulasi}|{doc_meta.tipe}|{doc_meta.tanggal}|||MALFORMED_JSON_LINE||||||||{cleaned_line}")

    return "\n".join(cleaned_lines), data_row_count, malformed_errors

def print_summary(csv_path, malformed_errors):
    if not os.path.exists(csv_path): return 0
        
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) <= 1: return 0

    data_lines = [l for l in lines if not l.startswith("Regulasi|")]
    data_row_count = len(data_lines)

    title = "Unknown (Check File)"
    sections = set()

    for row in data_lines:
        cols = row.split('|')
        if len(cols) == 14:
            if title == "Unknown (Check File)" and cols[0]: title = cols[0] 
            section_val = cols[5].strip()
            if section_val: sections.add(section_val)

    print("\n  ========================================================")
    print(f"  📊 EXTRACTION SUMMARY: {os.path.basename(csv_path)}")
    print("  ========================================================")
    print(f"  Title      : {title[:60]}...")
    print(f"  Total Rows : {data_row_count} data rows generated")
    
    # Restored: Unique section printouts
    section_list = sorted([s for s in list(sections) if s])
    display_sections = ", ".join(section_list) if section_list else "None found"
    print(f"  Sections   : {display_sections}")
    
    if malformed_errors:
        print("  --------------------------------------------------------")
        print(f"  🚨 DATA INTEGRITY WARNING: {len(malformed_errors)} JSON Lines Corrupted")
        for err in malformed_errors:
            print(f"      - Line {err.row_num}: {err.raw_fragment}...")
    print("  ========================================================\n")
    return data_row_count

# ==========================================
# EXTRACTION HELPERS
# ==========================================
def parse_metadata(meta_raw):
    """Hardened metadata parser that strips preambles and finds the first valid data row."""
    for line in meta_raw.replace('`', '').split('\n'):
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 3 and parts[0]:
            return DocumentMetadata(parts[0], parts[1], parts[2])
    return DocumentMetadata("Unknown", "POJK", "Unknown")

def is_complete_json_object(line):
    stripped = line.strip().rstrip(',')
    if not stripped or stripped in ['[', ']']:
        return True
    try:
        obj = json.loads(stripped)
        return isinstance(obj, dict)
    except (json.JSONDecodeError, ValueError):
        return False

def find_first_malformed(json_stream, start_from=0):
    """Return the index of the first line that isn't valid JSON, or -1.
    start_from skips already-validated lines for efficiency."""
    lines = json_stream.split('\n')
    for i in range(start_from, len(lines)):
        line = lines[i]
        cleaned = line.strip().rstrip(',')
        if not cleaned or cleaned in ['[', ']'] or is_document_ended(cleaned):
            continue
        try:
            json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return i
    return -1

def handle_cycle_save(full_json_text, csv_path, doc_meta, log_file):
    lines_in_buffer = full_json_text.split('\n')
    last_char = full_json_text[-1] if full_json_text else ""

    safe_json_to_save = full_json_text
    line_status = "Clean cut (Ends with newline)"

    # Only inspect the last line when it might be a truncated mid-stream fragment
    if not (last_char == '\n' or not lines_in_buffer or not full_json_text.strip()):
        candidate = lines_in_buffer[-1]
        if is_complete_json_object(candidate):
            line_status = "Clean cut (Last line is complete JSON)"
        else:
            lines_in_buffer.pop()
            safe_json_to_save = '\n'.join(lines_in_buffer) + '\n'
            line_status = f"Cut off mid-object (Deleted: '{candidate[:20]}...')"

    current_row_count = len([l for l in lines_in_buffer if l.strip() and l.strip() not in ['[', ']', ',']])
    partial_csv, _, _ = convert_json_stream_to_csv(safe_json_to_save, doc_meta)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(partial_csv)
    return safe_json_to_save, line_status, current_row_count

def build_continuation_history(user_string, full_json_text, error_nudge=False):
    """Builds the continuation prompt using an expanded 120-line context window."""
    raw_lines = full_json_text.split('\n')
    last_context = '\n'.join(raw_lines[-120:])

    resume_msg = "Resume generating the JSON array exactly where you left off. Output ONLY raw minified JSON objects, one per line. Do not wrap in new brackets or rewrite previous records."
    if error_nudge:
        resume_msg += " CRITICAL: Your previous generation contained invalid/malformed JSON. Ensure all quotes are escaped properly and output strictly valid JSON."

    return [
        {"role": "system", "content": MASTER_PROMPT},
        {"role": "user", "content": user_string},
        {"role": "assistant", "content": last_context},
        {"role": "user", "content": resume_msg}
    ]

# ==========================================
# CORE EXTRACTION LOGIC
# ==========================================
def convert_pdf_to_md(pdf_path, md_path, quiet=False):
    if not quiet: print(f"  -> Converting {os.path.basename(pdf_path)}...")
    md_converter = MarkItDown()
    result = md_converter.convert(pdf_path)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result.text_content)
    if not quiet: print(f"  [OK] Saved to {md_path}")
    return result.text_content

def call_openai_with_retry(client, model_id, conversation_history):
    max_attempts = CONFIG["API_RETRY_MAX"]
    base_delay = CONFIG["API_RETRY_BASE_DELAY_SEC"]
    attempt = 1
    
    while attempt <= max_attempts:
        try:
            return client.chat.completions.create(
                model=model_id,
                messages=conversation_history,
                temperature=CONFIG["TEMPERATURE"],
                stream=True,
                extra_body={"thinking": {"type": "disabled"}}
            )
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "credit limit" in error_msg.lower():
                if attempt >= max_attempts:
                    raise APIQuotaExhaustedError(f"Max attempts ({max_attempts}) exceeded for API limit.")
                
                delay = base_delay * (2 ** (attempt - 1))
                print(f"\n  [!] API Quota Limit Hit (403). Pausing for {delay}s (exponential backoff)... (Attempt {attempt}/{max_attempts})")
                print(f"  [!] {error_msg}")
                time.sleep(delay)
                attempt += 1
            else:
                raise e

def extract_metadata(client, md_content, log_file):
    print("  -> Extracting document metadata...")
    meta_history = [
        {"role": "system", "content": METADATA_PROMPT},
        {"role": "user", "content": f"Here is the document:\n\n{md_content}"}
    ]
    meta_resp = call_openai_with_retry(client, CONFIG["MODEL_ID"], meta_history)
    
    meta_raw = "".join([chunk.choices[0].delta.content or "" for chunk in meta_resp if chunk.choices])
    append_to_log(log_file, f"--- METADATA RESULT ---\n{meta_raw.strip()}\n-----------------------\n\n")

    meta_regulasi = parse_metadata(meta_raw)
    if meta_regulasi.regulasi == "Unknown": 
        print("  [!] Failed to cleanly parse metadata. Using fallbacks.")
        
    return meta_regulasi

def extract_main_content(client, md_content, csv_path, doc_meta, log_file):
    user_string = f"Here is the document:\n\n{md_content}\n\n[BEGIN EXTRACTION NOW]"
    conversation_history = [
        {"role": "system", "content": MASTER_PROMPT},
        {"role": "user", "content": user_string},
        {"role": "assistant", "content": "[\n"}
    ]

    max_loops = CONFIG["MAX_LOOPS"]
    loop_count = 1
    abort_batch = False
    full_json_text = "[\n"
    just_pruned_error = False
    malformed_retries = {}
    max_prune_retries = CONFIG["API_RETRY_MAX"]
    last_checked_line = 0

    print(f"  -> Firing main extraction request (Cycle {loop_count}/{max_loops})...")

    while loop_count <= max_loops:
        response = call_openai_with_retry(client, CONFIG["MODEL_ID"], conversation_history)

        new_chunk = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in response
            if chunk.choices and len(chunk.choices) > 0
        )
        full_json_text += new_chunk
        append_to_log(log_file, f"--- CYCLE {loop_count} OUTPUT ---\n{new_chunk}\n")

        if is_document_ended(full_json_text):
            break

        full_json_text, line_status, current_row_count = handle_cycle_save(
            full_json_text, csv_path, doc_meta, log_file)

        full_json_text, just_pruned_error, last_checked_line = prune_malformed_rows(
            full_json_text, last_checked_line, malformed_retries, max_prune_retries,
            csv_path, doc_meta, loop_count, log_file)

        if not CONFIG["CONTINUOUS_EXECUTION"]:
            print(f"\n  [?] AI paused at object count: {current_row_count} ({line_status}). Partial saved.")
            if input("      Continue processing THIS file? (y/n): ").strip().lower() != 'y':
                if input("      Skip REST of batch? (y/n): ").strip().lower() == 'y':
                    abort_batch = True
                break
        else:
            print(f"  -> AI paused at object count: {current_row_count} ({line_status}). Auto-continuing...")

        loop_count += 1
        print(f"  -> Continuing generation... (Cycle {loop_count}/{max_loops})")
        conversation_history = build_continuation_history(user_string, full_json_text, error_nudge=just_pruned_error)
        just_pruned_error = False

    if loop_count > max_loops:
        print("  [!] Safety limit reached: Max loops exceeded.")

    final_csv_to_save, final_row_count, malformed_errors = convert_json_stream_to_csv(
        full_json_text, doc_meta)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(final_csv_to_save)

    print(f"  [OK] Final stitched file compiled and saved to {csv_path}")
    print_summary(csv_path, malformed_errors)

    deduped_count = deduplicate_csv(csv_path)
    if deduped_count != final_row_count:
        print(f"  [OK] Deduplicated {final_row_count - deduped_count} duplicate row(s) from {csv_path}")

    return deduped_count, abort_batch, malformed_errors, loop_count


def prune_malformed_rows(full_json_text, last_checked_line, malformed_retries,
                         max_prune_retries, csv_path, doc_meta, loop_count, log_file):
    bad_idx = find_first_malformed(full_json_text, start_from=last_checked_line)
    if bad_idx == -1:
        return full_json_text, False, len(full_json_text.split('\n'))

    malformed_retries[bad_idx] = malformed_retries.get(bad_idx, 0) + 1
    if malformed_retries[bad_idx] > max_prune_retries:
        print(f"  [!] Malformed JSON at line ~{bad_idx} exceeded {max_prune_retries} retries. Accepting malformed rows.")
        append_to_log(log_file, f"--- MALFORMED ROW EXCEEDED RETRIES (cycle {loop_count}) at data line index {bad_idx} ---\n")
        return full_json_text, False, len(full_json_text.split('\n'))

    print(f"  [!] Pruned malformed JSON at line ~{bad_idx} (retry {malformed_retries[bad_idx]}/{max_prune_retries}). Regenerating from that point.")
    append_to_log(log_file, f"--- MALFORMED ROW PRUNE (cycle {loop_count}, retry {malformed_retries[bad_idx]}) at data line index {bad_idx} ---\n")
    full_json_text = '\n'.join(full_json_text.split('\n')[:bad_idx]) + '\n'
    partial_csv, _, _ = convert_json_stream_to_csv(full_json_text, doc_meta)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(partial_csv)
    return full_json_text, True, len(full_json_text.split('\n'))

def process_single_file(md_path, csv_path, log_file, md_content=None):
    filename_base = os.path.basename(md_path)
    print(f"  -> Processing {filename_base}...")
    
    if md_content is None:
        with open(md_path, "r", encoding="utf-8") as f: md_content = f.read()

    client = OpenAI(base_url=CONFIG["ROUTER_IP"], api_key=CONFIG["ROUTER_KEY"])
    append_to_log(log_file, f"\n\n{'='*80}\n=== START AI LOG | File: {filename_base} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n{'='*80}\n")

    doc_meta = extract_metadata(client, md_content, log_file)
    print(f"  [OK] Metadata stored: {doc_meta.regulasi[:45]}... | {doc_meta.tipe} | {doc_meta.tanggal}")

    return extract_main_content(client, md_content, csv_path, doc_meta, log_file)

# ==========================================
# COMBINE CSVS
# ==========================================
def combine_csvs(csv_paths, output_path):
    if not csv_paths:
        return 0

    total_rows = 0
    header_written = False

    with open(output_path, "w", encoding="utf-8") as out:
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                continue
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                continue

            data_start = 0
            if not header_written:
                if lines[0].startswith("Regulasi|"):
                    out.write(lines[0])
                    header_written = True
                data_start = 1
            else:
                for i, line in enumerate(lines):
                    if not line.startswith("Regulasi|"):
                        data_start = i
                        break
                    data_start = i + 1

            for line in lines[data_start:]:
                if line.strip():
                    out.write(line)
                    total_rows += 1

    return total_rows

def deduplicate_csv(csv_path):
    """Remove duplicate data rows (byte-for-byte identical) from a CSV in-place. Keeps the header."""
    if not os.path.exists(csv_path):
        return 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if len(lines) <= 1:
        return 0
    
    header = lines[0]
    data_lines = lines[1:]
    
    seen = set()
    unique = []
    for line in data_lines:
        stripped = line.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique.append(line)
    
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(unique)
    
    return len(unique)

# ==========================================
# CLI APPLICATION LOOP
# ==========================================
def parse_file_selection(input_str, total_files):
    """Parse comma/range selection like '2,4,5,6' or '2-4' or '1,3-5,7' into list of indices.
    
    Returns a sorted, deduplicated list of file indices (0-based).
    Choice '1' is reserved for 'all files' and is handled before calling this.
    """
    selected = set()
    parts = [p.strip() for p in input_str.split(",")]
    
    for part in parts:
        if not part:
            continue
        if "-" in part:
            range_parts = part.split("-", 1)
            try:
                start = int(range_parts[0].strip())
                end = int(range_parts[1].strip())
                for i in range(start, end + 1):
                    selected.add(i)
            except ValueError:
                return None  # Invalid range
        else:
            try:
                selected.add(int(part))
            except ValueError:
                return None  # Invalid number
    
    return sorted(selected)

def get_file_choice(files, file_type):
    if not files:
        print(f"\nNo .{file_type} files found.")
        return []
        
    print(f"\nWhich .{file_type} file(s) would you like to process?")
    print("  1. All files")
    for i, file_path in enumerate(files, start=2):
        print(f"  {i}. {os.path.basename(file_path)}")
    print("\n  Examples: '2,4,5,6' | '2-4' | '1,3-5,7' | '1' (all)")
    
    choice = input("\nEnter choice: ").strip()
    
    if choice == '1':
        return files
    
    parsed = parse_file_selection(choice, len(files))
    if parsed is None:
        print("Invalid input. Use numbers, commas, and dashes (e.g. '2,4,5,6' or '2-4').")
        return []
    
    # Map display numbers (1-based, 2+) to file indices (0-based)
    result = []
    for num in parsed:
        if num < 2 or num > len(files) + 1:
            print(f"  [!] '{num}' is out of range (valid: 2-{len(files) + 1}). Skipping.")
            continue
        result.append(files[num - 2])
    
    if not result:
        print("No valid files selected.")
        return []
    
    return result

def convert_pdfs_to_markdown_batch(selected_files):
    print(f"\n[PHASE 1] Multi-threaded PDF to Markdown...")
    converted = []
    with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 1) + 4)) as executor:
        futures = {}
        for file_path in selected_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            md_path = os.path.join(CONFIG["WORK_DIR"], f"{base_name}.md")
            choice = prompt_overwrite(md_path)
            if choice == 'c':
                break
            if choice == 'y':
                futures[executor.submit(convert_pdf_to_md, file_path, md_path, quiet=True)] = base_name

        for future in as_completed(futures):
            try:
                future.result()
                print(f"  [OK] Converted: {futures[future]}.pdf")
                converted.append(futures[future])
            except Exception as e:
                print(f"  [ERROR] Failed to convert {futures[future]}.pdf: {e}")

    return converted


def extract_csvs_from_markdown_batch(selected_files, log_file, move_pdf_on_failure=False):
    print(f"\n[PHASE 2] Sequential LLM Extraction...")
    api_start_time = time.time()
    batch_results = []

    for idx, file_path in enumerate(selected_files, start=1):
        file_start_time = time.time()
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        md_path = os.path.join(CONFIG["WORK_DIR"], f"{base_name}.md")
        csv_path = os.path.join(CONFIG["WORK_DIR"], f"{base_name}.csv")

        eta_str = "Calculating..." if idx == 1 else format_time(
            ((time.time() - api_start_time) / (idx - 1)) * (len(selected_files) - idx + 1))

        print(f"\n--- [File {idx}/{len(selected_files)}] {base_name} | ETA: {eta_str} ---")

        try:
            csv_choice = prompt_overwrite(csv_path)
            if csv_choice == 'c':
                break
            if csv_choice != 'y':
                batch_results.append({"file": base_name, "csv_path": "", "rows": 0, "cycles": 0, "status": "Skipped", "duration": 0})
                continue

            rows_ext, abort_batch, errs, cycles = process_single_file(md_path, csv_path, log_file)
            status = "Success" if not errs else f"Warn ({len(errs)} Bad)"
            batch_results.append({"file": base_name, "csv_path": csv_path, "rows": rows_ext,
                                  "cycles": cycles, "status": status, "duration": time.time() - file_start_time})
            if abort_batch:
                break

        except APIQuotaExhaustedError:
            append_to_log(log_file, "--- API QUOTA EXHAUSTED ---\n")
            print(f"\n  [FATAL] API quota exhausted. Stopping batch.")
            batch_results.append({"file": base_name, "csv_path": "", "rows": 0, "cycles": 1, "status": "Failed (Quota)",
                                   "duration": time.time() - file_start_time})
            break

        except Exception as e:
            append_to_log(log_file, f"--- CRITICAL ERROR ---\n{str(e)}\n")
            print(f"\n  [CRITICAL ERROR] {str(e)}")
            try:
                if move_pdf_on_failure:
                    shutil.move(file_path, os.path.join(CONFIG["FAILED_DIR"], os.path.basename(file_path)))
                shutil.move(md_path, os.path.join(CONFIG["FAILED_DIR"], os.path.basename(md_path)))
                print(f"  [QUARANTINED] Moved {base_name} to '{CONFIG['FAILED_DIR']}'")
            except Exception as quarantine_err:
                print(f"  [WARNING] Failed to quarantine files: {quarantine_err}")
            batch_results.append({"file": base_name, "csv_path": "", "rows": 0, "cycles": 1, "status": "Failed",
                                   "duration": time.time() - file_start_time})

    return batch_results


def print_batch_report(batch_results, global_start_time, total_selected=0):
    if not batch_results:
        return None

    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    total_global_time = time.time() - global_start_time

    print("\n\n" + "=" * 89)
    print("                                 BATCH EXECUTION REPORT")
    print("=" * 89)
    print(f"{'File Name':<35} | {'Rows':<6} | {'Cycles':<6} | {'Duration':<9} | {'Status'}")
    print("-" * 89)

    total_rows = success_count = 0
    for result in batch_results:
        display_name = result['file'][:33] + '..' if len(result['file']) > 35 else result['file']
        print(f"{display_name:<35} | {result['rows']:<6} | {result['cycles']:<6} | {format_time(result['duration']):<9} | {result['status']}")
        total_rows += result['rows']
        if "Success" in result['status'] or "Warn" in result['status']:
            success_count += 1

    print("-" * 89)
    attempted = len(batch_results)
    if total_selected and attempted < total_selected:
        print(f"  Files Attempted: {attempted}/{total_selected} (batch stopped early)")
    print(f"  Total Time: {format_time(total_global_time)} | Data Rows: {total_rows} | Success Rate: {success_count}/{attempted}")
    print("=" * 89)

    report_data = {
        "timestamp": TIMESTAMP,
        "total_time_sec": round(total_global_time, 2),
        "total_rows": total_rows,
        "success_count": success_count,
        "total_files": len(batch_results),
        "total_selected": total_selected,
        "files": [{"file": result["file"], "rows": result["rows"], "cycles": result["cycles"],
                   "duration_sec": round(result["duration"], 2), "status": result["status"]}
                  for result in batch_results]
    }
    report_path = os.path.join(CONFIG["LOG_DIR"], f"report_{TIMESTAMP}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n  [REPORT] Saved to: {report_path}")

    return TIMESTAMP


def main():
    log_file = setup_environment()
    print("\n===Script Configs===")
    print(CONFIG)
    print("\n=== POJK Data Extraction Pipeline ===")
    print("1. Convert PDF to MD\n2. Convert MD to CSV (Via Line JSON Extraction)\n3. Convert PDF to CSV (End-to-End)\n4. Combine existing CSVs\n0. Exit")

    action = input("\nSelect action: ").strip()
    if action not in ['1', '2', '3', '4']: sys.exit(0)
    if action in ['1', '3']:
        file_type = "pdf"
    elif action == '2':
        file_type = "md"
    else:
        file_type = "csv"

    target_files = glob.glob(os.path.join(CONFIG["WORK_DIR"], f"*.{file_type}"))
    selected_files = get_file_choice(target_files, file_type)
    if not selected_files:
        sys.exit(0)

    if action == '4':
        TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_path = os.path.join(CONFIG["WORK_DIR"], f"combined_{TIMESTAMP}.csv")
        combined_rows = combine_csvs(selected_files, combined_path)
        print(f"\n  [OK] Combined {len(selected_files)} CSV(s) -> {combined_path} ({combined_rows} data rows)")
        sys.exit(0)

    global_start_time = time.time()
    batch_results = []

    if action in ['1', '3']:
        convert_pdfs_to_markdown_batch(selected_files)

    if action in ['2', '3']:
        batch_results = extract_csvs_from_markdown_batch(selected_files, log_file, move_pdf_on_failure=(action == '3'))
    timestamp = print_batch_report(batch_results, global_start_time, total_selected=len(selected_files))

    if timestamp is None:
        return

    successful_csvs = []
    for result in batch_results:
        if result.get("csv_path") and os.path.exists(result["csv_path"]):
            successful_csvs.append(result["csv_path"])

    if len(successful_csvs) <= 1:
        return

    choice = input("\n  Combine all generated CSVs into one file? (y/n): ").strip().lower()

    if choice != 'y':
        return

    combined_name = f"combined_{timestamp}.csv"
    combined_path = os.path.join(CONFIG["WORK_DIR"], combined_name)
    combined_rows = combine_csvs(successful_csvs, combined_path)
    print(f"  [OK] Combined {len(successful_csvs)} CSV(s) -> {combined_path} ({combined_rows} data rows)")

if __name__ == "__main__":
    main()
