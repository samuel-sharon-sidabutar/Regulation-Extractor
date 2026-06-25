import os
import glob
import time
import shutil
import sys
from datetime import datetime
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from markitdown import MarkItDown
from openai import OpenAI

# ==========================================
# CONFIGURATION & CENTRALIZED SETTINGS
# ==========================================
CONFIG = {
    "WORK_DIR": "regulations",
    "FAILED_DIR": os.path.join("regulations", "failed_files"),
    "LOG_DIR": os.path.join("regulations", "minimax_logs"),
    "CONTINUOUS_EXECUTION": True,
    "FORCED_OVERWRITE": True,
    "ROUTER_IP": "https://api.tokenrouter.com/v1",
    "ROUTER_KEY": os.environ.get("TOKENROUTER_API_KEY"),
    "MODEL_ID": "MiniMax-M3"
}

# Used to structure data integrity errors dynamically
MalformedRow = namedtuple("MalformedRow", ["row_num", "col_count"])

LOG_FILE = "" # Will be initialized during setup_environment()

# ==========================================
# PROMPTS
# ==========================================
METADATA_PROMPT = """Act as an expert data extractor. Look at the provided Indonesian banking regulation (POJK).
Extract EXACTLY ONE line of text containing these 3 values separated by pipes (|). Do not include a header row.

Format: Regulasi|Tipe|Tanggal

1. Regulasi: The full title of the regulation document word-for-word.
2. Tipe: "POJK" (or based on the document type).
3. Tanggal: The date the regulation was enacted (Diundangkan/Ditetapkan).

Output only the data row.
"""

MASTER_PROMPT = """Act as an expert data extractor and compliance analyst. I am building a compliance app and need to import Indonesian banking regulations (POJK) into my database. 

Your task is to extract the hierarchical regulation data and output it strictly as a Pipe-Separated Values (PSV) text block.

**DATA SCHEMA (8 COLUMNS):**
`Bagian (Section)|Kode Item|Item Ref|Item|Item Description|Kode Sub Item|Sub Item Ref|Sub Item`

**INDONESIAN LEGAL HIERARCHY RULES (CRITICAL):**
You must strictly understand how Indonesian regulations are structured:
* **Pasal** (Article): The top-level grouping.
* **Ayat** (Paragraph): Numbered lists inside a Pasal, e.g., (1), (2). An 'Ayat' is ALWAYS an `Item`. You must NEVER group multiple Ayat together. Each Ayat gets its own row(s).
* **List Elements** (Huruf/Angka): Lists inside an Ayat or Pasal, whether they are letters (a, b, c) OR numbers (1, 2, 3). A List Element is ALWAYS a `Sub Item`.

**DYNAMIC HIERARCHY SCENARIOS:**
Not all Pasals use Ayat. You must analyze each Pasal dynamically:
* **Scenario A (Pasal has Ayat):** If a Pasal is divided into numbered Ayat (1, 2...), the **Ayat is the Item**. Any list elements (a, b... or 1, 2...) under it are Sub-items. The `Item Ref` is "Pasal X ayat Y".
* **Scenario B (Pasal jumps to a List):** If a Pasal has NO Ayat and jumps straight into a list of letters or numbers, the **Pasal is the Item**. The list elements are the Sub-items. The `Item Ref` is just "Pasal X".
* **Scenario C (Standalone Pasal):** If a Pasal has no sub-divisions at all, the **Pasal is the Item**. The Sub-item columns remain empty. The `Item Ref` is just "Pasal X".

**COLUMN MAPPING INSTRUCTIONS:**
1. **Bagian (Section)**: The Chapter/Section name (e.g., "Bab II"). *See "CATEGORIZATION RULES" below.*
2. **Kode Item**: A continuous, incremental integer starting from 1 for each new `Item Ref`.
3. **Item Ref**: The parent reference (e.g., "Pasal 2" or "Pasal 2 ayat 1"). NEVER include sub-item references like "huruf a" or "angka 1" here.
4. **Item**: The main text of the Item. Strip the numbering prefix (e.g., write "BPR wajib..." instead of "(1) BPR wajib..."). DO NOT put sub-item list text here.
5. **Item Description**: Find the explanation in "PENJELASAN". If it says "Cukup jelas.", write "Cukup jelas.". Only include the explanation for this specific Item Ref.
6. **Kode Sub Item**: Incremental integer (1, 2, 3...) for sub-items. Leave empty if none.
7. **Sub Item Ref**: The letter or number of the list item (e.g., "a", "b", "1", "2"). Leave empty if none.
8. **Sub Item**: The actual text of the sub-item. Strip the prefix (e.g., write "agar kualitas..." instead of "a. agar kualitas..." or "1. agar kualitas..."). If there is a 2nd-level nested list, concatenate them clearly within this cell. Leave empty if none.

**EXTRACTION & FORMATTING RULES:**
* **Column Count (CRITICAL)**: Every single row MUST have exactly 8 columns (which means exactly 7 pipe `|` characters). Check your output carefully.
* **Delimiter**: Use ONLY the pipe character (`|`). NEVER use a pipe character inside the actual text sentences.
* **Empty Fields**: If a field is empty, output nothing between the pipes (e.g., `value||value`). Do NOT skip the column.
* **CRITICAL - FLATTENING LISTS**: You MUST separate parent text from sub-item text. If an Item has Sub-items, you must output a new row for EACH Sub-item. You must repeat the parent `Item Ref` and parent `Item` text on EVERY row, while changing the `Sub Item` columns for the specific list element. Never promote a Sub-item to a main Item.

**CATEGORIZATION RULES (CRITICAL):**
DO NOT REORDER THE OUTPUT. Process CHRONOLOGICALLY. 
Overwrite `Bagian (Section)` column exactly as follows for special articles:
* **"Ketentuan"**: Glossaries or definitions of terms.
* **"Sanksi"**: Administrative consequences or penalties.
* **"Regulasi Lain"**: Legal meta-data (transitional timelines, effective dates).
* For all other operational rules, keep the original Chapter (e.g., "Bab II").

When finished extracting every article, output EXACTLY `[END_OF_DOCUMENT]` on a new line.
"""

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def setup_environment():
    """Initializes paths, checks dependencies, and enforces API key presence."""
    global LOG_FILE
    
    # Fail-Fast API check
    if not CONFIG["ROUTER_KEY"]:
        print("\n[CRITICAL ERROR] 'TOKENROUTER_API_KEY' environment variable is missing.")
        print("Please configure your API key before running this pipeline. Exiting...")
        sys.exit(1)
        
    for folder in [CONFIG["WORK_DIR"], CONFIG["FAILED_DIR"], CONFIG["LOG_DIR"]]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG_FILE = os.path.join(CONFIG["LOG_DIR"], f"run_{TIMESTAMP}.txt")

    if not glob.glob(os.path.join(CONFIG["WORK_DIR"], "*.pdf")) and not glob.glob(os.path.join(CONFIG["WORK_DIR"], "*.md")):
        print(f"I just ensured the '{CONFIG['WORK_DIR']}' folder exists. Please drop your PDFs/MDs in there and run me again!")
        sys.exit(0)

def append_to_log(text_block):
    with open(LOG_FILE, "a", encoding="utf-8") as lf:
        lf.write(text_block)

def prompt_overwrite(filepath):
    filename = os.path.basename(filepath)
    exists = os.path.exists(filepath)
    
    if CONFIG["FORCED_OVERWRITE"]:
        if exists:
            print(f"  [!] Overwriting existing file: '{filename}'")
        return 'y'
        
    if exists:
        while True:
            choice = input(f"  [!] '{filename}' already exists. Overwrite? (y/n / c to cancel): ").strip().lower()
            if choice in ['y', 'n', 'c']:
                return choice
    return 'y'

def is_document_ended(text):
    """Helper to catch multiple variations and cut-off fragments of the end trigger."""
    signals = [
        "[END_OF_DOCUMENT]", "[END_OF_DOCUMENT", "END_OF_DOCUMENT",
        "[END_OF_DOC", "END_OF_DOC", "[END_OF", "[END"
    ]
    return any(signal in text.upper() for signal in signals)

def format_time(seconds):
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"

# ==========================================
# DATA PROCESSING & SANITIZATION
# ==========================================
def sanitize_and_validate_csv(raw_text, meta_regulasi, meta_tipe, meta_tanggal):
    """Takes the 8-column AI output, cleans it, and reconstructs the 14-column DB schema."""
    lines = raw_text.split('\n')
    cleaned_lines = []
    malformed_errors = []
    data_row_count = 0

    db_header = "Regulasi|Tipe|Tanggal|Status|Properti Regulasi|Bagian (Section)|Kode Item|Item Ref|Item|Item Description|Property|Kode Sub Item|Sub Item Ref|Sub Item"
    cleaned_lines.append(db_header)

    for i, line in enumerate(lines):
        line = line.strip()
        # Drop empty lines, the header, and the end tags
        if not line or "Bagian (Section)|Kode Item" in line or is_document_ended(line):
            continue

        cols = line.split('|')
        
        if len(cols) != 8:
            malformed_errors.append(MalformedRow(row_num=i + 1, col_count=len(cols)))
            # Fallback for malformed rows
            cleaned_lines.append(f"{meta_regulasi}|{meta_tipe}|{meta_tanggal}|||{line}") 
        else:
            c = [col.strip() for col in cols]
            
            # Map 8 columns + Meta into the 14 column schema
            final_cols = [
                meta_regulasi, # 0
                meta_tipe,     # 1
                meta_tanggal,  # 2
                "",            # 3: Status
                "",            # 4: Properti Regulasi
                c[0],          # 5: Bagian (Section)
                c[1],          # 6: Kode Item
                c[2],          # 7: Item Ref
                c[3],          # 8: Item
                c[4],          # 9: Item Description
                "",            # 10: Property
                c[5],          # 11: Kode Sub Item
                c[6],          # 12: Sub Item Ref
                c[7]           # 13: Sub Item
            ]
            
            cleaned_lines.append("|".join(final_cols))
            data_row_count += 1

    return "\n".join(cleaned_lines), data_row_count, malformed_errors

def print_summary(csv_path, malformed_errors):
    """Outputs the specific integrity results without truncating row numbers."""
    if not os.path.exists(csv_path):
        return 0
        
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) <= 1:
        print("  [!] Summary: File is empty.")
        return 0

    data_lines = [l for l in lines if not l.startswith("Regulasi|")]
    data_row_count = len(data_lines)

    title = "Unknown (Check File)"
    sections = set()

    for row in data_lines:
        cols = row.split('|')
        if len(cols) == 14:
            if title == "Unknown (Check File)" and cols[0]: 
                title = cols[0] 
            section_val = cols[5].strip()
            if section_val:
                sections.add(section_val)

    print("\n  ========================================================")
    print(f"  📊 EXTRACTION SUMMARY: {os.path.basename(csv_path)}")
    print("  ========================================================")
    print(f"  Title      : {title[:60]}...")
    print(f"  Total Rows : {data_row_count} data rows generated")
    
    section_list = sorted([s for s in list(sections) if s])
    display_sections = ", ".join(section_list) if section_list else "None found"
    print(f"  Sections   : {display_sections}")
    
    if malformed_errors:
        print("  --------------------------------------------------------")
        print(f"  🚨 DATA INTEGRITY WARNING: {len(malformed_errors)} Row(s) Misaligned")
        
        error_groups = {}
        for err in malformed_errors:
            if err.col_count not in error_groups:
                error_groups[err.col_count] = []
            error_groups[err.col_count].append(err.row_num)
            
        for col_count, rows in error_groups.items():
            # Fixed: Removed the list slicing truncation, all rows displayed
            row_str = ", ".join(map(str, rows))
            print(f"      - {col_count} AI cols found (Expected 8) on {len(rows)} row(s): {row_str}")
            
    print("  ========================================================\n")
    return data_row_count

# ==========================================
# CORE EXTRACTION LOGIC
# ==========================================
def convert_pdf_to_md(pdf_path, md_path, quiet=False):
    """Phase 1 execution unit: PDF to Markdown mapping."""
    if not quiet:
        print(f"  -> Converting {os.path.basename(pdf_path)} to Markdown...")
    md_converter = MarkItDown()
    result = md_converter.convert(pdf_path)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result.text_content)
    if not quiet:
        print(f"  [OK] Saved to {md_path}")
    return result.text_content

def call_openai_with_retry(client, model_id, conversation_history):
    max_retries = 3
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=conversation_history,
                temperature=0.0,
                stream=True,
                extra_body={"thinking": {"type": "disabled"}}
            )
            return response
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "credit limit" in error_msg.lower():
                print(f"\n  [!] API Quota Limit Hit (403). Pausing for 2.5 minutes to let credits reset...")
                time.sleep(150) 
                retry_count += 1
                print(f"  -> Resuming API call (Retry {retry_count}/{max_retries})...")
            else:
                raise e
    raise Exception("Max retries exceeded for API limit.")

def extract_metadata(client, md_content):
    """Phase 2 Module: Handles extracting document context metadata."""
    print("  -> Extracting document metadata (Regulasi, Tipe, Tanggal)...")
    
    meta_history = [
        {"role": "system", "content": METADATA_PROMPT},
        {"role": "user", "content": f"Here is the document:\n\n{md_content}"}
    ]
    
    meta_resp = call_openai_with_retry(client, CONFIG["MODEL_ID"], meta_history)
    
    meta_raw = ""
    for chunk in meta_resp:
        if chunk.choices and len(chunk.choices) > 0:
            meta_raw += chunk.choices[0].delta.content or ""
            
    append_to_log("--- METADATA EXTRACTION RESULT ---\n" + meta_raw.strip() + "\n----------------------------------\n\n")

    # Clean the metadata response
    meta_parts = meta_raw.replace('`', '').split('\n')[0].split('|')
    if len(meta_parts) >= 3:
        meta_regulasi = meta_parts[0].strip()
        meta_tipe = meta_parts[1].strip()
        meta_tanggal = meta_parts[2].strip()
    else:
        print("  [!] Failed to cleanly parse metadata. Using fallbacks.")
        meta_regulasi, meta_tipe, meta_tanggal = "Unknown", "POJK", "Unknown"
        
    return meta_regulasi, meta_tipe, meta_tanggal

def extract_main_content(client, md_content, csv_path, meta_regulasi, meta_tipe, meta_tanggal):
    """Phase 2 Module: Handles the core data extraction loop & PSV output stitching."""
    db_header_8 = "Bagian (Section)|Kode Item|Item Ref|Item|Item Description|Kode Sub Item|Sub Item Ref|Sub Item"
    user_string = f"Here is the document:\n\n{md_content}\n\n[BEGIN EXTRACTION NOW]"

    conversation_history = [
        {"role": "system", "content": MASTER_PROMPT},
        {"role": "user", "content": user_string},
        {"role": "assistant", "content": db_header_8 + "\n"}
    ]

    max_loops = 20 
    loop_count = 1 
    print(f"  -> Firing main extraction request (Cycle {loop_count}/{max_loops})...")
    
    response = call_openai_with_retry(client, CONFIG["MODEL_ID"], conversation_history)
    
    full_csv_text = db_header_8 + "\n"
    abort_batch = False 
    
    chunk_buffer = ""
    for chunk in response:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta.content or ""
            full_csv_text += delta
            chunk_buffer += delta
            
    append_to_log(f"--- CYCLE {loop_count} OUTPUT ---\n{chunk_buffer}\n")

    # Cycle generation logic until exact document completion marker is reached
    while not is_document_ended(full_csv_text) and loop_count < max_loops:
        
        # --- FRAGMENT DELETION LOGIC ---
        lines_in_buffer = full_csv_text.split('\n')
        last_char = full_csv_text[-1] if full_csv_text else ""
        
        if last_char != '\n' and lines_in_buffer and full_csv_text.strip():
            # If the AI got cut off mid-sentence trying to write END_OF_DOCUMENT
            if is_document_ended(lines_in_buffer[-1]):
                break
                
            discarded_fragment = lines_in_buffer.pop()
            safe_csv_to_save = '\n'.join(lines_in_buffer) + '\n'
            line_status = f"Cut off mid-row (Deleted fragment: '{discarded_fragment[:20]}...')"
        else:
            safe_csv_to_save = full_csv_text
            line_status = "Clean cut (Row finished)"
            
        current_row_count = len([line for line in lines_in_buffer if line.strip()])
        full_csv_text = safe_csv_to_save
        
        # Stitched Partial Save
        partial_text_to_save, _, _ = sanitize_and_validate_csv(full_csv_text, meta_regulasi, meta_tipe, meta_tanggal)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(partial_text_to_save)
            
        if not CONFIG["CONTINUOUS_EXECUTION"]:
            print(f"\n  [?] AI paused at row {current_row_count} ({line_status}). Stitched partial file saved.")
            user_choice = input(f"      Continue processing THIS file? (y/n): ").strip().lower()
            if user_choice != 'y':
                print("  [!] Halting early for this file.")
                batch_choice = input("      Do you want to skip the REST of the files in this batch? (y/n): ").strip().lower()
                if batch_choice == 'y':
                    abort_batch = True
                break
        else:
            print(f"  -> AI paused at row {current_row_count} ({line_status}). Stitched partial file saved. Auto-continuing...")

        loop_count += 1 
        print(f"  -> Continuing generation... (Cycle {loop_count}/{max_loops})")

        # Context Injection for Continuation
        raw_lines = full_csv_text.split('\n')
        last_context = '\n'.join(raw_lines[-30:])

        continuation_prompt = "Resume generating exactly where you left off based on your last message. Output ONLY raw PSV data."
        continuation_history = [
            {"role": "system", "content": MASTER_PROMPT},
            {"role": "user", "content": user_string},
            {"role": "assistant", "content": last_context},
            {"role": "user", "content": continuation_prompt}
        ]
        
        response = call_openai_with_retry(client, CONFIG["MODEL_ID"], continuation_history)
        
        new_chunk = ""
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta.content or ""
                new_chunk += delta
                
        full_csv_text += new_chunk
        append_to_log(f"--- CYCLE {loop_count} OUTPUT ---\n{new_chunk}\n")

    if loop_count >= max_loops:
        print("  [!] Safety limit reached: Max loops exceeded.")

    final_text_to_save, final_row_count, malformed_errors = sanitize_and_validate_csv(full_csv_text, meta_regulasi, meta_tipe, meta_tanggal)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(final_text_to_save)
        
    print(f"  [OK] Final stitched file saved to {csv_path}")
    print_summary(csv_path, malformed_errors)
    
    return final_row_count, abort_batch, malformed_errors, loop_count

def process_single_file(md_path, csv_path, md_content=None):
    """Facade that unites metadata and main content logic into a single cohesive pipeline."""
    filename_base = os.path.basename(md_path)
    print(f"  -> Processing {filename_base}...")
    
    if md_content is None:
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

    client = OpenAI(
        base_url=CONFIG["ROUTER_IP"],
        api_key=CONFIG["ROUTER_KEY"]
    )

    log_header = f"\n\n{'='*80}\n=== START AI RESPONSE LOG | File: {filename_base} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n{'='*80}\n"
    append_to_log(log_header)

    meta_regulasi, meta_tipe, meta_tanggal = extract_metadata(client, md_content)
    
    display_title = meta_regulasi[:45] + "..." if len(meta_regulasi) > 45 else meta_regulasi
    print(f"  [OK] Metadata stored: {display_title} | {meta_tipe} | {meta_tanggal}")

    final_row_count, abort_batch, malformed_errors, loop_count = extract_main_content(
        client, md_content, csv_path, meta_regulasi, meta_tipe, meta_tanggal
    )
    
    return final_row_count, abort_batch, malformed_errors, loop_count

# ==========================================
# CLI APPLICATION LOOP
# ==========================================
def get_file_choice(files, file_type):
    if not files:
        print(f"\nNo .{file_type} files found in '{CONFIG['WORK_DIR']}'.")
        return []
        
    print(f"\nWhich .{file_type} file would you like to process?")
    print(f"1. All files of type .{file_type}")
    
    for i, file_path in enumerate(files, start=2):
        base_name = os.path.basename(file_path)
        print(f"{i}. {base_name}")
        
    choice = input("\nEnter your choice (number): ").strip()
    
    if choice == '1':
        return files
    else:
        try:
            index = int(choice) - 2
            if 0 <= index < len(files):
                return [files[index]]
            else:
                print("Invalid choice.")
                return []
        except ValueError:
            print("Please enter a valid number.")
            return []

def main():
    setup_environment()
    
    print(f"\n=== POJK Data Extraction Pipeline ===")
    print(f"Logs actively saving to: {LOG_FILE}")
    print("1. Convert PDF to MD")
    print("2. Convert MD to CSV")
    print("3. Convert PDF to CSV (End-to-End)")
    print("0. Exit")
    
    action = input("\nSelect an action (number): ").strip()
    
    if action == '0':
        print("Exiting...")
        sys.exit(0)
        
    if action not in ['1', '2', '3']:
        print("Invalid choice. Please run again.")
        sys.exit(0)

    # Initialize file pools purely based on intent
    if action in ['1', '3']:
        target_files = glob.glob(os.path.join(CONFIG["WORK_DIR"], "*.pdf"))
        file_type = "pdf"
    elif action == '2':
        target_files = glob.glob(os.path.join(CONFIG["WORK_DIR"], "*.md"))
        file_type = "md"

    selected_files = get_file_choice(target_files, file_type)
    
    if not selected_files:
        print("Processing cancelled.")
        sys.exit(0)

    print(f"\nStarting process for {len(selected_files)} file(s)...")
    
    batch_results = []
    global_start_time = time.time()

    if action in ['1', '3']:
        print(f"\n[PHASE 1] Multi-threaded PDF to Markdown Pre-processing...")
        
        # Fix: Compute balanced thread ceiling avoiding RAM spikes
        optimal_workers = min(8, (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            futures = {}
            # Logic Fix: Only loop over the specifically user-selected files, avoiding rogue globs
            for file_path in selected_files:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                md_path = os.path.join(CONFIG["WORK_DIR"], f"{base_name}.md")
                
                choice = prompt_overwrite(md_path)
                if choice == 'y':
                    futures[executor.submit(convert_pdf_to_md, file_path, md_path, quiet=True)] = base_name
                elif choice == 'c':
                    print("Conversion phase cancelled.")
                    break
                else:
                    print(f"  [SKIPPED] {base_name}.md (Already exists)")
            
            for future in as_completed(futures):
                b_name = futures[future]
                try:
                    future.result()
                    print(f"  [OK] Converted: {b_name}.pdf")
                except Exception as e:
                    print(f"  [ERROR] Failed to convert {b_name}.pdf: {e}")

    if action in ['2', '3']:
        print(f"\n[PHASE 2] Sequential LLM Extraction...")
        total_files = len(selected_files)
        api_start_time = time.time()
        
        for idx, file_path in enumerate(selected_files, start=1):
            file_start_time = time.time()
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            md_path = os.path.join(CONFIG["WORK_DIR"], f"{base_name}.md")
            csv_path = os.path.join(CONFIG["WORK_DIR"], f"{base_name}.csv")

            if idx > 1:
                avg_time_per_file = (time.time() - api_start_time) / (idx - 1)
                eta_seconds = avg_time_per_file * (total_files - idx + 1)
                eta_str = format_time(eta_seconds)
            else:
                eta_str = "Calculating..."

            print(f"\n--- [File {idx}/{total_files}] {base_name} | ETA: {eta_str} ---")
            
            try:
                csv_choice = prompt_overwrite(csv_path)
                if csv_choice == 'c':
                    break
                elif csv_choice == 'y':
                    if not os.path.exists(md_path):
                        raise FileNotFoundError(f"Missing Markdown file: {md_path}")
                        
                    rows_extracted, abort_batch, malformed_errors, cycles_used = process_single_file(md_path, csv_path)
                    duration = time.time() - file_start_time
                    
                    # Fix: Granular reporting showing exactly which rows require compliance review
                    status_text = "Success"
                    if malformed_errors:
                        bad_rows = ", ".join([str(err.row_num) for err in malformed_errors])
                        status_text = f"Warn (Check Rows: {bad_rows})"
                    
                    batch_results.append({
                        "file": base_name,
                        "rows": rows_extracted,
                        "cycles": cycles_used,
                        "status": status_text,
                        "duration": duration
                    })
                    
                    if abort_batch:
                        print("\n  [!] Batch processing aborted by user.")
                        break

                else:
                    print(f"  [SKIPPED] CSV generation.")
                    batch_results.append({
                        "file": base_name,
                        "rows": 0,
                        "cycles": 0,
                        "status": "Skipped",
                        "duration": 0
                    })
                    
            except Exception as e:
                duration = time.time() - file_start_time
                error_msg = str(e)
                print(f"\n  [CRITICAL ERROR] Failed to process {base_name}. Reason: {error_msg}")
                append_to_log(f"--- CRITICAL ERROR ---\n{error_msg}\n")
                
                try:
                    if os.path.exists(file_path) and action == '3':
                        shutil.move(file_path, os.path.join(CONFIG["FAILED_DIR"], os.path.basename(file_path)))
                    if os.path.exists(md_path):
                        shutil.move(md_path, os.path.join(CONFIG["FAILED_DIR"], os.path.basename(md_path)))
                    print(f"  [QUARANTINED] Moved {base_name} files to '{CONFIG['FAILED_DIR']}'")
                except Exception as move_err:
                    print(f"  [WARNING] Could not quarantine file: {move_err}")

                batch_results.append({
                    "file": base_name,
                    "rows": 0,
                    "cycles": 1,
                    "status": f"Failed ({error_msg[:25]}...)",
                    "duration": duration
                })

    if batch_results:
        total_global_time = time.time() - global_start_time
        print("\n\n")
        print("=========================================================================================")
        print("                                BATCH EXECUTION REPORT")
        print("=========================================================================================")
        # Status column is flexible to accommodate larger arrays of broken row strings
        print(f"{'File Name':<35} | {'Rows':<6} | {'Cycles':<6} | {'Duration':<9} | {'Status'}")
        print("-" * 89)
        
        total_rows = 0
        success_count = 0
        
        for res in batch_results:
            dur_str = format_time(res['duration'])
            f_name = res['file'][:33] + ".." if len(res['file']) > 35 else res['file']
            print(f"{f_name:<35} | {res['rows']:<6} | {res['cycles']:<6} | {dur_str:<9} | {res['status']}")
            total_rows += res['rows']
            if "Success" in res['status'] or "Warn" in res['status']: success_count += 1
            
        print("-" * 89)
        print(f"  Total Batch Time : {format_time(total_global_time)}")
        print(f"  Total Data Rows  : {total_rows}")
        print(f"  Processed Rate   : {success_count}/{len(batch_results)} files processed")
        print("=========================================================================================\n")

if __name__ == "__main__":
    main()