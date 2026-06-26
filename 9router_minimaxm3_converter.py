import os
import glob
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from markitdown import MarkItDown
from openai import OpenAI

# ==========================================
# CONFIGURATION & FOLDER SETUP
# ==========================================
WORK_DIR = "regulations"
FAILED_DIR = os.path.join(WORK_DIR, "failed_files")

CONTINUOUS_EXECUTION = True
FORCED_OVERWRITE = True

ROUTER_IP = "http://43.157.202.81:20128/v1" 
ROUTER_KEY = "sk-6b3ac6ef8e3b70c9-qy4p0x-767d185d" # deleted
MODEL_ID = "MiniMax-M3"  

for folder in [WORK_DIR, FAILED_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

if not glob.glob(os.path.join(WORK_DIR, "*.pdf")) and not glob.glob(os.path.join(WORK_DIR, "*.md")):
    print(f"I just ensured the '{WORK_DIR}' folder exists. Please drop your PDFs in there and run me again!")
    exit()

# === PROMPT UPDATED: 11 COLUMNS & DASHES ===
MASTER_PROMPT = """Act as an expert data extractor and compliance analyst. I am building a compliance app and need to import Indonesian banking regulations (POJK) into my database. 

Your task is to extract the hierarchical regulation data from the provided markdown document and output it strictly as a Pipe-Separated Values (PSV) text block.

**DATA SCHEMA (11 COLUMNS):**
You must generate a PSV with the following exact headers:
`Regulasi|Tipe|Tanggal|Bagian (Section)|Kode Item|Item Ref|Item|Item Description|Kode Sub Item|Sub Item Ref|Sub Item`

**COLUMN MAPPING INSTRUCTIONS:**
1. **Regulasi**: The full title of the regulation document word-for-word.
2. **Tipe**: "POJK" (or based on the document type).
3. **Tanggal**: The date the regulation was enacted (Diundangkan/Ditetapkan).
4. **Bagian (Section)**: The Chapter/Section name (e.g., "Bab II"). *See "CATEGORIZATION RULES" below.*
5. **Kode Item**: A continuous, incremental integer starting from 1 for each new `Item Ref`.
6. **Item Ref**: The specific article and paragraph reference (e.g., "Pasal 2 ayat 1").
7. **Item**: The main text of the article/paragraph.
8. **Item Description**: Find the explanation in "PENJELASAN". If it says "Cukup jelas.", write "Cukup jelas.".
9. **Kode Sub Item**: Incremental integer (1, 2, 3...) for sub-items. If none, output exactly: -
10. **Sub Item Ref**: The letter/number of the list item (e.g., "a", "b"). If none, output exactly: -
11. **Sub Item**: The actual text of the sub-item. If none, output exactly: -

**EXTRACTION & FORMATTING RULES:**
* **Delimiter**: Use ONLY the pipe character (`|`).
* **Empty Fields**: If a field is empty (like a sub-item), you MUST output a single dash `-`. Do not leave a space between pipes.
* **Flatten Merged Cells**: Do NOT leave parent columns blank when listing sub-items. You must repeat the parent data on every single row.

**CATEGORIZATION RULES (CRITICAL):**
DO NOT REORDER THE OUTPUT. Process CHRONOLOGICALLY. 
Overwrite `Bagian (Section)` column exactly as follows for special articles:
* Glossaries/Definitions: `Ketentuan`
* Administrative penalties: `Sanksi`
* Legal meta-data (effective dates, revocations): `Regulasi Lain`
* All other operational rules: keep the original Chapter (e.g., "Bab II").

**END OF DOCUMENT TRIGGER (CRITICAL):**
When finished extracting every article, output EXACTLY `[END_OF_DOCUMENT]` on a new line.

**CRITICAL FORMATTING EXAMPLE (MIMIC EXACTLY):**
Nama Regulasi|POJK|12 Agustus 2024|Bab II|1|Pasal 3 ayat 1|Bank Wajib.|Cukup jelas.|1|a|Edukasi;
Nama Regulasi|POJK|12 Agustus 2024|Bab II|1|Pasal 3 ayat 1|Bank Wajib.|Cukup jelas.|2|b|Sanksi;
Nama Regulasi|POJK|12 Agustus 2024|Bab II|2|Pasal 4|Sanksi berlaku.|Denda 1 Miliar.|-|-|-
"""

def prompt_overwrite(filepath):
    if FORCED_OVERWRITE:
        return 'y'
        
    if os.path.exists(filepath):
        while True:
            filename = os.path.basename(filepath)
            choice = input(f"  [!] '{filename}' already exists. Overwrite? (y/n / c to cancel): ").strip().lower()
            if choice in ['y', 'n', 'c']:
                return choice
    return 'y'

def sanitize_and_validate_csv(raw_text):
    """Takes the 11-column AI output, cleans it, and reconstructs the 14-column DB schema."""
    lines = raw_text.split('\n')
    cleaned_lines = []
    malformed_errors = []
    data_row_count = 0

    # Inject the final 14-column database header
    db_header = "Regulasi|Tipe|Tanggal|Status|Properti Regulasi|Bagian (Section)|Kode Item|Item Ref|Item|Item Description|Property|Kode Sub Item|Sub Item Ref|Sub Item"
    cleaned_lines.append(db_header)

    for i, line in enumerate(lines):
        line = line.strip()
        # Skip empty lines, the end trigger, and the AI's 11-column header
        if not line or line == "[END_OF_DOCUMENT]" or "Regulasi|Tipe|Tanggal" in line:
            continue

        cols = line.split('|')
        
        # Schema Check: We expect the AI to output exactly 11 columns
        if len(cols) != 11:
            malformed_errors.append((i + 1, len(cols)))
            cleaned_lines.append(line) # Keep raw so data isn't lost
        else:
            # Clean out the dashes and whitespace
            c = [col.strip() if col.strip() != '-' else "" for col in cols]
            
            # Reconstruct to the 14-column format by injecting blanks
            final_cols = [
                c[0],  # 0: Regulasi
                c[1],  # 1: Tipe
                c[2],  # 2: Tanggal
                "",    # 3: Status (INJECTED BLANK)
                "",    # 4: Properti Regulasi (INJECTED BLANK)
                c[3],  # 5: Bagian (Section)
                c[4],  # 6: Kode Item
                c[5],  # 7: Item Ref
                c[6],  # 8: Item
                c[7],  # 9: Item Description
                "",    # 10: Property (INJECTED BLANK)
                c[8],  # 11: Kode Sub Item
                c[9],  # 12: Sub Item Ref
                c[10]  # 13: Sub Item
            ]
            
            cleaned_lines.append("|".join(final_cols))
            data_row_count += 1

    return "\n".join(cleaned_lines), data_row_count, malformed_errors

def print_summary(csv_path, malformed_errors):
    """Parses the finalized CSV and prints a UX-friendly summary."""
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
    print(f"  Title      : {title}")
    print(f"  Total Rows : {data_row_count} data rows generated")
    
    section_list = sorted([s for s in list(sections) if s])
    display_sections = ", ".join(section_list) if section_list else "None found"
    print(f"  Sections   : {display_sections}")
    
    if malformed_errors:
        print("  --------------------------------------------------------")
        print(f"  🚨 DATA INTEGRITY WARNING: {len(malformed_errors)} Row(s) Misaligned")
        
        error_groups = {}
        for err in malformed_errors:
            row_num, col_count = err[0], err[1]
            if col_count not in error_groups:
                error_groups[col_count] = []
            error_groups[col_count].append(row_num)
            
        for col_count, rows in error_groups.items():
            row_str = ", ".join(map(str, rows[:10]))
            if len(rows) > 10: row_str += f" ... (+{len(rows)-10} more)"
            print(f"      - {col_count} AI cols found (Expected 11) on {len(rows)} row(s): {row_str}")
            
    print("  ========================================================\n")
    
    return data_row_count

def convert_pdf_to_md(pdf_path, md_path, quiet=False):
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
    """Wrapper to handle 403 Rate Limit / Credit Limit pauses."""
    max_retries = 3
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=conversation_history,
                temperature=0.0,
                max_tokens=4096, 
                stream=True
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

def convert_md_to_csv(md_path, csv_path, md_content=None):
    print(f"  -> Sending {os.path.basename(md_path)} to Router...")
    
    if md_content is None:
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

    client = OpenAI(
        base_url=ROUTER_IP,
        api_key=ROUTER_KEY
    )

    user_string = f"{MASTER_PROMPT}\n\nHere is the document to process strictly according to system instructions:\n\n{md_content}\n\nPlease generate the PSV data for the attached document now."

    conversation_history = [
        {"role": "user", "content": user_string}
    ]

    max_loops = 20 
    loop_count = 1 
    
    print(f"  -> Firing initial request (Cycle {loop_count}/{max_loops})...")
    
    response = call_openai_with_retry(client, MODEL_ID, conversation_history)
    
    full_csv_text = ""
    abort_batch = False 
    
    for chunk in response:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta.content or ""
            full_csv_text += delta

    while "[END_OF_DOCUMENT]" not in full_csv_text and loop_count < max_loops:
        
        # --- NEW FRAGMENT DELETION LOGIC ---
        lines_in_buffer = full_csv_text.split('\n')
        last_char = full_csv_text[-1] if full_csv_text else ""
        
        if last_char != '\n' and lines_in_buffer:
            # It was cut off mid-sentence. Pop off the broken line and discard it.
            discarded_fragment = lines_in_buffer.pop()
            
            # Re-assemble the text so it ends cleanly on the last full row
            full_csv_text = '\n'.join(lines_in_buffer) + '\n'
            
            line_status = f"Cut off mid-row (Deleted fragment: '{discarded_fragment[:20]}...')"
        else:
            line_status = "Clean cut (Row finished)"
            
        current_row_count = len([line for line in lines_in_buffer if line.strip()])
        # -----------------------------------
        
        # --- NEW: Save intermediate state on EVERY cycle, regardless of execution mode ---
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(full_csv_text)
            
        if not CONTINUOUS_EXECUTION:
            print(f"\n  [?] AI paused at row {current_row_count} ({line_status}). Partial file saved.")
            user_choice = input(f"      Continue processing THIS file? (y/n): ").strip().lower()
            if user_choice != 'y':
                print("  [!] Halting early for this file.")
                batch_choice = input("      Do you want to skip the REST of the files in this batch? (y/n): ").strip().lower()
                if batch_choice == 'y':
                    abort_batch = True
                break
        else:
            print(f"  -> AI paused at row {current_row_count} ({line_status}). Partial file saved. Auto-continuing...")

        loop_count += 1 
        if not CONTINUOUS_EXECUTION:
            print(f"  -> Continuing generation... (Cycle {loop_count}/{max_loops})")

        # Grab the last 20 complete rows to give the AI a "running start" for numbering
        last_context_rows = "\n".join(full_csv_text.strip().split('\n')[-20:])

        # Updated prompt: AI starts the next row immediately based on the context
        continuation_prompt = "Continue extracting the document. I have provided your last completely finished rows above. Start generating the next chronological row immediately on a new line. Do not repeat headers. Keep outputting the 11-column PSV data using `-` for empty fields until the document is finished, then output EXACTLY [END_OF_DOCUMENT]."

        # --- NEW: Removed the explicitly printed string truncation brackets ---
        conversation_history = [
            {"role": "user", "content": user_string},
            {"role": "assistant", "content": last_context_rows},
            {"role": "user", "content": continuation_prompt}
        ]
        
        response = call_openai_with_retry(client, MODEL_ID, conversation_history)
        
        new_chunk = ""
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta.content or ""
                new_chunk += delta
                
        full_csv_text += new_chunk

    if loop_count >= max_loops:
        print("  [!] Safety limit reached: Max loops exceeded.")

    final_text_to_save, final_row_count, malformed_errors = sanitize_and_validate_csv(full_csv_text)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(final_text_to_save)
        
    print(f"  [OK] Final file saved to {csv_path}")
    print_summary(csv_path, malformed_errors)
    
    return final_row_count, abort_batch, len(malformed_errors), loop_count

def get_file_choice(files, file_type):
    if not files:
        print(f"\nNo .{file_type} files found in '{WORK_DIR}'.")
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

def format_time(seconds):
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"

def main():
    print("\n=== POJK Data Extraction Pipeline ===")
    print("1. Convert PDF to MD")
    print("2. Convert MD to CSV")
    print("3. Convert PDF to CSV (End-to-End)")
    print("0. Exit")
    
    action = input("\nSelect an action (number): ").strip()
    
    if action == '0':
        print("Exiting...")
        exit()
        
    if action not in ['1', '2', '3']:
        print("Invalid choice. Please run again.")
        exit()

    if action in ['1', '3']:
        target_files = glob.glob(os.path.join(WORK_DIR, "*.pdf"))
        file_type = "pdf"
    elif action == '2':
        target_files = glob.glob(os.path.join(WORK_DIR, "*.md"))
        file_type = "md"

    selected_files = get_file_choice(target_files, file_type)
    
    if not selected_files:
        print("Processing cancelled.")
        exit()

    print(f"\nStarting process for {len(selected_files)} file(s)...")
    
    batch_results = []
    global_start_time = time.time()

    if action in ['1', '3']:
        print(f"\n[PHASE 1] Multi-threaded PDF to Markdown Pre-processing...")
        with ThreadPoolExecutor() as executor:
            futures = {}
            for file_path in selected_files:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                md_path = os.path.join(WORK_DIR, f"{base_name}.md")
                
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
            md_path = os.path.join(WORK_DIR, f"{base_name}.md")
            csv_path = os.path.join(WORK_DIR, f"{base_name}.csv")

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
                        
                    rows_extracted, abort_batch, schema_errors, cycles_used = convert_md_to_csv(md_path, csv_path)
                    duration = time.time() - file_start_time
                    
                    status_text = "Success"
                    if schema_errors > 0:
                        status_text = f"Warning: {schema_errors} Rows Corrupt"
                    
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
                
                try:
                    if os.path.exists(file_path) and action == '3':
                        shutil.move(file_path, os.path.join(FAILED_DIR, os.path.basename(file_path)))
                    if os.path.exists(md_path):
                        shutil.move(md_path, os.path.join(FAILED_DIR, os.path.basename(md_path)))
                    print(f"  [QUARANTINED] Moved {base_name} files to '{FAILED_DIR}'")
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
        print(f"{'File Name':<35} | {'Rows':<6} | {'Cycles':<6} | {'Duration':<9} | {'Status'}")
        print("-" * 89)
        
        total_rows = 0
        success_count = 0
        
        for res in batch_results:
            dur_str = format_time(res['duration'])
            f_name = res['file'][:33] + ".." if len(res['file']) > 35 else res['file']
            print(f"{f_name:<35} | {res['rows']:<6} | {res['cycles']:<6} | {dur_str:<9} | {res['status']}")
            total_rows += res['rows']
            if "Success" in res['status'] or "Warning" in res['status']: success_count += 1
            
        print("-" * 89)
        print(f"  Total Batch Time : {format_time(total_global_time)}")
        print(f"  Total Data Rows  : {total_rows}")
        print(f"  Processed Rate   : {success_count}/{len(batch_results)} files processed")
        print("=========================================================================================\n")

if __name__ == "__main__":
    main()