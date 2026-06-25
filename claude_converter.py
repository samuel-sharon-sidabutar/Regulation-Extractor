import os
import glob
from markitdown import MarkItDown
import anthropic

# ==========================================
# CONFIGURATION & FOLDER SETUP
# ==========================================
WORK_DIR = "regulations"

# SETTING: Auto-continue if the document is too long?
# False = Asks for your permission (y/n) before spending more tokens.
# True = Automatically continues (up to a safe limit of 10 times).
CONTINUOUS_EXECUTION = False 

# SETTING: The AI Engine
MODEL_ID = "claude-sonnet-4-6" 

if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)
    print(f"I just created a folder named '{WORK_DIR}'. Please drop your PDFs in there and run me again!")
    exit()

MASTER_PROMPT = """Act as an expert data extractor and compliance analyst. I am building a compliance app and need to import Indonesian banking regulations (POJK) into my database. 

Your task is to extract the hierarchical regulation data from the provided markdown document and output it strictly as a Pipe-Separated Values (PSV) text block, ready for Excel import.

**DATA SCHEMA (COLUMNS):**
You must generate a PSV with the following exact headers:
`Regulasi|Tipe|Tanggal|Status|Properti Regulasi|Bagian (Section)|Kode Item|Item Ref|Item|Item Description|Property|Kode Sub Item|Sub Item Ref|Sub Item`

**COLUMN MAPPING INSTRUCTIONS:**
1. **Regulasi**: The full title of the regulation document.
2. **Tipe**: "POJK" (or based on the document type).
3. **Tanggal**: The date the regulation was enacted (Diundangkan/Ditetapkan).
4. **Status**: Leave blank.
5. **Properti Regulasi**: Leave blank.
6. **Bagian (Section)**: The Chapter/Section name (e.g., "Bab II"). *See "CATEGORIZATION RULES" below for critical exceptions.*
7. **Kode Item**: A continuous, incremental integer starting from 1 for each new `Item Ref`.
8. **Item Ref**: The specific article and paragraph reference (e.g., "Pasal 2 ayat 1").
9. **Item**: The main text of the article/paragraph.
10. **Item Description**: You MUST look at the "PENJELASAN" (Elucidation) section at the bottom of the markdown file. Find the corresponding explanation for this specific Pasal/Ayat. If it says "Cukup jelas.", write "Cukup jelas.".
11. **Property**: Leave blank.
12. **Kode Sub Item**: An incremental integer (1, 2, 3...) for sub-items (huruf/angka) under a specific Item. Leave blank if the Item has no sub-items.
13. **Sub Item Ref**: The letter or number of the list item (e.g., "a", "b", "1", "2"). Leave blank if the Item has no sub-items.
14. **Sub Item**: The actual text of the sub-item. *If there is a 2nd-level nested list (e.g., numbers under letters), concatenate them clearly within this cell using commas or semicolons so no information is lost.*

**EXTRACTION & FORMATTING RULES:**
* **Delimiter**: Use ONLY the pipe character (`|`) to separate columns.
* **Flatten Merged Cells**: Do NOT leave parent columns blank when listing sub-items. You must repeat the parent data on *every single row* that belongs to that Item.
* **Ignore Attachments**: Do not process the Appendices/Lampiran.

**CATEGORIZATION RULES (CRITICAL):**
**DO NOT REORDER THE OUTPUT.** You must process the document STRICTLY CHRONOLOGICALLY from top to bottom. Do not skip any articles to save space. 

However, you must logically analyze the articles and **OVERWRITE** their `Bagian (Section)` column exactly as follows:
* **"Ketentuan"**: Articles that are purely glossaries or definitions of terms (e.g., "Yang dimaksud dengan..."). Overwrite `Bagian (Section)` to exactly `Ketentuan`.
* **"Sanksi"**: Articles that strictly declare administrative consequences or penalties for violating preceding rules. Overwrite `Bagian (Section)` to exactly `Sanksi`.
* **"Regulasi Lain"**: Articles that are purely legal meta-data (transitional timelines, revoking old laws, effective dates). Overwrite `Bagian (Section)` to exactly `Regulasi Lain`.
* For all other operational rules, keep the `Bagian (Section)` as the original Chapter (e.g., "Bab II").

**OUTPUT FORMAT:**
Provide ONLY the raw PSV data. Do not include any conversational text, and do not wrap it in markdown code blocks.

**CRITICAL FORMATTING EXAMPLE (MIMIC THIS EXACTLY):**
DO NOT output nested lists, bullet points, asterisks, or JSON structures. You must ONLY output flat text rows separated by pipes (`|`). Every single row must have exactly 14 columns.

Example of correct output:
Nama Regulasi|POJK|12 Agustus 2024|||Bab II|1|Pasal 3 ayat 1|Bank Wajib menerapkan Anti Fraud.|Cukup jelas.||1|a|Meliputi edukasi;
Nama Regulasi|POJK|12 Agustus 2024|||Bab II|1|Pasal 3 ayat 1|Bank Wajib menerapkan Anti Fraud.|Cukup jelas.||2|b|Meliputi sanksi;
Nama Regulasi|POJK|12 Agustus 2024|||Bab II|2|Pasal 4|Sanksi berlaku.|Denda 1 Miliar.||||

If you generate a bullet point (`*`), you have failed your instructions.
"""

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def prompt_overwrite(filepath):
    if os.path.exists(filepath):
        while True:
            filename = os.path.basename(filepath)
            choice = input(f"  [!] '{filename}' already exists. Overwrite? (y/n / c to cancel): ").strip().lower()
            if choice in ['y', 'n', 'c']:
                return choice
    return 'y'

def convert_pdf_to_md(pdf_path, md_path):
    print(f"  -> Converting {os.path.basename(pdf_path)} to Markdown...")
    md_converter = MarkItDown()
    result = md_converter.convert(pdf_path)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result.text_content)
    print(f"  [OK] Saved to {md_path}")
    return result.text_content

def convert_md_to_csv(md_path, csv_path, md_content=None):
    print(f"  -> Sending {os.path.basename(md_path)} to Claude API...")
    
    if md_content is None:
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

    client = anthropic.Anthropic() 

    # Dual Cache Breakpoints: 
    # 1. Caches the rules across multiple files
    # 2. Caches the heavy document across continuation loops
    system_prompt_blocks = [
        {
            "type": "text",
            "text": MASTER_PROMPT,
            "cache_control": {"type": "ephemeral"} 
        },
        {
            "type": "text",
            "text": f"Here is the document to process strictly according to system instructions:\n\n{md_content}",
            "cache_control": {"type": "ephemeral"}
        }
    ]

    conversation_history = [
        {
            "role": "user", 
            "content": "Please generate the PSV data for the attached document now."
        }
    ]

    # 1. Fire the initial request to Claude
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=8192,
        temperature=0.0,
        system=system_prompt_blocks,
        messages=conversation_history
    )
    
    full_csv_text = response.content[0].text

    # 2. The Safe Auto-Continue Loop
    loop_count = 0
    max_loops = 10 
    
    while response.stop_reason == "max_tokens" and loop_count < max_loops:
        loop_count += 1
        
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(full_csv_text)
            
        if not CONTINUOUS_EXECUTION:
            user_choice = input(f"\n  [?] Output limit reached. Partial CSV saved to '{os.path.basename(csv_path)}'.\n      Continue processing? (y/n): ").strip().lower()
            if user_choice != 'y':
                print("  [!] Halting early by user request. (Partial file kept).")
                break
            print(f"  -> Continuing... (Cycle {loop_count})")
        else:
            print(f"  -> Output limit reached. Auto-continuing... (Cycle {loop_count}/{max_loops})")

        # Trim history optimization: Keep the required initial user message, add the latest output, and continue
        conversation_history = [
            conversation_history[0], 
            {"role": "assistant", "content": response.content[0].text},
            {
                "role": "user", 
                "content": "Continue exactly where you left off. DO NOT use bullet points or lists. You MUST continue outputting raw, flat Pipe-Separated Values (PSV) rows with exactly 14 columns. Do not repeat the header row."
            }
        ]
        
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=8192,
            temperature=0.0,
            system=system_prompt_blocks,
            messages=conversation_history
        )
        
        full_csv_text += "\n" + response.content[0].text

    if loop_count >= max_loops:
        print("  [!] Safety limit reached: Max loops exceeded to prevent infinite billing.")

    # 3. Final Save
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(full_csv_text)
        
    print(f"  [OK] Final file saved to {csv_path}")

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

# ==========================================
# MAIN INTERACTIVE MENU
# ==========================================
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

    print(f"\nStarting process for {len(selected_files)} file(s)...\n")

    for file_path in selected_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        md_path = os.path.join(WORK_DIR, f"{base_name}.md")
        csv_path = os.path.join(WORK_DIR, f"{base_name}.csv")

        print(f"--- Processing: {base_name} ---")
        try:
            if action == '1':
                choice = prompt_overwrite(md_path)
                if choice == 'c': break
                elif choice == 'y': convert_pdf_to_md(file_path, md_path)
                else: print(f"  [SKIPPED] {base_name}.md")
            
            elif action == '2':
                choice = prompt_overwrite(csv_path)
                if choice == 'c': break
                elif choice == 'y': convert_md_to_csv(md_path, csv_path)
                else: print(f"  [SKIPPED] {base_name}.csv")
                
            elif action == '3':
                md_content = None
                md_choice = prompt_overwrite(md_path)
                
                if md_choice == 'c': break
                elif md_choice == 'y': md_content = convert_pdf_to_md(file_path, md_path)
                else: 
                    print(f"  [SKIPPED] MD conversion. Reading existing {base_name}.md...")
                    with open(md_path, "r", encoding="utf-8") as f:
                        md_content = f.read()
                
                csv_choice = prompt_overwrite(csv_path)
                if csv_choice == 'c': break
                elif csv_choice == 'y': convert_md_to_csv(md_path, csv_path, md_content=md_content)
                else: print(f"  [SKIPPED] CSV generation.")
                
            print(f"--- [DONE] {base_name} ---\n")
            
        except Exception as e:
            print(f"--- [ERROR] Failed to process {base_name}. Reason: {e} ---\n")

    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()