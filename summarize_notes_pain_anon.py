#!/usr/bin/env python3

import pandas as pd
import os
import sys
import time
import requests
from tqdm import tqdm

# ============================================================================
# Azure OpenAI Configuration - GPT-4.1
# ============================================================================
AZURE_API_KEY = ""## PLEASE PUT API HERE
AZURE_ENDPOINT = ""
DEPLOYMENT_NAME = "gpt-4.1"
API_VERSION = "2024-05-01-preview"

# ============================================================================
# Summarization Settings
# ============================================================================
DELAY_BETWEEN_CALLS = 0.5  
MAX_NOTE_CHARS = 3000      


# ============================================================================
# Pain-Specific System Prompt
# ============================================================================
PAIN_SYSTEM_PROMPT = """You are a clinician extracting pain-related information from clinical notes. Your task is to produce a concise, structured summary focusing EXCLUSIVELY on pain-relevant data. 

**EXTRACTION RULES:** 

1. Extract ONLY information related to pain.  

2. Preserve clinical specificity: include exact pain scores, medication names, doses, routes, frequencies, anatomical locations, and pain descriptors. 

3. Explicitly capture NEGATIONS (e.g., "denies pain", "pain free", "no complaints of discomfort", "pain score 0/10") — these are as important as positive findings. 

4. Capture CHANGES in pain over time if documented (e.g., "pain improved from 8/10 to 4/10 after morphine", "pain worsening over shift"). 

5. If the note contains NO pain-relevant information at all, provide a single-line general clinical summary of the patient's status instead. Format this as: 

GENERAL SUMMARY: [One sentence describing the patient's overall clinical status from the note.] 

**STRUCTURED OUTPUT FORMAT (use these exact headers when pain data IS found):** 
 
PAIN REPORT: [Direct patient reports of pain. Note if pain is ABSENT or explicitly NEGATED.] 

PAIN CHARACTERISTICS: [Location (specific anatomical site), quality (sharp, dull, burning, aching, throbbing, stabbing, cramping), onset/duration, aggravating/alleviating factors. State "Not described" if characteristics not documented.] 

SEVERITY/INTENSITY: [Report pain severity as a format of 1–5 Likert scale (1 = severe, 2 = substantial, 3 = moderate, 4 = mild, 5 = none/no symptoms)] 

PHARMACOLOGICAL TREATMENT: [Pain medications — include drug names, doses, routes (PO/IV/IM/epidural/PCA/topical/patch), frequency, PRN vs scheduled. Note new orders, dose changes, medication switches, or discontinuations. Include patient refusal of pain medication if documented.] 

FUNCTIONAL IMPACT: [How pain affects mobility, ambulation, sleep, appetite, breathing (splinting), participation in therapy, ADLs (activities of daily living), mood. Note if pain is limiting recovery or discharge readiness.] 

BEHAVIORAL INDICATORS: [Observable signs —restlessness, reluctance to move, crying, moaning, panting, facial tension, body positioning changes. Especially important when patient cannot self-report.] 

PATIENT RESPONSE: [Response to pain interventions — did pain improve, worsen, or remain unchanged after treatment? Include re-assessment scores if documented (e.g., "pain decreased from 8/10 to 3/10 thirty minutes after IV morphine")] 

**IMPORTANT:** 

- Keep each section to 1-2 sentences maximum 

- Use clinical abbreviations where standard (VAS, PCA, PRN, PO, IV, NSAID, etc.) 

- If a section has denies of pain in the note, write “denies pain” for that section 

- If there is no pain documented, write “no pain documented” for that section 

- Do NOT fabricate or infer information not present in the note 

- NEVER return an empty response. If no pain data exists, you MUST provide the GENERAL SUMMARY fallback line.""" 

def summarize_note_pain(note_text, max_retries=3, retry_delay=5):
    """
    Summarize a clinical note with pain-specific focus using GPT-4.1.
    
    Args:
        note_text (str): The clinical note to summarize
        max_retries (int): Maximum retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        str: Pain-focused structured summary or error message
    """
    
    if pd.isna(note_text) or str(note_text).strip() == '':
        return None
    
    # Convert to string and truncate if too long
    note_text = str(note_text)
    
    if len(note_text) > MAX_NOTE_CHARS:
        note_text = note_text[:MAX_NOTE_CHARS]
        # Try to cut at a sentence boundary
        last_period = note_text.rfind('.')
        if last_period > MAX_NOTE_CHARS * 0.8:
            note_text = note_text[:last_period + 1]
        note_text = note_text + " [Note truncated due to length]"
    
    # Build API URL
    api_url = f"{AZURE_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
    }
    
    # User message with the clinical note
    user_message = f"""Extract all pain-related information from the following clinical note. Follow the structured output format exactly. If no pain information is found, provide a 1-line general clinical summary instead.

**Clinical Note:**
{note_text}"""
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": PAIN_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "temperature": 0.1,   # Low temperature for consistent, factual extraction
        "max_tokens": 500,    # Enough for structured summary across all sections
        "top_p": 1.0,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=90  # Longer timeout for GPT-4.1
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    summary = result['choices'][0]['message']['content'].strip()
                    return summary
                else:
                    return "ERROR: No summary generated"
            
            elif response.status_code == 429:
                wait_time = retry_delay * (attempt + 1)
                print(f"    Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            else:
                error_msg = f"ERROR: HTTP {response.status_code}"
                error_detail = response.text[:200] if response.text else "No details"
                if attempt < max_retries - 1:
                    print(f"    {error_msg} - {error_detail}. Retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return f"{error_msg} - {error_detail}"
                    
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"    Timeout. Retrying...")
                time.sleep(retry_delay)
                continue
            else:
                return "ERROR: Request timeout"
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Error: {e}. Retrying...")
                time.sleep(retry_delay)
                continue
            else:
                return f"ERROR: {str(e)}"
    
    return "ERROR: Max retries exceeded"


def summarize_notes_file(input_file, output_file=None, notes_column='Notes'):
    """
    Summarize all notes in a CSV file with pain-specific extraction.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        notes_column (str): Name of the column containing notes
    """
    
    print("=" * 80)
    print("PAIN-SPECIFIC CLINICAL NOTES SUMMARIZATION")
    print("Azure OpenAI GPT-4.1")
    print("=" * 80)
    print(f"\nInput file: {input_file}")
    print(f"Notes column: {notes_column}")
    print(f"Model: {DEPLOYMENT_NAME}")
    print(f"Endpoint: {AZURE_ENDPOINT}")
    print(f"API Version: {API_VERSION}")
    print(f"Max note chars: {MAX_NOTE_CHARS:,}\n")
    
    # Read input file
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df):,} rows")
        print(f"  Columns: {', '.join(df.columns.tolist())}")
    except Exception as e:
        print(f"✗ ERROR reading file: {e}")
        sys.exit(1)
    
    # Verify notes column exists
    if notes_column not in df.columns:
        print(f"\n✗ ERROR: Column '{notes_column}' not found!")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Try to find a notes column
        possible_cols = [col for col in df.columns if 'note' in col.lower()]
        if possible_cols:
            print(f"\nPossible notes columns: {possible_cols}")
            notes_column = possible_cols[0]
            print(f"Using: {notes_column}")
        else:
            sys.exit(1)
    
    # Count notes to process
    notes_to_process = df[notes_column].notna().sum()
    notes_empty = df[notes_column].isna().sum()
    print(f"\nNotes to summarize: {notes_to_process:,} (out of {len(df):,} rows)")
    print(f"Empty/missing notes: {notes_empty:,}")
    
    if notes_to_process == 0:
        print("\n✗ ERROR: No notes to summarize!")
        sys.exit(1)
    
    # Note length statistics
    note_lengths = df[df[notes_column].notna()][notes_column].astype(str).str.len()
    print(f"\nNote length statistics:")
    print(f"  Min: {note_lengths.min():,} chars")
    print(f"  Max: {note_lengths.max():,} chars")
    print(f"  Mean: {note_lengths.mean():,.0f} chars")
    print(f"  Notes > {MAX_NOTE_CHARS:,} chars (will be truncated): {(note_lengths > MAX_NOTE_CHARS).sum():,}")
    
    # Create Summary column
    df['Summary'] = None
    
    # Process each note
    print(f"\nCalling Azure OpenAI GPT-4.1 for pain-specific extraction...")
    print(f"Estimated time: ~{notes_to_process * (DELAY_BETWEEN_CALLS + 2):.0f} seconds")
    print(f"This may take a while...\n")
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    truncated_count = 0
    general_summary_count = 0
    
    for idx in tqdm(df.index, desc="Summarizing", unit="note"):
        note = df.at[idx, notes_column]
        
        # Skip empty notes
        if pd.isna(note) or str(note).strip() == '':
            skipped_count += 1
            continue
        
        # Track truncation
        if len(str(note)) > MAX_NOTE_CHARS:
            truncated_count += 1
        
        # Call API
        summary = summarize_note_pain(note)
        
        # Store result
        df.at[idx, 'Summary'] = summary
        
        # Track statistics
        if summary is None:
            skipped_count += 1
        elif summary.startswith("ERROR"):
            error_count += 1
        elif summary.strip().startswith("GENERAL SUMMARY:"):
            general_summary_count += 1
            success_count += 1  # Still a valid response
        else:
            success_count += 1
        
        # Delay to avoid rate limits
        time.sleep(DELAY_BETWEEN_CALLS)
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_summarized.csv"
    
    # Save results
    try:
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved: {output_file}")
    except Exception as e:
        print(f"\n✗ ERROR saving file: {e}")
        sys.exit(1)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Model: {DEPLOYMENT_NAME}")
    print(f"Total rows: {len(df):,}")
    print(f"Notes processed: {notes_to_process:,}")
    print(f"Notes truncated (>{MAX_NOTE_CHARS:,} chars): {truncated_count:,}")
    print(f"Successful summaries: {success_count:,}")
    print(f"  - With pain data: {success_count - general_summary_count:,}")
    print(f"  - No pain data (general summary fallback): {general_summary_count:,}")
    print(f"Errors: {error_count:,}")
    print(f"Skipped (no notes): {skipped_count:,}")
    
    if success_count > 0:
        coverage = (success_count / notes_to_process) * 100
        pain_rate = ((success_count - general_summary_count) / notes_to_process) * 100
        print(f"\nSummary coverage: {coverage:.1f}%")
        print(f"Notes with pain data: {pain_rate:.1f}%")
        print(f"\n✓ Pain-specific summarization completed successfully!")
    else:
        print(f"\n⚠ WARNING: No summaries were generated successfully")
    
    print("=" * 80)
    
    # Show sample summaries
    if success_count > 0:
        print("\nSAMPLE SUMMARIES (First 3 with pain data):")
        print("=" * 80)
        
        has_summary = df['Summary'].notna()
        not_error = df['Summary'].apply(
            lambda x: not str(x).startswith('ERROR') if pd.notna(x) else False
        )
        has_pain = df['Summary'].apply(
            lambda x: not str(x).strip().startswith('GENERAL SUMMARY:') if pd.notna(x) else False
        )
        sample_df = df[has_summary & not_error & has_pain].head(3)
        
        if len(sample_df) == 0:
            # Fall back to any successful summary
            sample_df = df[has_summary & not_error].head(3)
        
        for i, (idx, row) in enumerate(sample_df.iterrows(), 1):
            print(f"\n--- Sample {i} ---")
            if 'MRN' in df.columns:
                print(f"MRN: {row.get('MRN', 'N/A')}")
            if 'EncounterID' in df.columns:
                print(f"EncounterID: {row.get('EncounterID', 'N/A')}")
            
            print(f"Original note length: {len(str(row[notes_column])):,} characters")
            print(f"\nPain Summary:")
            print(row['Summary'])
            print("-" * 80)


if __name__ == "__main__":
    # Get input file
    if len(sys.argv) > 1:
        input_file_path = sys.argv[1]
    else:
        # Auto-detect *_notes.csv file in current directory
        notes_files = [f for f in os.listdir('.') if f.endswith('_notes.csv') and os.path.isfile(f)]
        
        if not notes_files:
            print("✗ ERROR: No *_notes.csv file found in current directory")
            print(f"\nUsage: python {sys.argv[0]} <input_file.csv> [output_file.csv] [notes_column_name]")
            print(f"Example: python {sys.argv[0]} NOC_PainLevel_EncounterSummary_notes.csv")
            sys.exit(1)
        
        if len(notes_files) > 1:
            print(f"Found multiple files: {notes_files}")
            print(f"Using: {notes_files[0]}")
        
        input_file_path = notes_files[0]
    
    # Get output file (optional)
    output_file_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Get notes column name (optional)
    notes_col_name = sys.argv[3] if len(sys.argv) > 3 else 'Notes'
    
    # Check if input file exists
    if not os.path.exists(input_file_path):
        print(f"✗ ERROR: File not found: {input_file_path}")
        sys.exit(1)
    
    # Run summarization
    summarize_notes_file(input_file_path, output_file_path, notes_col_name)
