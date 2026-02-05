"""
Preprocess DiffusionDB dataset to create CSV file with primary and secondary prompts.
"""

import os
import csv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.llm_interface import LLMInterface


def preprocess_diffusion_db(input_file: str = "data/diffusion_prompts.txt", 
                            output_file: str = "data/diffusion_db.csv",
                            num_rows: int = 1000,
                            batch_size: int = 10,
                            start_from_row: int = None):
    """
    Preprocess DiffusionDB prompts into CSV format.
    
    Args:
        input_file: Path to input text file (one prompt per line)
        output_file: Path to output CSV file
        num_rows: Number of rows to process (default: 1000)
        batch_size: Number of prompts to process in each LLM batch (default: 10)
        start_from_row: Row index to start processing from. If None, process from beginning and write new CSV.
                       If set, continue from that row and append to existing CSV.
    """
    # Check if output file exists
    file_exists = os.path.exists(output_file)
    append_mode = file_exists and start_from_row is not None
    
    # Read prompts from input file
    print(f"Reading prompts from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    
    # Determine starting point
    if start_from_row is not None:
        if start_from_row >= len(all_prompts):
            print(f"start_from_row ({start_from_row}) exceeds total prompts ({len(all_prompts)}). Nothing to process.")
            return
        prompts_to_process = all_prompts[start_from_row:start_from_row + num_rows]
        print(f"Starting from row {start_from_row}, processing {len(prompts_to_process)} prompts in batches of {batch_size}...")
    else:
        prompts_to_process = all_prompts[:num_rows]
        print(f"Processing {len(prompts_to_process)} prompts in batches of {batch_size}...")
    
    # Initialize LLM interface
    llm_interface = LLMInterface()
    
    # Prepare CSV file
    if append_mode:
        print(f"Appending to existing {output_file}...")
        # Read existing header to get max secondaries
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            existing_header = next(reader)
            max_secondaries_existing = len(existing_header) - 1  # minus primary column
        file_mode = 'a'
        header = existing_header
    else:
        print(f"Writing to {output_file}...")
        max_secondaries_existing = 0
        file_mode = 'w'
        header = None  # Will be set after first batch
    
    # Open CSV file for writing
    csv_file = open(output_file, file_mode, newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    
    # Write header only if creating new file
    if not append_mode:
        # We'll write header after first batch when we know max secondaries
        pass
    
    # Process in batches and write immediately
    processed_count = 0
    max_secondaries = max_secondaries_existing
    
    try:
        for i in range(0, len(prompts_to_process), batch_size):
            batch = prompts_to_process[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1} ({len(batch)} prompts)...")
            
            # Decompose batch
            batch_results = llm_interface.decompose_prompts_batch(batch)
            
            # Process results for this batch
            batch_valid_results = []
            for original_prompt, decomposition in batch_results.items():
                if decomposition.get("ignore", False):
                    continue  # Skip prompts with no secondaries
                
                primary = decomposition.get("primary", "")
                secondaries = decomposition.get("secondaries", [])
                
                if primary and secondaries:
                    batch_valid_results.append({
                        "primary": primary,
                        "secondaries": secondaries
                    })
            
            # Update max secondaries
            if batch_valid_results:
                max_secondaries_batch = max(len(r["secondaries"]) for r in batch_valid_results)
                max_secondaries = max(max_secondaries, max_secondaries_batch)
                
                # Write header if this is the first batch and creating new file
                if not append_mode and header is None:
                    header = ["primary"] + [f"secondary_{i+1}" for i in range(max_secondaries)]
                    writer.writerow(header)
                
                # Write batch results immediately
                for result in batch_valid_results:
                    row = [result["primary"]] + result["secondaries"]
                    # Pad with empty strings if needed
                    while len(row) < len(header):
                        row.append("")
                    writer.writerow(row)
                    processed_count += 1
                
                # Flush to ensure data is written
                csv_file.flush()
            
            print(f"  Processed {processed_count} valid prompts so far (wrote {len(batch_valid_results)} in this batch)...")
    
    finally:
        csv_file.close()
    
    print(f"\n✓ Total valid prompts written: {processed_count}")
    if not append_mode:
        print(f"  Max secondaries per prompt: {max_secondaries}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess DiffusionDB dataset")
    parser.add_argument("--input", type=str, default="data/diffusion_prompts.txt",
                        help="Input text file path")
    parser.add_argument("--output", type=str, default="data/diffusion_db.csv",
                        help="Output CSV file path")
    parser.add_argument("--num_rows", type=int, default=1000,
                        help="Number of rows to process")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for LLM processing")
    parser.add_argument("--start_from_row", type=int, default=None,
                        help="Row index to start processing from. If None, process from beginning. If set, continue from that row and append to existing CSV.")
    
    args = parser.parse_args()
    
    preprocess_diffusion_db(
        input_file=args.input,
        output_file=args.output,
        num_rows=args.num_rows,
        batch_size=args.batch_size,
        start_from_row=args.start_from_row
    )
