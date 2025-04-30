import os
import gzip
import json
import shutil
from pathlib import Path

# Define paths
source_file = Path("/scratch/prj/dh_golden_triangle/scratch_tmp/filtered_results_2022.jsonl")
# source_file = Path("/scratch_tmp/prj/dh_golden_triangle/sample.jsonl")
output_file = Path("/scratch/prj/dh_golden_triangle/scratch_tmp/some_data_papers.csv")


def process_single_year_jsonl():
    num_found = 0
    max_found = 1000
    with source_file.open('r') as inf, output_file.open('w+') as outf:
        outf.write("title,doi,journal\n")
        for linei, line in enumerate(inf):
            # if linei == 1000: break # smoke test
            try:        
                work = json.loads(line.strip())                
                if work and work['gt_is_data_journal']:
                    outf.write(f"\"{work['title']}\",{work['doi']},\"{work['primary_location']['source']['display_name']}\"\n")
                    num_found += 1
                    print(f'Found so far: {num_found} / {max_found}\r', end='')                    
                    if num_found == max_found:
                        break
            except Exception as e:
                print(e)
                continue
                # raise
        
# Run the processing function
if __name__ == "__main__":
    process_single_year_jsonl()
    print(f"Processing complete. Results saved in {output_file}.")
