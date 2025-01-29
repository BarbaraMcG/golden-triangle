import os
import gzip
import json
import shutil
from pathlib import Path

# Define paths
source_file = Path("/scratch_tmp/prj/dh_golden_triangle/results_2022.jsonl")
# source_file = Path("/scratch_tmp/prj/dh_golden_triangle/sample.jsonl")
output_file = Path("/scratch_tmp/prj/dh_golden_triangle/filtered_results_2022.jsonl")
DATA_JOURNALS_FILE = Path(__file__).parent / Path('data_journals.txt')

DATA_JOURNALS = [x.strip() for x in DATA_JOURNALS_FILE.read_text().splitlines()]

def get_single_jsonl_output(work: dict):
    if (work['type'] not in ['article', 'book-chapter', 'preprint'] or
        work['primary_location'] is None or
        work['primary_location']['source'] is None or
        not work['primary_location']['source']['is_oa'] or 
        work['language'] != 'en'):
        return None
    
    work['gt_is_data_journal'] = (work['primary_location']['source']['display_name'].lower()
            in DATA_JOURNALS)
    return work
    

def process_single_year_jsonl():
    with source_file.open('r') as inf, output_file.open('w+') as outf:
        for line in inf:
            try:        
                work = get_single_jsonl_output(json.loads(line.strip()))
                if work:
                    outf.write(line)
            except Exception as e:
                print(e)
                continue
                # raise
        
# Run the processing function
if __name__ == "__main__":
    process_single_year_jsonl()
    print("Processing complete. Filtered results saved in /scratch_tmp/prj/dh_golden_triangle/filtered_results_2022.jsonl.")
