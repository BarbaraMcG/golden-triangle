import os
import gzip
import json
import shutil
from pathlib import Path

# Define paths
source_dir = Path("/scratch_tmp/prj/dh_golden_triangle/full_data/openalex-snapshot/data/works/")
destination_dir = Path("/dev/shm/tmp_gt")
destination_dir.mkdir(parents=True, exist_ok=True)

# Function to filter JSONL lines directly from a .gz file
def filter_jsonl_gz(gz_file_path, output_file, folder, gz_name):
    with gzip.open(gz_file_path, 'rt') as infile, open(output_file, 'a') as outfile:
        for line in infile:
            try:
                record = json.loads(line)
                publication_date = record.get("publication_date")
                publication_year = record.get("publication_year")
                oa_status = record.get("open_access", {}).get("oa_status")

                # Check the conditions
                if (publication_date and "2022" in publication_date) or publication_year == 2022:
                    if oa_status != "closed":
                        # Add metadata fields
                        record["source_folder"] = str(folder)
                        record["source_gz"] = str(gz_name)
                        outfile.write(json.dumps(record) + "\n")
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file {gz_file_path}, skipping line.")

# DFS traversal and processing
def process_gz_files():
    output_filtered_file = "/scratch_tmp/prj/dh_golden_triangle/filtered_results.jsonl"
    all_gz_files = list(source_dir.rglob("*.gz"))
    for i, gz_file in enumerate(all_gz_files):
        print(f'{i}/{len(all_gz_files)}')
        dest_path = destination_dir / gz_file.name
        shutil.copy(gz_file, dest_path)
        filter_jsonl_gz(dest_path, output_filtered_file, gz_file.parent, gz_file.name)

# Run the processing function
if __name__ == "__main__":
    process_gz_files()
    print("Processing complete. Filtered results saved in /scratch_tmp/prj/dh_golden_triangle/filtered_results.jsonl.")
