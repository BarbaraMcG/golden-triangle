import gzip
import json
from pathlib import Path

def read_jsonl_gz(fn: Path) -> list:
    objs = []
    with gzip.open(fn, 'rt') as f:
        for line in f:
            objs.append(json.loads(line))
    return objs

def write_jsonl(objs: list, fn: Path):
     with fn.open('w') as f:
        for obj in objs:
            f.write(json.dumps(obj) + '\n')


openalex_data_dir = Path('data/openalex-full-data/data/works/')
filtered_data_dir = Path('data/openalex-filtered')
filtered_data_dir.mkdir(parents=True, exist_ok=True)

def save_filtered(filtered):
    existing_files = sorted(filtered_data_dir.glob('*.json'))
    if existing_files:
        last_file = existing_files[-1].stem
        next_number = int(last_file) + 1
    else:
        next_number = 1

    next_file_name = f'{next_number:03}.json'
    write_jsonl(filtered, filtered_data_dir / next_file_name)


filtered = []
gz_files = list(openalex_data_dir.rglob('*.gz'))
num_gz_files = len(gz_files)
for gz_file_i, gz_file in enumerate(gz_files):
    works = read_jsonl_gz(gz_file)
    # is_oa:true,publication_year:2023,type:article|book-chapter,language:en
    for work_i, work in enumerate(works):
        if (work.get('open_access', {}).get('is_oa', False) and
            work.get('publication_year', -1) == 2022 and
            work.get('type', None) in ['article', 'book-chapter'] and
            work.get('language', None) == 'en' and 
            work.get('abstract_inverted_index', None) is not None
            # ... concepts ...
        ):
            work2 = {}
            work2['display_name'] = work.get('display_name', None)
            work2['doi'] = work.get('doi', None)
            work2['gt_file_path'] = str(gz_file)
            work2['gt_line_no'] = work_i
            filtered.append(work2)
            if len(filtered) == 100:
                print(f'{gz_file_i}/{num_gz_files}')
                save_filtered(filtered)
                filtered = []

save_filtered(filtered)
filtered = []