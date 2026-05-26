import duckdb
import os
import json
from tqdm import tqdm

# Database paths
source_db_path = '/hpc/scratch/prj/dh_golden_triangle/recovered/full_data/openalex-snapshot/data/combined.duckdb'
target_db_path = '/hpc/scratch/prj/dh_golden_triangle/recovered/full_data/openalex-snapshot/data/evaluation.duckdb'

# Input files
data_paper_ids_file = '/hpc/scratch/prj/dh_golden_triangle/recovered/data_paper_ids_2022.txt'
research_papers_file = '/hpc/scratch/prj/dh_golden_triangle/recovered/research_papers_urls_related_to_2022.txt'

def re_invert_abstract(inverted_index):
    if inverted_index == None:
        return ''
    
    if len([index for indices in inverted_index.values() for index in indices]) == 0:
        return ''
    max_index = max([index for indices in inverted_index.values() for index in indices])
    text_list = [""] * (max_index + 1)
    for word, indices in inverted_index.items():
        for index in indices:
            text_list[index] = word
    text = " ".join(text_list)
    return text

def load_target_ids():
    """Load target IDs from both files"""
    print("Loading target IDs...")
    
    # Load data paper IDs (is_data_paper = True)
    data_paper_ids = set()
    with open(data_paper_ids_file, 'r') as f:
        for line in f:
            id_ = line.strip()
            if id_:
                data_paper_ids.add(id_)
    
    # Load research paper IDs (is_data_paper = False)
    research_paper_ids = set()
    with open(research_papers_file, 'r') as f:
        for line in f:
            # Extract ID (everything before first comma)
            id_ = line.split(',')[0].strip()
            if id_:
                research_paper_ids.add(id_)
    
    print(f"Loaded {len(data_paper_ids):,} data paper IDs")
    print(f"Loaded {len(research_paper_ids):,} research paper IDs")
    
    return data_paper_ids, research_paper_ids

def create_evaluation_table(target_conn):
    """Create the evaluation table structure"""
    target_conn.execute("""
        CREATE TABLE IF NOT EXISTS works (
            id VARCHAR,
            title VARCHAR,
            doi VARCHAR,
            publication_year INTEGER,
            source_display_name VARCHAR,
            urls VARCHAR,
            abstract VARCHAR,
            author_ids VARCHAR,
            is_data_paper BOOLEAN
        )
    """)

def process_chunks(data_paper_ids, research_paper_ids):
    """Process data in chunks to handle memory limitations"""
    
    # All target IDs
    all_target_ids = data_paper_ids | research_paper_ids
    
    # First, get the total count for progress bar
    source_conn = duckdb.connect(source_db_path, read_only=True)
    count_query = "SELECT COUNT(*) FROM works"
    total_rows = source_conn.execute(count_query).fetchone()[0]
    source_conn.close()
    print(f"Total rows to process: {total_rows:,}")
    
    # Remove existing evaluation database
    if os.path.exists(target_db_path):
        print(f'Removing existing evaluation database...')
        os.remove(target_db_path)
    
    # Create target database connection
    target_conn = duckdb.connect(target_db_path)
    create_evaluation_table(target_conn)
    
    # Query to get all works
    query = """
    SELECT id, title, doi, publication_year, source_display_name, urls, abstract_inverted_index, author_ids
    FROM works
    """
    
    chunk_size = 100_000  # Process 100k rows at a time
    offset = 0
    found_papers = 0
    
    # Create progress bar
    with tqdm(total=total_rows, desc="Processing rows", unit="rows") as pbar:
        
        while True:
            # Create new connection for each chunk to prevent memory accumulation
            source_conn = duckdb.connect(source_db_path, read_only=True)
            
            # Get chunk of data
            chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
            result = source_conn.execute(chunk_query).fetchall()
            
            # Close connection immediately after getting data
            source_conn.close()
            
            # If no more data, break
            if not result:
                break
            
            # Process each row in the chunk
            rows_to_insert = []
            for row in result:
                id_, title, doi, publication_year, source_display_name, urls, abstract_inverted_index, author_ids = row
                
                # Check if this ID is in our target sets
                if id_ in all_target_ids:
                    # Determine if it's a data paper
                    is_data_paper = id_ in data_paper_ids
                    
                    # Convert abstract_inverted_index to abstract
                    abstract = ''
                    if abstract_inverted_index:
                        try:
                            inverted_index = json.loads(abstract_inverted_index)
                            abstract = re_invert_abstract(inverted_index)
                        except (json.JSONDecodeError, Exception):
                            abstract = ''
                    
                    rows_to_insert.append((
                        id_, title, doi, publication_year, source_display_name, 
                        urls, abstract, author_ids, is_data_paper
                    ))
                    found_papers += 1
            
            # Insert batch of rows
            if rows_to_insert:
                target_conn.executemany("""
                    INSERT INTO works (id, title, doi, publication_year, source_display_name, 
                                     urls, abstract, author_ids, is_data_paper)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows_to_insert)
            
            # Update progress bar
            pbar.update(len(result))
            
            # Move to next chunk
            offset += chunk_size
    
    # Close target connection
    target_conn.close()
    
    print(f"Processing complete. Found {found_papers:,} papers for evaluation.")
    print(f"Evaluation database created at {target_db_path}")

if __name__ == "__main__":
    # Load the target IDs
    data_paper_ids, research_paper_ids = load_target_ids()
    
    # Process papers to create evaluation database
    process_chunks(data_paper_ids, research_paper_ids)
