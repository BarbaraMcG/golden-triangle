import duckdb
import pandas as pd
import logging
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from pypdf import PdfReader
import pdfplumber
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import sys
import nltk

# ==== CONFIGURATION ====
if len(sys.argv) < 2:
    print("Usage: python match_pipeline.py <batch_file.txt>")
    sys.exit(1)

batch_file = Path(sys.argv[1])

duckdb_path = "/scratch/prj/dh_golden_triangle/recovered/full_data/openalex-snapshot/data/filtered_matches.duckdb"
download_dir = Path("/scratch/prj/dh_golden_triangle/recovered/downloaded")

output_folder = Path("/scratch/prj/dh_golden_triangle/recovered/algorithm_results")
output_folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# NLTK setup
nltk_path = "/scratch/prj/dh_golden_triangle/scratch_tmp/venv/nltk_data"
nltk.data.path.append(nltk_path)
stop_words = set(stopwords.words("english"))

# ==== HELPERS ====
def preprocess(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

def parse_authors(auth_str):
    if not isinstance(auth_str, str):
        return set()
    return set(a.strip() for a in auth_str.split(",") if a.strip())

def extract_text_from_pdf(pdf_path):
    """Try pdfplumber first (better at text layout), fallback to pypdf."""
    try:
        if Path(pdf_path).stat().st_size == 0:
            return ""

        # Try pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text_parts = [page.extract_text() or "" for page in pdf.pages]
            text = " ".join(text_parts).strip()
            if text:
                return text

        # Fallback: pypdf
        reader = PdfReader(str(pdf_path))
        text_parts = [page.extract_text() or "" for page in reader.pages]
        return " ".join(text_parts).strip()

    except Exception as e:
        logging.warning(f"Failed to extract PDF text from {pdf_path}: {e}")
        return ""

def extract_text_from_html(html_path):
    try:
        with open(html_path, "rb") as f:
            raw = f.read()
        soup = BeautifulSoup(raw, "html.parser")
        return soup.get_text(separator=" ")
    except Exception as e:
        logging.warning(f"Failed to extract HTML text from {html_path}: {e}")
        return ""

def load_full_text(openalex_id):
    pdf_path = download_dir / f"{openalex_id}.pdf"
    html_path = download_dir / f"{openalex_id}.html"
    if pdf_path.exists():
        return extract_text_from_pdf(pdf_path)
    elif html_path.exists():
        return extract_text_from_html(html_path)
    else:
        return ""

# ==== SIMILARITY FUNCTIONS ====
def jaccard(tokens1, tokens2):
    """Standard Jaccard similarity (set-based)."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def multiset_jaccard(tokens1, tokens2):
    """Multiset Jaccard similarity (frequency-based)."""
    c1, c2 = Counter(tokens1), Counter(tokens2)
    all_keys = set(c1.keys()) | set(c2.keys())
    intersection = sum(min(c1[k], c2[k]) for k in all_keys)
    union = sum(max(c1[k], c2[k]) for k in all_keys)
    return intersection / union if union > 0 else 0.0

def top_n_tokens(tokens, n):
    """Return the top-N most frequent tokens."""
    if not tokens:
        return []
    freq = Counter(tokens).most_common(n)
    return [t for t, _ in freq]

def compute_similarities(dp_text, rp_text):
    dp_tokens, rp_tokens = preprocess(dp_text), preprocess(rp_text)

    results = {}

    # --- All tokens ---
    results["jaccard_all"] = jaccard(dp_tokens, rp_tokens)
    results["multiset_jaccard_all"] = multiset_jaccard(dp_tokens, rp_tokens)

    # --- Top-N tokens ---
    for n in [10, 20, 50]:
        dp_top, rp_top = top_n_tokens(dp_tokens, n), top_n_tokens(rp_tokens, n)
        results[f"jaccard_top{n}"] = jaccard(dp_top, rp_top)
        results[f"multiset_jaccard_top{n}"] = multiset_jaccard(dp_top, rp_top)

    return results

# ==== PAIRWISE COMPARISON ====
def compare_pair(dp, rp):
    dp_id = dp["id"].replace("https://openalex.org/", "")
    rp_id = rp["id"].replace("https://openalex.org/", "")

    dp_abs = dp["abstract"] or ""
    rp_abs = rp["abstract"] or ""
    dp_full = load_full_text(dp_id)
    rp_full = load_full_text(rp_id)

    abs_scores = compute_similarities(dp_abs, rp_abs)
    full_scores = compute_similarities(dp_full, rp_full)

    return {
        "data_openalex_id": dp_id,
        "research_openalex_id": rp_id,
        **{f"abstract_{k}": v for k, v in abs_scores.items()},
        **{f"fulltext_{k}": v for k, v in full_scores.items()},
    }

# ==== MAIN PIPELINE ====
logging.info(f"Processing batch file: {batch_file}")

with open(batch_file, "r") as f:
    data_ids = [line.strip() for line in f if line.strip()]

logging.info("Loading data from DuckDB...")
con = duckdb.connect(duckdb_path, read_only=True)
df = con.execute("SELECT id, abstract, author_ids, is_data_paper FROM works").fetchdf()

data_papers = df[df["id"].isin(data_ids)]
research_papers = df[(~df["id"].isin(data_ids)) & (df["is_data_paper"] == False)]
research_papers = research_papers[research_papers["author_ids"].notna()].copy()
research_papers["parsed_authors"] = research_papers["author_ids"].apply(parse_authors)

# Precompute author → research paper mapping
author_to_rp = defaultdict(list)
for idx, rp in research_papers.iterrows():
    for author in rp["parsed_authors"]:
        author_to_rp[author].append(idx)

MAX_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

for dp_idx, (_, dp) in enumerate(data_papers.iterrows()):
    dp_authors = parse_authors(dp["author_ids"])
    rp_indices = set()
    for a in dp_authors:
        rp_indices.update(author_to_rp.get(a, []))
    if not rp_indices:
        logging.info(f"Data paper {dp_idx+1}/{len(data_papers)} {dp['id']} → no overlaps")
        continue

    overlaps = research_papers.loc[list(rp_indices)]
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(compare_pair, dp, rp) for _, rp in overlaps.iterrows()]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logging.error(f"Error comparing pair: {e}")

    logging.info(f"Data paper {dp_idx+1}/{len(data_papers)} {dp['id']} → compared with {len(overlaps)} research papers")

    # Save per data paper CSV
    dp_id_clean = dp["id"].replace("https://openalex.org/", "")
    output_path = output_folder / f"{dp_id_clean}_jaccard.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    logging.info(f"Saved {len(results)} matches to {output_path}")
