import sys
import logging
from pathlib import Path
from collections import defaultdict
import duckdb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import concurrent.futures
from pypdf import PdfReader
from bs4 import BeautifulSoup

# ========== CONFIGURATION ==========
if len(sys.argv) < 2:
    print("Usage: python tfidf_alg.py <batch_file.txt>")
    sys.exit(1)

batch_file = Path(sys.argv[1])
duckdb_path = "/scratch/prj/dh_golden_triangle/recovered/full_data/openalex-snapshot/data/filtered_matches.duckdb"
download_dir = Path("/scratch/prj/dh_golden_triangle/recovered/downloaded")
output_folder = Path("/scratch/prj/dh_golden_triangle/recovered/algorithm_results")
output_folder.mkdir(parents=True, exist_ok=True)

PDF_TIMEOUT = 15

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ========== NLTK SETUP ==========
nltk_path = "/scratch/prj/dh_golden_triangle/scratch_tmp/venv/nltk_data"
nltk.data.path.append(nltk_path)
stop_words = set(stopwords.words("english"))

# ========== TEXT HELPERS ==========
def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    return " ".join([t for t in tokens if t.isalpha() and t not in stop_words])

def parse_authors(auth_str):
    if not isinstance(auth_str, str):
        return set()
    return set(a.strip() for a in auth_str.split(",") if a.strip())

# ========== SAFE TEXT EXTRACTION ==========
def extract_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        return ""
    if os.path.getsize(pdf_path) < 1024:
        return ""

    def _extract():
        try:
            reader = PdfReader(str(pdf_path))
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or "")
            return "\n".join(text)
        except Exception:
            return ""

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_extract)
            return future.result(timeout=PDF_TIMEOUT)
    except concurrent.futures.TimeoutError:
        logging.warning(f"Timeout extracting PDF: {pdf_path}")
        return ""
    except Exception as e:
        logging.warning(f"Failed to extract PDF: {pdf_path}: {e}")
        return ""

def extract_html_text(html_path: Path) -> str:
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""

def extract_fulltext_for_id(openalex_id: str) -> str:
    pdf_path = download_dir / f"{openalex_id}.pdf"
    html_path = download_dir / f"{openalex_id}.html"
    if pdf_path.exists():
        return extract_pdf_text(pdf_path)
    elif html_path.exists():
        return extract_html_text(html_path)
    else:
        return ""

# ========== LOAD GLOBAL DATA ==========
logging.info(f"Loading DuckDB from {duckdb_path}")
con = duckdb.connect(duckdb_path, read_only=True)
df = con.execute("SELECT id, abstract, author_ids, is_data_paper FROM works").fetchdf()

research_papers = df[(df["is_data_paper"] == False) & (df["author_ids"].notna())].copy()
research_papers["parsed_authors"] = research_papers["author_ids"].apply(parse_authors)

# Precompute author → research paper index mapping
author_to_rp = defaultdict(list)
for idx, rp in research_papers.iterrows():
    for a in rp["parsed_authors"]:
        author_to_rp[a].append(idx)

# ========== LOAD BATCH FILE ==========
with open(batch_file, "r") as f:
    data_ids = [line.strip() for line in f if line.strip()]

logging.info(f"Processing {len(data_ids)} data papers from {batch_file}")

# ========== MAIN LOOP ==========
for dp_num, data_paper_id in enumerate(data_ids, start=1):
    data_paper = df[df["id"] == data_paper_id].copy()
    if data_paper.empty:
        logging.warning(f"Data paper {data_paper_id} not found in DB, skipping")
        continue

    dp_id_clean = data_paper_id.replace("https://openalex.org/", "")
    dp_abs = data_paper.iloc[0]["abstract"] or ""
    dp_authors = parse_authors(data_paper.iloc[0]["author_ids"])
    dp_fulltext = extract_fulltext_for_id(dp_id_clean)
    dp_text_full = preprocess(dp_abs + " " + dp_fulltext)

    if not dp_text_full.strip():
        logging.warning(f"No usable text for data paper {dp_id_clean}, skipping")
        continue

    # Find research papers with shared authors
    rp_indices = set()
    for a in dp_authors:
        rp_indices.update(author_to_rp.get(a, []))
    if not rp_indices:
        logging.info(f"No research papers found for {data_paper_id}")
        continue

    overlaps = research_papers.loc[list(rp_indices)]
    logging.info(f"[{dp_num}/{len(data_ids)}] Comparing {dp_id_clean} with {len(overlaps)} research papers")

    # Abstract-based TF-IDF
    dp_abs_text = preprocess(dp_abs)
    rp_abs_texts, rp_ids_abs = [], []
    for rp in overlaps.itertuples():
        rp_id_clean = rp.id.replace("https://openalex.org/", "")
        rp_abs_processed = preprocess(rp.abstract or "")
        if rp_abs_processed:
            rp_abs_texts.append(rp_abs_processed)
            rp_ids_abs.append(rp_id_clean)

    if rp_abs_texts:
        vectorizer_abs = TfidfVectorizer(max_features=30000)
        tfidf_abs_matrix = vectorizer_abs.fit_transform([dp_abs_text] + rp_abs_texts)
        abstract_sims = cosine_similarity(tfidf_abs_matrix[0:1], tfidf_abs_matrix[1:]).flatten()
    else:
        abstract_sims = []

    # Fulltext-based TF-IDF
    rp_full_texts, rp_ids_full = [], []
    for i, rp in enumerate(overlaps.itertuples(), 1):
        rp_id_clean = rp.id.replace("https://openalex.org/", "")
        rp_fulltext = preprocess((rp.abstract or "") + " " + extract_fulltext_for_id(rp_id_clean))
        if rp_fulltext:
            rp_full_texts.append(rp_fulltext)
            rp_ids_full.append(rp_id_clean)
        if i % 200 == 0:
            logging.info(f"Processed {i}/{len(overlaps)} research papers for full text...")

    if rp_full_texts:
        vectorizer_full = TfidfVectorizer(max_features=30000)
        tfidf_full_matrix = vectorizer_full.fit_transform([dp_text_full] + rp_full_texts)
        fulltext_sims = cosine_similarity(tfidf_full_matrix[0:1], tfidf_full_matrix[1:]).flatten()
    else:
        fulltext_sims = []

    # Merge results
    all_rp_ids = sorted(set(rp_ids_abs) | set(rp_ids_full))
    results = []
    for rp_id in all_rp_ids:
        idx_abs = rp_ids_abs.index(rp_id) if rp_id in rp_ids_abs else None
        idx_full = rp_ids_full.index(rp_id) if rp_id in rp_ids_full else None
        results.append({
            "data_openalex_id": dp_id_clean,
            "research_openalex_id": rp_id,
            "tfidf_abstract_cosine": float(abstract_sims[idx_abs]) if idx_abs is not None else 0.0,
            "tfidf_fulltext_cosine": float(fulltext_sims[idx_full]) if idx_full is not None else 0.0,
        })

    out_path = output_folder / f"{dp_id_clean}_tfidf.csv"
    pd.DataFrame(results).sort_values(by="tfidf_fulltext_cosine", ascending=False).to_csv(out_path, index=False)
    logging.info(f"Saved {len(results)} matches for {dp_id_clean} to {out_path}")

