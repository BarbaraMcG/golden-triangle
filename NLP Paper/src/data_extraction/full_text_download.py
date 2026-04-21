import csv
import json
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging

# ==== CONFIGURATION ====
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

base_path = Path("/scratch/prj/dh_golden_triangle/recovered")
input_file = base_path / "research_papers_urls_related_to_2022.txt" # List of OpenAlex Ids to download, extracted from duckdb
download_folder = base_path / "downloaded"
failed_log_file = base_path / "failed_downloads.jsonl"
checkpoint_file = base_path / "download_checkpoint.json"
results_file = base_path / "download_results.jsonl"

download_folder.mkdir(parents=True, exist_ok=True)

MAX_RUNTIME_SECONDS = 23 * 3600 + 50 * 60  # 23h50m
MAX_RETRIES = 1
CONCURRENT_DOWNLOADS = 10

# ==== STATS ====
stats = {
    "total_rows": 0,
    "ignored_rows": 0,
    "valid_rows": 0,
    "successful_downloads": 0,
    "failed_downloads": 0,
    "pdf_downloads": 0,
    "html_downloads": 0,
    "skipped_existing": 0,
    "session_start_time": time.time(),
}

processed_rows = set()

# ==== HELPERS ====
def openalex_to_filename(openalex_url: str):
    """Extract OpenAlex ID (W123...)"""
    if not openalex_url or "openalex.org/" not in openalex_url:
        return None
    identifier = openalex_url.rstrip("/").split("/")[-1]
    if identifier.startswith("W"):
        return identifier
    return None

def is_valid_url(url):
    return url and url != "None" and url.startswith(("http://", "https://"))

def normalize_url(url: str) -> str:
    """Convert arXiv abs → pdf"""
    if "arxiv.org/abs/" in url:
        return url.replace("arxiv.org/abs/", "arxiv.org/pdf/") + ".pdf"
    return url

def should_redownload(filename, first_url):
    """Skip existing unless first URL is arXiv/pdf"""
    file_html = download_folder / f"{filename}.html"
    file_pdf = download_folder / f"{filename}.pdf"

    if file_html.exists() or file_pdf.exists():
        if "arxiv.org" in first_url or first_url.lower().endswith(".pdf"):
            return True
        return False
    return True

# ==== DOWNLOAD FUNCTION ====
def download_content(session, urls, filename, max_retries=MAX_RETRIES):
    """Try multiple URLs until one succeeds"""
    for url in urls:
        for attempt in range(max_retries + 1):
            try:
                response = session.get(url, timeout=30)
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "").lower()

                    if "pdf" in content_type or url.lower().endswith(".pdf"):
                        file_path = download_folder / f"{filename}.pdf"
                        with open(file_path, "wb") as f:
                            f.write(response.content)
                        stats["pdf_downloads"] += 1
                    else:
                        file_path = download_folder / f"{filename}.html"
                        with open(file_path, "w", encoding="utf-8", errors="ignore") as f:
                            f.write(response.text)
                        stats["html_downloads"] += 1

                    return {
                        "success": True,
                        "filename": file_path.name,
                        "status_code": response.status_code,
                        "content_type": content_type,
                        "file_size": len(response.content),
                        "attempts": attempt + 1,
                        "url": url,
                    }
                elif attempt < max_retries:
                    time.sleep(1)
                    continue
                else:
                    break
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                else:
                    break
    return {"success": False, "error": "All URLs failed", "attempts": max_retries + 1}

# ==== CHECKPOINT ====
def save_checkpoint():
    checkpoint_data = {
        "processed_rows": list(processed_rows),
        "stats": stats,
        "timestamp": time.time(),
    }
    with checkpoint_file.open("w") as f:
        json.dump(checkpoint_data, f, indent=2)

def load_checkpoint():
    if checkpoint_file.exists():
        try:
            with checkpoint_file.open("r") as f:
                data = json.load(f)
                processed_rows.update(data.get("processed_rows", []))
                saved_stats = data.get("stats", {})
                for key in stats.keys():
                    if key in saved_stats:
                        stats[key] = saved_stats[key]
                stats["session_start_time"] = time.time()
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

def should_stop(start_time):
    return (time.time() - start_time) > MAX_RUNTIME_SECONDS

# ==== PARSE INPUT FILE ====
def parse_input_file():
    entries = []
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            stats["total_rows"] += 1
            if len(row) < 2:
                stats["ignored_rows"] += 1
                continue

            openalex_url = row[0].strip()
            urls = [normalize_url(u.strip().strip('"'))
                    for u in row[1:] if is_valid_url(u.strip().strip('"'))]

            filename = openalex_to_filename(openalex_url)
            if not filename or not urls:
                stats["ignored_rows"] += 1
                continue

            entries.append({
                "row_id": f"row_{idx}_{filename}",
                "row_idx": idx,
                "openalex_url": openalex_url,
                "urls": urls,
                "filename": filename
            })
            stats["valid_rows"] += 1
    return entries

# ==== MAIN FUNCTION ====
def main():
    load_checkpoint()
    entries = parse_input_file()
    remaining_entries = [e for e in entries if e["row_id"] not in processed_rows]

    if not remaining_entries:
        logger.info("All entries already processed!")
        return

    with requests.Session() as session, \
         open(results_file, "a", encoding="utf-8") as f_results, \
         open(failed_log_file, "a", encoding="utf-8") as f_failed:

        session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; Academic Research Bot)'})
        batch_size = 50

        for i in range(0, len(remaining_entries), batch_size):
            if should_stop(stats["session_start_time"]):
                break

            batch = remaining_entries[i:i+batch_size]

            with ThreadPoolExecutor(max_workers=CONCURRENT_DOWNLOADS) as executor:
                futures = {}
                for entry in batch:
                    if not should_redownload(entry["filename"], entry["urls"][0]):
                        stats["skipped_existing"] += 1
                        logger.info(f"Skipping {entry['filename']} (exists, not arxiv/pdf)")
                        processed_rows.add(entry["row_id"])
                        continue
                    future = executor.submit(download_content, session, entry["urls"], entry["filename"])
                    futures[future] = entry

                for future in as_completed(futures):
                    entry = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {"success": False, "error": str(e), "attempts": 0}

                    record = {
                        "row_idx": entry["row_idx"],
                        "openalex_url": entry["openalex_url"],
                        "urls": entry["urls"],
                        "filename": entry["filename"],
                        "timestamp": datetime.now().isoformat(),
                        **result
                    }

                    if result["success"]:
                        stats["successful_downloads"] += 1
                        f_results.write(json.dumps(record) + "\n")
                        f_results.flush()
                        logger.info(f"Downloaded {entry['filename']}")
                    else:
                        stats["failed_downloads"] += 1
                        f_failed.write(json.dumps(record) + "\n")
                        f_failed.flush()
                        logger.warning(f"Failed {entry['filename']}")

                    processed_rows.add(entry["row_id"])

            save_checkpoint()

if __name__ == "__main__":
    main()
