import json
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure correct NLTK data directory
nltk_data_path = "/scratch_tmp/prj/dh_golden_triangle/myvenv/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Define file paths
input_file = "matched_papers_abstracts_2022_1000.jsonl"
output_file = "similarity_score_1000.jsonl"

# Download necessary NLTK resources (only if missing)
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)

# Load NLTK stopwords
stop_words = set(stopwords.words("english"))

# Function to preprocess text
def preprocess_text(text):
    if not text:
        return ""
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = re.sub(r"\d+", "", text)  # Remove numbers
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]  # Remove stopwords
    return " ".join(tokens)

# Function to extract text from an HTML file
def extract_text_from_html(html_path):
    try:
        with open(html_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            return soup.get_text(separator=" ")  # Extract readable text
    except (FileNotFoundError, IOError):
        return None  # Return None if file is missing or unreadable

# Open the input JSONL file and process each line separately
with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line_number, line in enumerate(f_in, start=1):
        try:
            # Print current line number for SLURM output
            print(f"Processing line {line_number}...", flush=True)

            # Load JSON line
            data = json.loads(line.strip())

            # Extract data paper details
            data_paper = data.get("data_journal_paper", {})
            data_paper_info = {
                "paper_id": data_paper.get("paper_id", "unknown"),
                "publication_year": data_paper.get("publication_year", "unknown"),
                "title": data_paper.get("title", "unknown"),
                "authors": [author["name"] for author in data_paper.get("authors", [])]
            }

            # Preprocess abstract and full text
            data_paper_abstract = preprocess_text(data_paper.get("abstract", ""))
            data_paper_fulltext = None
            if "html_file" in data_paper and data_paper["html_file"]:
                extracted_text = extract_text_from_html(data_paper["html_file"])
                data_paper_fulltext = preprocess_text(extracted_text) if extracted_text else None

            # Extract research papers
            research_papers = data.get("research_papers", [])

            # Check if all research papers have HTML
            all_have_html = all("html_file" in rp and rp["html_file"] for rp in research_papers)

            # Prepare for TF-IDF
            abstract_documents = [data_paper_abstract]
            fulltext_documents = [data_paper_fulltext] if all_have_html and data_paper_fulltext else None
            research_paper_info = []

            for research_paper in research_papers:
                research_paper_info.append({
                    "paper_id": research_paper.get("paper_id", "unknown"),
                    "publication_year": research_paper.get("publication_year", "unknown"),
                    "title": research_paper.get("title", "unknown"),
                    "authors": [author["name"] for author in research_paper.get("authors", [])],
                })

                research_paper_abstract = preprocess_text(research_paper.get("abstract", ""))
                research_paper_fulltext = None
                if "html_file" in research_paper and research_paper["html_file"]:
                    extracted_text = extract_text_from_html(research_paper["html_file"])
                    research_paper_fulltext = preprocess_text(extracted_text) if extracted_text else None

                abstract_documents.append(research_paper_abstract)
                if fulltext_documents is not None:
                    fulltext_documents.append(research_paper_fulltext)

            # Compute TF-IDF & Cosine Similarity (Abstract-based)
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(abstract_documents)
            abstract_similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            # Compute TF-IDF & Cosine Similarity (Full-text-based, if applicable)
            fulltext_similarity_scores = None
            if fulltext_documents is not None:
                fulltext_tfidf_matrix = vectorizer.fit_transform(fulltext_documents)
                fulltext_similarity_scores = cosine_similarity(fulltext_tfidf_matrix[0:1], fulltext_tfidf_matrix[1:]).flatten()

            # Attach similarity scores to research papers
            for i, research_paper in enumerate(research_paper_info):
                research_paper["abstract_score"] = round(abstract_similarity_scores[i], 4)
                research_paper["fullpaper_score"] = round(fulltext_similarity_scores[i], 4) if fulltext_similarity_scores is not None else None

            # Write structured JSONL output
            output_data = {
                "data_paper": data_paper_info,
                "research_papers": research_paper_info
            }
            f_out.write(json.dumps(output_data) + "\n")
            f_out.flush()
            print(f"Finished processing line {line_number}.", flush=True)

        except Exception as e:
            print(f"Error processing line {line_number}: {e}", flush=True)

print(f"Processing completed. Results saved in {output_file}", flush=True)
