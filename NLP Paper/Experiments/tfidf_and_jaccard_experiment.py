import pandas as pd
import json
from pathlib import Path
from experiments import RankingResult, ExperimentResults

# ==== CONFIGURATION ====
csv_dir = Path("/scratch/prj/dh_golden_triangle/recovered/algorithm_results")

experiment_results_dir = Path("/scratch/prj/dh_golden_triangle/recovered/combined_experiment_results")
experiment_results_dir.mkdir(parents=True, exist_ok=True)

match_excel_path = Path("/scratch/prj/dh_golden_triangle/recovered/match.xlsx")
output_csv = experiment_results_dir / "experiment_summary_all.csv"

# Define methods from both TF-IDF and Jaccard algorithms
methods = [
    # TF-IDF methods
    "tfidf_abstract_cosine",
    "tfidf_fulltext_cosine",
    # Jaccard methods
    "abstract_jaccard_all",
    "abstract_multiset_jaccard_all",
    "abstract_jaccard_top10",
    "abstract_multiset_jaccard_top10",
    "abstract_jaccard_top20",
    "abstract_multiset_jaccard_top20",
    "abstract_jaccard_top50",
    "abstract_multiset_jaccard_top50",
    "fulltext_jaccard_all",
    "fulltext_multiset_jaccard_all",
    "fulltext_jaccard_top10",
    "fulltext_multiset_jaccard_top10",
    "fulltext_jaccard_top20",
    "fulltext_multiset_jaccard_top20",
    "fulltext_jaccard_top50",
    "fulltext_multiset_jaccard_top50"
]

# ==== STEP 1: PREPROCESS CSVs TO JSON ====
print("=" * 60)
print("STEP 1: Preprocessing CSV files to JSON")
print("=" * 60)

all_method_results = {method: {} for method in methods}
total_matches = {method: 0 for method in methods}

# Process Jaccard CSV files
for csv_file in csv_dir.glob("*_jaccard.csv"):
    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        data_id = str(row['data_openalex_id'])
        research_id = row['research_openalex_id']

        if pd.isna(research_id) or research_id == "":
            continue

        for method in methods:
            if method not in row:
                continue
            
            score = row[method]
            if pd.isna(score):
                continue
            
            if data_id not in all_method_results[method]:
                all_method_results[method][data_id] = {}
            
            all_method_results[method][data_id][str(research_id)] = float(score)
            total_matches[method] += 1

# Process TF-IDF CSV files
for csv_file in csv_dir.glob("*_tfidf.csv"):
    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        data_id = str(row['data_openalex_id'])
        research_id = row['research_openalex_id']

        if pd.isna(research_id) or research_id == "":
            continue

        for method in methods:
            if method not in row:
                continue
            
            score = row[method]
            if pd.isna(score):
                continue
            
            if data_id not in all_method_results[method]:
                all_method_results[method][data_id] = {}
            
            all_method_results[method][data_id][str(research_id)] = float(score)
            total_matches[method] += 1

print("\nPreprocessing Summary:")
for method, count in total_matches.items():
    print(f"  {method}: {count} matches")

# ==== STEP 2: RUN EXPERIMENTS ====
print("\n" + "=" * 60)
print("STEP 2: Running experiments on all methods")
print("=" * 60)

def rank_pairs_from_dict(sim_dict, data_doi, ans_doi):
    """Return a single RankingResult or None if missing."""
    if data_doi not in sim_dict:
        return None

    candidates_dict = sim_dict[data_doi]
    if ans_doi not in candidates_dict:
        return None

    ranked_df = pd.DataFrame({
        'doi': list(candidates_dict.keys()),
        'score': list(candidates_dict.values())
    })

    try:
        rr = RankingResult.from_ans_and_ranked_df(
            query_doi=data_doi,
            ans_doi=ans_doi,
            ranked_df=ranked_df
        )
        return rr
    except ValueError as e:
        print(f"Skipping {data_doi} -> {ans_doi}: {e}")
        return None

match_df = pd.read_excel(match_excel_path)
experiment_results_dict = {}

for method in methods:
    if method not in all_method_results or not all_method_results[method]:
        print(f"\nSkipping {method} (no data)")
        continue
    
    print(f"\nProcessing {method}...")
    sim_dict = all_method_results[method]

    results = []
    skipped_pairs = 0

    for _, row in match_df.iterrows():
        data_doi = row.get('openalex_data_paper')
        research_doi = row.get('openalex_research')

        if pd.isna(data_doi) or pd.isna(research_doi):
            skipped_pairs += 1
            continue

        rr = rank_pairs_from_dict(sim_dict, data_doi, research_doi)
        if rr is not None:
            results.append(rr)
        else:
            skipped_pairs += 1

    if results:
        exp_results = ExperimentResults(results)
        experiment_results_dict[method] = exp_results

        output_file = experiment_results_dir / f"{method}_experiment_results_all.json"
        exp_results.save_to_json(output_file)
        print(f"Saved {method} ExperimentResults to {output_file} ({len(results)} valid pairs, {skipped_pairs} skipped)")
    else:
        print(f"Warning: No valid RankingResults for {method} (skipped {skipped_pairs} pairs)")

print("\n=== Experiment Processing Summary ===")
for method, exp in experiment_results_dict.items():
    print(f"{method}: {len(exp.ranking_results)} ranking results saved")

# ==== STEP 3: GENERATE STATISTICS ====
print("\n" + "=" * 60)
print("STEP 3: Generating experiment statistics")
print("=" * 60)

summary_rows = []

for json_file in sorted(experiment_results_dir.glob("*_experiment_results_all.json")):
    method_name = json_file.stem
    exp = ExperimentResults.load_from_json(json_file)
    
    # Top-N accuracy
    top_ns = [5, 10, 50]
    top_n_acc = {f"Top-{n}": exp._calculate_top_n_accuracy(n) for n in top_ns}
    
    # Top-p accuracy
    top_ps = [0.01, 0.05, 0.10]
    top_p_acc = {f"Top-{int(p*100)}%": exp._calculate_top_p_accuracy(p) for p in top_ps}
    
    # MRR and std
    rrs = exp.get_reciprocal_ranks()
    mrr = sum(rrs)/len(rrs) if rrs else 0
    sdt_rr = pd.Series(rrs).std() if rrs else 0
    
    row = {"method": method_name, "MRR": mrr, "RR_std": sdt_rr}
    row.update(top_n_acc)
    row.update(top_p_acc)
    
    summary_rows.append(row)

# Save to CSV
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(output_csv, index=False)
print(f"Saved experiment summary to {output_csv}")
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"\nProcessed {len(summary_rows)} methods")
print(f"Experiment results: {experiment_results_dir}")
print(f"Summary CSV: {output_csv}")