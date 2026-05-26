# converting OpenAlex IDs in 'query_doi' field to actual DOIs in experiment result files of Viola's experiments
import ast
import json
import io
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import spacy
import duckdb
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai
import re

import sys
sys.path.append('')

from experiments import ExperimentResults, RankingResult

tqdm.pandas()


db_file = 'data/filtered_matches.duckdb'
cache_dir = Path('cache')
cache_dir.mkdir(exist_ok=True, parents=True)

# cache the models so they're loaded once
_SBERT_MODEL = None
_CROSS_ENCODER_MODEL = None

def _get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _SBERT_MODEL = SentenceTransformer(model_name, device=device)
    return _SBERT_MODEL

def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _CROSS_ENCODER_MODEL
    if _CROSS_ENCODER_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CROSS_ENCODER_MODEL = CrossEncoder(model_name, device=device)
    return _CROSS_ENCODER_MODEL

def _parse_gpt_ranking(gpt_response: str, expected_length: int) -> list:
    """Parse GPT's ranking response and return list of indices."""
    # Extract numbers from the response
    numbers = re.findall(r'\b\d+\b', gpt_response)
    
    try:
        ranking = [int(num) for num in numbers]
        
        # Validate that all numbers are in expected range
        valid_numbers = [num for num in ranking if 1 <= num <= expected_length]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ranking = []
        for num in valid_numbers:
            if num not in seen:
                unique_ranking.append(num)
                seen.add(num)
        
        # If we don't have all numbers, fill in the missing ones
        missing = [i for i in range(1, expected_length + 1) if i not in seen]
        unique_ranking.extend(missing)
        
        return unique_ranking[:expected_length]
        
    except (ValueError, TypeError):
        # If parsing fails, return original order
        print(f"Failed to parse GPT ranking: {gpt_response}")
        raise
        return list(range(1, expected_length + 1))


def get_ner(spacy_ner: list) -> list:
    return [x for x,y in spacy_ner]

def extract_named_entities_doc(doc):
    return [(ent.text, ent.label_) for ent in doc.ents]

def tokenize_doc(doc):
    return [token.text for token in doc if not token.is_punct and not token.is_space and not token.is_stop]

def get_sbert(text: str) -> np.ndarray:
    model = _get_model()
    emb = model.encode(text, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)

def author_ids_csv_to_list(s):
    return s.split(',')

def author_ids_csv_to_list_pandas(s):
    df = pd.read_csv(io.StringIO(s), header=None)
    return df[0].tolist()

def apply_ner(df):
    nlp = spacy.load("en_core_web_sm")
    n_workers = 3   # e.g., multiprocessing.cpu_count() - 1
    batch_size = 100
    m = 10
    chunks = np.array_split(df, m)
    print('Running...')
    for i, chunk in enumerate(chunks):
        texts = chunk['abstract'].fillna('').astype(str).tolist()
    
        chunk_results = []
        for doc in tqdm(nlp.pipe(texts, n_process=n_workers, batch_size=batch_size),
                        total=len(texts), desc=f"NER [{i+1}/{m}]"):
            chunk_results.append(extract_named_entities_doc(doc))
    
        # assign back to the right rows (preserves order by index)
        df.loc[chunk.index, 'ner'] = pd.Series(chunk_results, index=chunk.index, dtype='object')
    
    return df

def apply_tokenize(df):
    nlp = spacy.load("en_core_web_sm")
    n_workers = 3   # e.g., multiprocessing.cpu_count() - 1
    batch_size = 100
    m = 10
    chunks = np.array_split(df, m)
    print('Running...')
    for i, chunk in enumerate(chunks):
        texts = chunk['abstract'].fillna('').astype(str).str.lower().tolist()
    
        chunk_results = []
        for doc in tqdm(nlp.pipe(texts, n_process=n_workers, batch_size=batch_size),
                        total=len(texts), desc=f"Tokenize [{i+1}/{m}]"):
            chunk_results.append(tokenize_doc(doc))
    
        # assign back to the right rows (preserves order by index)
        df.loc[chunk.index, 'toks'] = pd.Series(chunk_results, index=chunk.index, dtype='object')

    return df


def apply_sbert(df):
    print('Running...')
    m = 10
    chunks = np.array_split(df, m)
    for i, chunk in enumerate(chunks):
        texts = chunk['abstract'].fillna('').astype(str).str.lower().tolist()
    
        chunk_results = []
        for doc in tqdm(texts, desc=f"SBERT [{i+1}/{m}]"):
            chunk_results.append(get_sbert(doc))
    
        # assign back to the right rows (preserves order by index)
        df.loc[chunk.index, 'sbert'] = pd.Series(chunk_results, index=chunk.index, dtype='object')

    return df

def create_or_load_from_cache():
    con = duckdb.connect(db_file)
    df = con.execute('SELECT * FROM works').df()
    con.close()
    
    cached_cols = ['ner', 'toks', 'sbert']

    for cached_col in cached_cols:
        cache_file = cache_dir / f'{cached_col}.csv'
        if not cache_file.exists():
            print(f'Creating cache for column: {cached_col}')
            if cached_col == 'ner':
                df = apply_ner(df)
            elif cached_col == 'toks':
                df = apply_tokenize(df)
            elif cached_col == 'sbert':
                df = apply_sbert(df)
            df[[cached_col]].to_csv(cache_file, index=False)
        
        df[cached_col] = pd.read_csv(cache_file)[cached_col]

    df['author_ids'] = df['author_ids'].progress_apply(lambda x: author_ids_csv_to_list(x))
    df['ner'] = df['ner'].progress_apply(ast.literal_eval)
    df['toks'] = df['toks'].progress_apply(ast.literal_eval)
    df['sbert'] = df['sbert'].progress_apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=np.float32).tolist())

    return df


def get_candidates(df, query_doi):
    query_row = df[df['doi'].str.lower() == query_doi.lower()]
    if query_row.empty:
        raise ValueError(f"DOI {query_doi} not found in DataFrame.")
    query_is_data_paper = query_row.iloc[0]['is_data_paper']
    if query_is_data_paper:
        # Candidates: research papers (not data journals)
        candidates = df[df['is_data_paper'] == False]
    else:
        # Candidates: data papers (data journals)
        candidates = df[df['is_data_paper'] == True]
    
    authors = query_row.iloc[0]['author_ids']
    mask = candidates['author_ids'].apply(lambda x: bool(set(x) & set(authors)))
    candidates = candidates[mask]
    return candidates

def rank_pairs_tfidf(df, query_doi, ans_doi, text_col): 
    # -> [list of top_n ranks, 
    # Select the row for the given data_paper_doi
    
    query_row = df[df['doi'].str.lower() == query_doi.lower()]
    if query_row.empty:
        raise ValueError(f"DOI {query_doi} not found in DataFrame.")

    candidates = get_candidates(df, query_doi)

    query_text = query_row.iloc[0][text_col]
    candidate_texts = list(candidates[text_col])
    
    # Compute similarity
    tfidf = TfidfVectorizer().fit([query_text] + candidate_texts)
    query_vec = tfidf.transform([query_text])
    candidate_vecs = tfidf.transform(candidate_texts)

    sims = cosine_similarity(query_vec, candidate_vecs).flatten()
    candidates = candidates.copy()
    candidates['score'] = sims

    return RankingResult.from_ans_and_ranked_df(query_doi, ans_doi, candidates)


def rank_pairs_sbert(df, query_doi, ans_doi, rerank=None, rerank_top_k=50): 
    # -> [list of top_n ranks, 
    # Select the row for the given data_paper_doi
    
    query_row = df[df['doi'].str.lower() == query_doi.lower()]
    if query_row.empty:
        raise ValueError(f"DOI {query_doi} not found in DataFrame.")

    candidates = get_candidates(df, query_doi)

    query_vec = np.array(query_row.iloc[0]['sbert']).reshape(1, -1)
    candidate_vecs = np.array(list(candidates['sbert']))
    sims = cosine_similarity(query_vec, candidate_vecs).flatten()
    candidates = candidates.copy()
    candidates['score'] = sims

    if rerank == 'cross-encoder':
        # Sort by initial SBERT scores and take top candidates for reranking
        top_k = min(rerank_top_k, len(candidates))  # Rerank top 100 candidates
        top_k_candidates_sorted = candidates.sort_values('score', ascending=False)
        top_candidates = top_k_candidates_sorted.head(top_k)
        original_top_k_dois = set(top_candidates['doi'].str.lower())
        
        # Prepare query-candidate pairs for cross-encoder
        query_text = query_row.iloc[0]['abstract']
        candidate_texts = top_candidates['abstract'].fillna('').astype(str).tolist()
        
        # Create pairs for cross-encoder
        pairs = [(query_text, candidate_text) for candidate_text in candidate_texts]
        
        # Get cross-encoder scores
        cross_encoder = _get_cross_encoder()
        cross_scores = cross_encoder.predict(pairs)
        cross_scores -= min(cross_scores)  # Normalize to be non-negative
        assert all(cross_scores >= 0), "Cross-encoder returned negative scores, which is unexpected."
        
        # Update scores with cross-encoder results
        top_candidates = top_candidates.copy()
        top_candidates['score'] = cross_scores + 1 + max(sims)  # Ensure reranked scores are higher than original SBERT scores
        
        # Combine reranked top-k with remaining candidates (keeping original SBERT scores)
        remaining_candidates = top_k_candidates_sorted.iloc[top_k:]
        candidates = pd.concat([top_candidates, remaining_candidates], ignore_index=True)
        
        # Assert that the top-k DOIs remain the same
        final_top_k_dois = set(candidates.sort_values('score', ascending=False).head(top_k)['doi'].str.lower())
        assert original_top_k_dois == final_top_k_dois, f"Top-k DOIs changed during reranking. Original: {len(original_top_k_dois)}, Final: {len(final_top_k_dois)}"
    
    elif rerank == 'gpt':
        # Sort by initial SBERT scores and take top candidates for reranking
        top_k = min(rerank_top_k, len(candidates))
        top_k_candidates_sorted = candidates.sort_values('score', ascending=False)
        top_candidates = top_k_candidates_sorted.head(top_k)
        original_top_k_dois = set(top_candidates['doi'].str.lower())
        
        # Prepare data for GPT reranking
        query_text = query_row.iloc[0]['abstract']
        query_title = query_row.iloc[0]['title']
        
        # Create numbered list of candidates for GPT
        candidate_list = []
        for i, (_, row) in enumerate(top_candidates.iterrows(), 1):
            title = row['title']
            # abstract = row['abstract'][:500] + '...' if len(row['abstract']) > 500 else row['abstract']  # Truncate for token limits
            abstract = row['abstract']
            candidate_list.append(f"{i}. Title: {title}\n   Abstract: {abstract}")
        
        candidates_text = '\n\n'.join(candidate_list)
        
        # Create GPT prompt
        prompt = f"""You are helping to find research papers that are related to a given data paper. 

DATA PAPER:
Title: {query_title}
Abstract: {query_text}

Below are {len(candidate_list)} research papers ranked by semantic similarity. Your task is to rerank them based on how likely each research paper is to be related to the data paper above (i.e., the research paper likely uses or references the dataset described in the data paper).

CANDIDATES TO RERANK:
{candidates_text}

Please provide your reranking as a comma-separated list of numbers, with the most relevant paper first. For example: 3,1,7,2,5,4,6

Your reranking:"""
        for retries_i in range(3):
            try:
                # Call OpenAI API
                client = openai.OpenAI()  # Assumes OPENAI_API_KEY is set in environment
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                gpt_response = response.choices[0].message.content.strip()
                
                # Parse GPT response to extract ranking
                gpt_ranking = _parse_gpt_ranking(gpt_response, len(candidate_list))
                break  # Exit retry loop if successful
            except Exception as e:
                print(f"GPT reranking failed: {e}, retrying ... [{retries_i+1}/{3}].")
            
        # Apply GPT reranking
        reranked_candidates = []
        for rank_idx in gpt_ranking:
            original_idx = rank_idx - 1  # Convert to 0-based indexing
            reranked_candidates.append(top_candidates.iloc[original_idx])
        
        # Create new dataframe with reranked candidates
        top_candidates_reranked = pd.DataFrame(reranked_candidates)
        
        # Assign new scores (higher scores for better ranks)
        max_original_score = max(sims)
        new_scores = [max_original_score + 1 + (len(gpt_ranking) - i) for i in range(len(gpt_ranking))]
        top_candidates_reranked['score'] = new_scores
        
        # Combine reranked top-k with remaining candidates
        remaining_candidates = top_k_candidates_sorted.iloc[top_k:]
        candidates = pd.concat([top_candidates_reranked, remaining_candidates], ignore_index=True)
        
        # Assert that the top-k DOIs remain the same
        final_top_k_dois = set(candidates.sort_values('score', ascending=False).head(top_k)['doi'].str.lower())
        assert original_top_k_dois == final_top_k_dois, f"Top-k DOIs changed during GPT reranking. Original: {len(original_top_k_dois)}, Final: {len(final_top_k_dois)}"
        
        
    return RankingResult.from_ans_and_ranked_df(query_doi, ans_doi, candidates)


def rank_pairs_jaccard(df, query_doi, ans_doi, toks_col, multiset=False):
    # Locate the query row
    query_row = df[df['doi'].str.lower() == query_doi.lower()]
    if query_row.empty:
        raise ValueError(f"DOI {query_doi} not found in DataFrame.")

    # Get candidate set (research vs. data papers)
    candidates = get_candidates(df, query_doi)
    if candidates.empty:
        raise ValueError(f"No candidates found for DOI {query_doi}.")

    if multiset:
        query_toks = set(f'{x}_{i}' for i, x in enumerate(query_row.iloc[0][toks_col]))
    else:
        query_toks = set(query_row.iloc[0][toks_col])

    # Compute Jaccard similarities
    def jaccard_score(toks):
        if multiset:
            toks_set = set(f'{x}_{i}' for i, x in enumerate(toks))
        else:
            toks_set = set(toks)
        intersection = len(query_toks & toks_set)
        union = len(query_toks | toks_set)
        return intersection / union if union > 0 else 0.0

    sims = candidates[toks_col].apply(jaccard_score)

    # Attach scores
    candidates = candidates.copy()
    candidates["score"] = sims

    return RankingResult.from_ans_and_ranked_df(query_doi, ans_doi, candidates)


def exp_ner_tfidf():
    for i, bad_ner in enumerate(bad_ners):
        print("\n\n" + "-"*40)
        print(f'Filtering out NER types: {bad_ner}')
        df['ner_filtered'] = df['ner'].apply(lambda x: [item for item in x if item[1] not in bad_ner])
        df['ner_filtered'] = df['ner_filtered'].apply(lambda x: ' '.join(get_ner(x)))
        data_to_research_ranks = []
        
        for idx, data_paper_doi, research_paper_doi in tqdm(ground_truth_both_presents_df.itertuples()):
            print(idx)
            # Data to research
            rank_result = rank_pairs_tfidf(df, query_doi=data_paper_doi, ans_doi=research_paper_doi, text_col='ner_filtered')
            data_to_research_ranks.append(rank_result)

        Path('exps').mkdir(exist_ok=True, parents=True)
        ExperimentResults(data_to_research_ranks).save_to_json(f'exps/ner_tfidf_variant_{i}.json')

def exp_toks_tfidf():
    data_to_research_ranks = []
    
    for idx, data_paper_doi, research_paper_doi in tqdm(ground_truth_both_presents_df.itertuples()):
        print(idx)
        # Data to research
        rank_result = rank_pairs_tfidf(df, query_doi=data_paper_doi, ans_doi=research_paper_doi, text_col='abstract')
        data_to_research_ranks.append(rank_result)

    Path('exps').mkdir(exist_ok=True, parents=True)
    ExperimentResults(data_to_research_ranks).save_to_json(f'exps/toks_tfidf.json')


def exp_toks_sbert(rerank=None):
    data_to_research_ranks = []
    
    for idx, data_paper_doi, research_paper_doi in tqdm(ground_truth_both_presents_df.itertuples()):
        print(idx)
        # Data to research
        rank_result = rank_pairs_sbert(df, query_doi=data_paper_doi, ans_doi=research_paper_doi, rerank=rerank)
        data_to_research_ranks.append(rank_result)

    Path('exps').mkdir(exist_ok=True, parents=True)
    if rerank:
        suffix = f'_{rerank}'
    else:
        suffix = ''
    ExperimentResults(data_to_research_ranks).save_to_json(f'exps/toks_sbert{suffix}.json')

def exp_ner_jaccard(multiset: bool):
    for i, bad_ner in enumerate(bad_ners):
        print("\n\n" + "-"*40)
        print(f'Filtering out NER types: {bad_ner}')
        df['ner_filtered'] = df['ner'].apply(lambda x: [item for item in x if item[1] not in bad_ner])
        df['ner_filtered'] = df['ner_filtered'].apply(lambda x: get_ner(x))
        data_to_research_ranks = []
        
        for idx, data_paper_doi, research_paper_doi in tqdm(ground_truth_both_presents_df.itertuples()):
            # Data to research
            rank_result = rank_pairs_jaccard(df, query_doi=data_paper_doi, ans_doi=research_paper_doi,
                toks_col='ner_filtered', multiset=multiset)
            data_to_research_ranks.append(rank_result)

        Path('exps').mkdir(exist_ok=True, parents=True)
        multiset_str = 'multiset' if multiset else 'standard'
        ExperimentResults(data_to_research_ranks).save_to_json(f'exps/ner_jaccard_{multiset_str}_variant_{i}.json')


def exp_toks_jaccard(multiset: bool):
    # use spacy to remove punctuation, remove stopwords, and tokenize the abstract
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
    
    
    df['all_toks'] = df['abstract'].apply(nlp)
    for top_n in [10, 20, 30, 40,1000_000]:
        print("\n\n" + "-"*40)
        print(f'Top N: {top_n}')
        data_to_research_ranks = []
        
        for idx, data_paper_doi, research_paper_doi in tqdm(ground_truth_both_presents_df.itertuples()):
            print(idx)
            # Data to research
            rank_result = rank_pairs_jaccard(df, query_doi=data_paper_doi, ans_doi=research_paper_doi,
                toks_col='toks', multiset=multiset)
            data_to_research_ranks.append(rank_result)

        Path('exps').mkdir(exist_ok=True, parents=True)
        multiset_str = 'multiset' if multiset else 'standard'
        ExperimentResults(data_to_research_ranks).save_to_json(f'exps/toks_jaccard_{multiset_str}_top_n_{top_n}.json')


if __name__ == '__main__':
    df = create_or_load_from_cache()

    ground_truth_both_presents_df = pd.read_csv('data/match.csv')

    bad_ners = [
        {},
        {'ORG'},
        {'ORDINAL', 'DATE', 'CARDINAL', 'PERCENT', 'QUANTITY', 'TIME'},
        {'ORG', 'ORDINAL', 'DATE', 'CARDINAL', 'PERCENT', 'QUANTITY', 'TIME'}
    ]


    # exp_ner_tfidf()
    # exp_toks_tfidf()
    # exp_toks_sbert(rerank=None)
    # exp_toks_sbert(rerank='cross-encoder')
    exp_toks_sbert(rerank='gpt')

    # exp_ner_jaccard(multiset=False)
    # exp_ner_jaccard(multiset=True)

    # exp_toks_jaccard(multiset=False)
    # exp_toks_jaccard(multiset=True)
