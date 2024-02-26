import csv
from itertools import chain
import pandas as pd
from pyalex import Works
from Bio import Entrez

####
## Set up Entrez (PubMed) email
####
Entrez.email = "your_email@example.com"  # Replace with your email

####
## Request OpenAlex
####
def get_openalex_dataframe(year, field, subfield):
    results = Works().filter(publication_year=year, open_access={"is_oa":True},topics={"field": {"id": field}, "subfield": {"id": subfield}}).select(["id", "title", "authorships", "doi", "cited_by_count", "publication_date"])
    papers = []
    papers_without_abstract = 0

    for record in chain(*results.paginate(per_page=200)):
        temp_abstract = Works()[record['id']]['abstract']
        if not temp_abstract:
            papers_without_abstract += 1
            temp_abstract = ""
        authors = [author['author']['display_name'] for author in record.get('authorships', []) if 'author' in author and 'display_name' in author['author']]
        papers.append({
            'title': record.get('title', ''),
            'authors': ", ".join(authors),
            'DOI': record.get('doi', ''),
            'abstract': temp_abstract,
            'cited_by_count':record.get('cited_by_count',0),
            'source': 'OpenAlex'
        })

    print(f"Papers without abstract: {papers_without_abstract}")
    return pd.DataFrame(papers)

####
## Request Pubmed
####
def search_and_fetch_pubmed(year, mesh_terms):
    formatted_terms = ' AND '.join([f'"{term}"[MeSH Terms]' for term in mesh_terms.split(' AND ')])
    query = f"({formatted_terms}) AND ({year}[PDAT])"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=500)
    record = Entrez.read(handle)
    handle.close()
    id_list = record['IdList']
    
    ids = ','.join(id_list)
    handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    articles = []
    for article in records['PubmedArticle']:
        title = article['MedlineCitation']['Article']['ArticleTitle']
        authors_list = article['MedlineCitation']['Article'].get('AuthorList', [])
        authors = [f"{author.get('ForeName', '')} {author.get('LastName', '')}".strip() for author in authors_list]
        abstract = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', [''])[0] if 'Abstract' in article['MedlineCitation']['Article'] else ""
        doi = None
        for elocation in article['MedlineCitation']['Article'].get('ELocationID', []):
            if elocation.attributes.get('EIdType') == 'doi':
                doi = str(elocation)
                break
        articles.append({
            'title': title,
            'authors': ", ".join(authors),
            'DOI': doi,
            'abstract': abstract,
            'source': 'PubMed'
        })

    return pd.DataFrame(articles)

# Fetch data from OpenAlex and PubMed
df_openalex = get_openalex_dataframe(2023, "13", "1203")
df_pubmed = search_and_fetch_pubmed("2023", "linguistics AND biology")
df_openalex.to_csv('openalex.csv', index=False)
df_pubmed.to_csv('pubmed.csv', index=False)

# Number of papers for each source
num_papers_openalex = len(df_openalex)
num_papers_pubmed = len(df_pubmed)

# Number of papers missing abstracts for each source
# Assuming that an empty string or NaN in the 'abstract' column indicates a missing abstract
num_abstract_missing_openalex = df_openalex['abstract'].apply(lambda x: x == '' or pd.isna(x)).sum()
num_abstract_missing_pubmed = df_pubmed['abstract'].apply(lambda x: x == '' or pd.isna(x)).sum()

print(f"Number of papers from OpenAlex: {num_papers_openalex}")
print(f"Number of papers from PubMed: {num_papers_pubmed}")
print(f"Number of papers missing abstracts from OpenAlex: {num_abstract_missing_openalex}")
print(f"Number of papers missing abstracts from PubMed: {num_abstract_missing_pubmed}")

# Combine the two datasets
combined_df = pd.concat([df_openalex, df_pubmed], ignore_index=True)

# Identify duplicates based on DOI, if DOI is not None/NaN, and mark the source accordingly
# First, fill NaN DOIs with a temporary placeholder to ensure the comparison works for records without a DOI
combined_df['DOI'].fillna('No DOI', inplace=True)

# Identify duplicates based on DOI (including 'No DOI' placeholders) and mark the source accordingly
combined_df['is_duplicate'] = combined_df.duplicated('DOI', keep=False)
combined_df.loc[combined_df['is_duplicate'] & (combined_df['DOI'] != 'No DOI'), 'source'] = 'Both'

# Remove the 'is_duplicate' column as it's no longer needed
combined_df.drop(columns=['is_duplicate'], inplace=True)

# Optionally, if you want to replace 'No DOI' placeholders back to NaN for clarity in the final output
combined_df['DOI'].replace('No DOI', pd.NA, inplace=True)

# Export to CSV
combined_df.to_csv('combined_search_results.csv', index=False)
