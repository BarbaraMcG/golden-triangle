import pandas as pd
from itertools import chain
from Bio import Entrez
from pyalex import Works  # Ensure pyalex is correctly installed

# Set up Entrez (PubMed) email
Entrez.email = "your_email@example.com"  # Replace with your email

####
## OpenAlex Request Adjusted for Specific Fields and Subfields
####
def get_openalex_dataframe(year, month_day, field_id, subfield_id):
    # Query for field ID
    results_field = Works().filter(from_publication_date=f"{year}-{month_day}", to_publication_date=f"{year}-{month_day}", open_access={"is_oa":True}, topics={"field":{"id":field_id}}).select(["id", "title", "authorships", "doi", "cited_by_count"])
    print("Field")
    # Query for subfield ID
    results_subfield = Works().filter(from_publication_date=f"{year}-{month_day}", to_publication_date=f"{year}-{month_day}", open_access={"is_oa":True}, topics={"subfield":{"id":subfield_id}}).select(["id", "title", "authorships", "doi", "cited_by_count"])
    print("Subfield")
    combined_results = []
    seen_ids = set()  # Use set for faster lookup
    for results in [results_field, results_subfield]:
        for record in chain(*results.paginate(per_page=200)):
            if record['id'] in seen_ids:
                continue  # Skip duplicates
            seen_ids.add(record['id'])  # Mark as seen
            temp_abstract = ""
            temp_abstract = Works()[record['id']]['abstract']
            authors = [author['author']['display_name'] for author in record.get('authorships', []) if 'author' in author and 'display_name' in author['author']]
            combined_results.append({
                'title': record.get('title', ''),
                'authors': ", ".join(authors),
                'DOI': record.get('doi', ''),
                'abstract': temp_abstract,
                'cited_by_count': record.get('cited_by_count',0),
                'source': 'OpenAlex'
            })
    print("Done")
    return pd.DataFrame(combined_results)

####
## PubMed Request Adjusted for Specific Date
####
def search_and_fetch_pubmed(date, mesh_terms):
    formatted_terms = ' OR '.join([f'"{term}"[MeSH Terms]' for term in mesh_terms.split(' OR ')])
    query = f"({formatted_terms}) AND ({date}[PDAT])"
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

# Fetch data for February 1st, 2023
df_openalex = get_openalex_dataframe("2023", "02-02", "13", "1203")
df_pubmed = search_and_fetch_pubmed("2023/02/02", "linguistics OR biology")
df_openalex.to_csv('openalex_2023-02-02-1.csv', index=False)
df_pubmed.to_csv('pubmed_2023-02-02-1.csv', index=False)

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
# Identify and process duplicates based on DOI
combined_df['DOI'].fillna('No DOI', inplace=True)
combined_df['is_duplicate'] = combined_df.duplicated('DOI', keep=False)
combined_df.loc[combined_df['is_duplicate'] & (combined_df['DOI'] != 'No DOI'), 'source'] = 'Both'
combined_df.drop(columns=['is_duplicate'], inplace=True)
combined_df['DOI'].replace('No DOI', pd.NA, inplace=True)

# Export to CSV
combined_df.to_csv('combined_search_results_2023-02-02-1.csv', index=False)
