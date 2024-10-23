import json
import os
from os import path
from tqdm import tqdm
import urllib.request
import pandas as pd

json_dir = 'data/json'

def get_openalex_json(doi, force=False):
    """Get metadata of a paper from OpenAlex database."""
    fn = path.join(json_dir, doi.replace('/', '_') + '.json')
    if not force and os.path.exists(fn):
        return
    api_url = 'https://api.openalex.org/works/https://doi.org/' + doi
    os.makedirs(json_dir, exist_ok=True)
    try:
        urllib.request.urlretrieve(api_url, fn)
    except urllib.error.HTTPError as e:
        print(f'Error downloading {doi}:', e)


links_df = pd.read_csv('data/links.csv', usecols=['data_paper_doi', 'research_paper_doi'])
dois = list(set(links_df['data_paper_doi']).union(set(links_df['research_paper_doi'])))
for doi in tqdm(dois):
    # print('Downloading ' + doi)
    get_openalex_json(doi)
