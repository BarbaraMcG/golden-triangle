# Automatic Linking of Data and Research Papers
This project is a follow-up to the following paper: [Deep Impact: A Study on the Impact of Data Papers and Datasets in the Humanities and Social Sciences](https://doi.org/10.3390/publications10040039).  Its aim is to automate the linking between data and research papers.

Data (in January 2024):

    - 'data/links.csv': contains 80 pairs (29 from original JOHD, 51 from Jade) and 152 unique DOIs. However, not all of them are present in OpenAlex (see parse_metadata_json.py below). There 148 unique DOIs with available metadata. 
    - 'data/research_papers', 'data/data_papers', 'data/all_papers': PDF files of research, data, and all papers (there are some unneeded PDF files in these folders) (all papers is used for LLM-based indexing which we didn't pursue further).

Code:

    NER method:
        - 'download.py': downloads pdf files from OpenAlex data set
        - 'ner.py': named-entity recognition method

    ML method:
        - 'download_metadata.py': downloads OpenAlex metadata (a raw json files) of all DOIs in 'data/links.csv' into folder 'data/json' (148 unique jsons)
        - 'parse_metadata_json.py': parses downloaded metadata from folder 'json' and creates 'data/paper-features.csv' (148 unique DOIs)
        - 'ml-method.ipynb': a notebook containing a neural network using features derived from 'title', 'publication_date', 'authors', 'concepts' (see docx document for more detail)
