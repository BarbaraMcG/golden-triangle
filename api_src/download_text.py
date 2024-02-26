from os import makedirs, path
import json
from glob import glob
import urllib.request
from urllib.parse import urlparse
import mimetypes
import pandas as pd

def ensure_dir(file_path):
    """Ensure that a directory exists."""
    directory = path.dirname(file_path)
    if not path.exists(directory):
        makedirs(directory)

def get_openalex_file_url(doi):
    """Get url of a paper's file from OpenAlex database."""
    api_url = 'https://api.openalex.org/works/https://doi.org/' + doi

    with urllib.request.urlopen(api_url) as f:
        data = json.load(f)
        oa = data['open_access']
        return oa['is_oa'], oa['oa_status'], oa['oa_url']

def download_by_openalex(doi, fn, force_download=False):
    """Download a paper based on data in OpenAlex database."""
    try:
        is_oa, oa_status, oa_url = get_openalex_file_url(doi)

        print(is_oa, oa_status, oa_url)
        if oa_url is None:
            print('WARNING: CLOSED ARTICLE')
        else:
            return direct_download(oa_url, fn, force_download)
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        print(e)

def direct_download(url, fn, force_download):
    req = urllib.request.Request(
        url,
        data=None,
        headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
        }
    )

    with urllib.request.urlopen(req) as resp:
        content_type = resp.info().get_content_type()
        ext = mimetypes.guess_extension(content_type)
        
        # Check if the guessed extension is .bin
        if ext == '.bin':
            print('File likely to be a .bin, skipping download.')
            return
        
        if not fn.endswith(ext):
            fn += ext
        if not force_download and path.exists(fn):
            print('File exists, ignored. (use force_download=True to re-download)')
            return
        
        with open(fn, 'wb') as of:
            of.write(resp.read())
        print('Done!')
        
def get_fn_of_doi(doi, dest_dir, ext='', already_exists=False):
    """ Get filename and path from a DOI.

    Args:
        ext: File extension with the leading dot (such as '.html').
        already_exists: A Boolean flag that if true, assumes the file exists and finds the extension based on the saved file.
    """
    assert not(ext != '' and already_exists)  
    if '://' in doi:
        d = urlparse(doi)
        fn = path.basename(d.path)
    else:
        fn = doi.replace('/', '_') + ext
    
    fn = path.join(dest_dir, fn)
    if not already_exists:
        return fn
    else:
        a = list(glob(f'{fn}.*'))
        a = [x for x in a if '.abstract.' not in x]
        assert len(a) <= 2 # at most one pdf and one non-pdf files should exist for a file (according to the rest of the codes)
        return a[0] if len(a) > 0 else None

def download_paper(doi, dest_dir, force_download=False):
    """Download a paper with a specified DOI."""
    print(f'Downloading {doi}')
    # Ensure the directory path ends with a path separator if not already
    if not dest_dir.endswith(path.sep):
        dest_dir += path.sep
    fn = get_fn_of_doi(doi, dest_dir, already_exists=True)
    if fn is None:  # If file does not exist or extension is unknown
        # This time ensure to pass the correct dest_dir to get_fn_of_doi
        fn = get_fn_of_doi(doi, dest_dir)  # Get base filename with correct directory
    download_by_openalex(doi, fn, force_download)  # Ensure fn is passed, not dest_dir

# Main process
csv_file = 'pubmed_2023-02-02.csv'
dest_dir = csv_file.replace('.csv', '')  # Destination directory

# Ensure the folder exists
ensure_dir(dest_dir + '/')  # Ensure it's treated as a directory

df = pd.read_csv(csv_file)
for index, row in df.iterrows():
    doi = row.get('DOI')
    if pd.isnull(doi) or doi == 'No DOI':
        print(f'Skipping row {index} due to missing DOI')
        continue
    download_paper(doi, dest_dir)