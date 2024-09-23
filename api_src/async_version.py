import asyncio
import aiohttp
import os
import json
import pandas as pd
from pyalex import Works
import time
from pathlib import Path

# Constants
BASE_URL = "https://api.openalex.org/works"
PARAMS = {
    'filter': 'is_oa:true,publication_year:2023,type:article|book-chapter,language:en',
    'per_page': 200
}
OUTPUT_DIR = 'OpenAlex_Files/'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'openalex_data_test.csv')
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint_test.json')
SUMMARY_FILE = os.path.join(OUTPUT_DIR, 'summary.json')
DOWNLOAD_DIR = os.path.join(OUTPUT_DIR, 'downloads')

# Ensure output and download directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

DATA_DIR = Path('data')
DATA_DIR.mkdir(parents=True, exist_ok=True)

def extract_doi(doi_url):
    return doi_url.replace('https://doi.org/', '').replace('/', '_')
    


async def fetch_data(session, page):
    """
    Asynchronously fetches data from the OpenAlex API for a given page number.
    
    Args:
    session (aiohttp.ClientSession): The session used for making HTTP requests.
    page (int): The page number to fetch data for.

    Returns:
    dict: The JSON response from the API containing the data.
    """
    params = PARAMS.copy()
    params['page'] = page
    async with session.get(BASE_URL, params=params) as response:
        response.raise_for_status()
        return await response.json()

async def download_file(session, url, filename):
    """
    Asynchronously downloads a file from the specified URL and saves it to the predefined download directory.

    Args:
    session (aiohttp.ClientSession): The session used for HTTP requests.
    url (str): The URL from which the file is to be downloaded.
    filename (str): The name of the file to save the downloaded content.

    Returns:
    bool: True if file was successfully downloaded and saved, False otherwise.
    """
    if not url:
        return False
    try:
        path = os.path.join(DOWNLOAD_DIR, filename)
        async with session.get(url) as response:
            response.raise_for_status()
            content = await response.read()
            with open(path, 'wb') as f:
                f.write(content)
            return True
    except Exception as e:
        return False

async def process_data(page):
    """
    Processes data from a specified page by first fetching the data and then parsing it.
    Extracts relevant information and attempts to download associated HTML content if available.

    Args:
    session (aiohttp.ClientSession): The session used for HTTP requests.
    page (int): The page number to process.

    Returns:
    list: A list of dictionaries containing parsed data for each record.
    int: The count of entries for which HTML content could not be downloaded.
    """
    print(f'Page {page}: started processing ...')
    PAGE_FN = DATA_DIR / str(page)
    if PAGE_FN.exists():
        obj = json.loads(PAGE_FN.read_text())
        print(f'Page {page}: file found, ignored.')
        return obj['data_list'], obj['missing_html_count']

    start_time = time.time()
    print(f'Page {page}: start_time={start_time}')
    async with aiohttp.ClientSession() as session: 
        try:
            results = await fetch_data(session, page)
        except Exception as e:
            print(f"Error fetching data for page {page}: {e}")
            return [], 0
        data_list = []
        missing_html_count = 0
        for result in results['results']:
            try:
                id = result.get('id')
                doi = result.get('doi')
                title = result.get('title')
                abstract = Works()[result['id']]['abstract']
                publication_date = result.get('publication_date')
                authors = [{'Name': a_info.get('author', {}).get('display_name'), 
                            'ORCID': a_info.get('author', {}).get('orcid'), 
                            'Institutions': [inst.get('display_name') for inst in a_info.get('institutions', [])]} 
                        for a_info in result.get('authorships', [])]
                primary_location = result.get('primary_location', {})
                locations = result.get('locations', [])
                source_display_name = ''
                try:
                    best_oa = result.get('best_oa_location', {})
                    if best_oa:
                        source = best_oa.get('source', {})
                        if source:
                            source_display_name = source.get('display_name', '')
                except Exception as e:
                    print(f"Error fetching best OA location source display name: {e}")
                landing_page = primary_location.get('landing_page_url')
                html_missing = not landing_page or not await download_file(session, landing_page, f"{extract_doi(doi)}.html")
                missing_html_count += html_missing
                data_list.append({
                    'ID': id, 'DOI': doi, 'Title': title, 'Abstract': abstract, 
                    'Publication Date': publication_date, 'Authors': authors, 
                    'Total Citations': result.get('cited_by_count'),
                    'Cited By Year': result.get('counts_by_year', []), 
                    'Mesh': result.get('mesh', []), 
                    'Referenced Count': result.get('referenced_works_count'),
                    'Referenced Works': result.get('referenced_works', []), 
                    'Countries Count': result.get('countries_distinct_count'),
                    'Institutions Count': result.get('institutions_distinct_count'), 
                    'Corresponding Institution IDs': result.get('corresponding_institution_ids', []),
                    'Locations': locations,
                    'Source Display Name': source_display_name,
                    'Topics': [{'ID': topic.get('id'), 'Name': topic.get('display_name')} for topic in result.get('topics', [])],
                    'Landing Page URL': landing_page, 'HTML Missing': html_missing
                })
            except Exception as e:
                print(f"Error processing result {result}: {e}")
                continue
        
        PAGE_FN.write_text(json.dumps({'data_list': data_list, 'missing_html_count': missing_html_count}))
        end_time = time.time()
        print(f'Page {page}: end_time={end_time}, run_time={end_time - start_time}')
        return data_list, missing_html_count

def load_checkpoint():
    """
    Loads the last processed page number from a checkpoint file.
    This helps in resuming the data fetching process from the last saved state in case of interruptions.

    Returns:
    dict: A dictionary containing the last processed page number.
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'last_page_processed': 0}

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f)

def load_summary():
    """
    Loads a summary of the data processing so far from a summary file.
    This includes the total number of entries processed and the number of HTML files that were missing.

    Returns:
    dict: A dictionary containing the summary of data processing.
    """
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, 'r') as f:
            return json.load(f)
    return {
        'Total Entries': 0,
        'Total Missing HTML': 0
    }

def save_summary(summary):
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)

async def main():
    all_data = []
    START_PAGE = 1
    LAST_PAGE = 1000 # excluding
    BATCH_SIZE = 10
    
    cumulative_summary = {'Total Entries': 0, 'Total Missing HTML': 0}
    
    for start_range in range(START_PAGE, LAST_PAGE, BATCH_SIZE):
        tasks = []
        for current_page in range(start_range, start_range + BATCH_SIZE): 
            if current_page == LAST_PAGE:
                break
            tasks.append(process_data(current_page)) 
            
        results = await asyncio.gather(*tasks)
        for data, missing_html_count in results:
            all_data.extend(data)
            cumulative_summary['Total Entries'] += len(data)
            cumulative_summary['Total Missing HTML'] += missing_html_count

    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False, mode='a', header=not os.path.exists(OUTPUT_FILE))
    print(f"Total entries processed: {cumulative_summary['Total Entries']}")
    print(f"Total missing HTML files: {cumulative_summary['Total Missing HTML']}")

if __name__ == '__main__':
    asyncio.run(main())

