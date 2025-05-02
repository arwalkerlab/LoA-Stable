import os
import requests
import time
import json
from datetime import datetime
from src.extract import extract
from src.utils import (doi_to_filename, is_file_processed, write_to_csv)
from src.classes import JobSettings

def read_api_count():
    """
    Read the current API call count for Unpaywall.

    Returns:
    tuple: (date, count) of the last API call count.
    """
    try:
        with open(os.path.join(os.getcwd(), 'search_info', 'unpaywall', 'api_call_count.txt'), "r") as f:
            date,count = f.read().split("\n")
        return date, int(count)
    except FileNotFoundError:
        return None, 0

def write_api_count(date, count):
    """
    Write the current API call count for Unpaywall.

    Args:
    date (str): Current date.
    count (int): Current API call count.
    """
    with open(os.path.join(os.getcwd(), 'search_info', 'unpaywall', 'api_call_count.txt'), "w") as f:
        f.write(f"{date}\n{count}")

def read_last_state():
    """
    Read the last state of Unpaywall search.

    Returns:
    tuple: (last_chunk, last_page) of the last search state.
    """
    try:
        with open(os.path.join(os.getcwd(), 'search_info', 'unpaywall', 'last_state.txt'), "r") as f:
            data = json.load(f)
        return data['last_chunk'], data['last_page']
    except FileNotFoundError:
        return None, 1

def write_last_state(last_chunk, last_page):
    """
    Write the current state of Unpaywall search.

    Args:
    last_chunk (list): Last processed search chunk.
    last_page (int): Last processed page number.
    """
    with open(os.path.join(os.getcwd(), 'search_info', 'unpaywall', 'last_state.txt'), "w") as f:
        json.dump({'last_chunk': last_chunk, 'last_page': last_page}, f)


def download_pdf(url, doi):
    """
    Download a PDF from a given URL.

    Args:
    url (str): URL of the PDF.
    doi (str): DOI of the paper.

    Returns:
    str: Filename of the downloaded PDF, or None if download failed.
    """
    try:
        pdf_response = requests.get(url)
        pdf_response.raise_for_status()
        filename = f"unpaywall_{doi_to_filename(doi)}.pdf"
        with open(os.path.join(os.getcwd(), 'scraped_docs', filename), 'wb') as f:
            f.write(pdf_response.content)
        return filename
    except requests.exceptions.RequestException as e:
        print(f"PDF download failed: {e}")
        return None

def unpaywall_search(job_settings:JobSettings): #, query_chunks, retmax, email, concurrent=False, schema_file=None, user_instructions=None, model_name_version=None
    """
    Search and download papers from Unpaywall API, with optional concurrent extraction.

    Args:
    query_chunks (list): List of search term chunks to query.
    retmax (int): Maximum number of results to retrieve.
    email (str): Email address for Unpaywall API authentication.
    concurrent (bool): Whether to perform extraction concurrently with downloading.
    schema_file (str): Path to the schema file for extraction (required if concurrent is True).
    user_instructions (str): Instructions for the extraction model (required if concurrent is True).
    model_name_version (str): Name and version of the model to use for extraction (required if concurrent is True).

    Returns:
    list: List of filenames of downloaded papers and JSON metadata.
    """
    if job_settings.concurrent and any([job_settings.files.schema is None, job_settings.extract.user_instructions is None, job_settings.model_name_version is None]):
        raise ValueError("schema_file, user_instructions, and model_name_version must be provided when concurrent is True")

    # Read the last API call count and date
    last_date, api_count = read_api_count()
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(os.path.join(os.getcwd(), 'search_info', 'unpaywall'), exist_ok=True)

    # Reset API count if it's a new day
    if last_date != today:
        api_count = 0

    # Check if daily API limit is reached
    if api_count >= 100000:
        print("Reached daily API call limit. Try again tomorrow.")
        return []

    # Read the last search state for resuming interrupted searches
    last_chunk, last_page = read_last_state()

    resume = False if last_chunk is None else True
    scraped_files = []
    total_downloaded = 0

    # Iterate through search term chunks
    for chunk in job_settings.query_chunks:
        if resume and chunk != last_chunk:
            continue
        query = " AND ".join(chunk)
        page = last_page if resume and chunk == last_chunk else 1

        # Continue searching while within limits
        while total_downloaded <= job_settings.scrape.retmax and api_count < 100000:
            unpaywall_params = {
                'query': query,
                'is_oa': 'true',
                'email': job_settings.scrape.email,
                'page': page
            }

            print(f"Now scraping Unpaywall for {chunk}, page {page}")

            # Make API request
            try:
                unpaywall_response = requests.get("https://api.unpaywall.org/v2/search", params=unpaywall_params)
                unpaywall_response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                return scraped_files

            # Update API call count
            api_count += 1
            if api_count >= 100000:
                print("Reached daily API call limit. Exiting.")
                write_api_count(today, api_count)
                write_last_state(chunk, page)
                return scraped_files

            write_api_count(today, api_count)

            unpaywall_data = json.loads(unpaywall_response.text)

            # Check if results are present
            if 'results' not in unpaywall_data:
                print("No results found in the Unpaywall API response.")
                break

            # Extract DOIs from the response
            doi_list = [result['response']['doi'] for result in unpaywall_data['results'] if
                        'response' in result and 'doi' in result['response']]

            if not doi_list:
                print("No DOIs found in the Unpaywall API response.")
                break

            # Process each DOI
            for doi in doi_list:
                if total_downloaded >= job_settings.scrape.retmax:
                    print(f"Reached retmax of {job_settings.scrape.retmax}. Stopping search.")
                    return scraped_files

                pdf_filename = f"unpaywall_{doi_to_filename(doi)}.pdf"
                json_filename = f"unpaywall_{doi_to_filename(doi)}.json"
                pdf_path = os.path.join(os.getcwd(), 'scraped_docs', pdf_filename)
                json_path = os.path.join(os.getcwd(), 'scraped_docs', json_filename)

                # Check if files already exist
                if any([os.path.exists(pdf_path),os.path.exists(json_path)]):
                    print(f"Files for DOI {doi} already exist.")
                    if job_settings.concurrent and not is_file_processed(job_settings.files.csv, pdf_filename):
                        print(f"{pdf_filename} not extracted for this task; performing extraction...")
                        try:
                            extracted_data = extract(pdf_path, job_settings)
                            if extracted_data:
                                print(f"Successfully extracted data from {pdf_filename}")
                            else:
                                print(f"Failed to extract data from {pdf_filename}")
                                failed_result = [["failed" for _ in range(job_settings.extract.num_columns)] + [
                                    os.path.splitext(os.path.basename(pdf_path))[0]]]
                                write_to_csv(failed_result, job_settings.extract.headers,
                                             filename=job_settings.files.csv)
                        except Exception as e:
                            print(f"Error extracting data from {pdf_filename}: {e}")
                    continue

                print(f"Now fetching data for DOI {doi}...")
                try:
                    doi_response = requests.get(f"https://api.unpaywall.org/v2/{doi}?email={job_settings.scrape.email}")
                    doi_response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}")
                    continue

                # Update API call count
                api_count += 1
                if api_count >= 100000:
                    print("Reached daily API call limit. Exiting.")
                    write_api_count(today, api_count)
                    write_last_state(chunk, page)
                    return scraped_files

                write_api_count(today, api_count)

                doi_data = doi_response.json()

                # Download PDF if available
                if doi_data.get('is_oa'):
                    pdf_url = doi_data['best_oa_location']['url_for_pdf']
                    if not pdf_url:
                        print(f"No PDF URL found for DOI {doi}")
                    else: ##rearranged if/else to make it easier to follow the flow.
                        pdf_filename = download_pdf(pdf_url, doi)
                        if pdf_filename:
                            scraped_files.append(pdf_filename)
                            total_downloaded += 1

                            if job_settings.concurrent:
                                try:
                                    extracted_data = extract(pdf_path, job_settings)
                                    if extracted_data:
                                        print(f"Successfully extracted data from {pdf_filename}")
                                    else:
                                        print(f"Failed to extract data from {pdf_filename}")
                                        failed_result = [
                                            ["failed" for _ in range(job_settings.extract.num_columns)] + [
                                                os.path.splitext(os.path.basename(pdf_path))[0]]]
                                        write_to_csv(failed_result, job_settings.extract.headers,
                                                     filename=job_settings.files.csv)
                                except Exception as e:
                                    print(f"Error extracting data from {pdf_filename}: {e}")

                # Save metadata as JSON
                doi_data_str = json.dumps(doi_data, indent=4)
                with open(json_path, 'w') as f:
                    f.write(doi_data_str)
                scraped_files.append(json_filename)
                total_downloaded += 1

                time.sleep(1 / 10)  # Rate limiting

            page += 1
            write_last_state(chunk, page)

    return scraped_files