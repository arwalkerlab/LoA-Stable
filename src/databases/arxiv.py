import os
import requests
import time
import subprocess
from bs4 import BeautifulSoup
from src.extract import extract
from src.utils import (doi_to_filename, is_file_processed, write_to_csv, begin_ollama_server)
from src.classes import JobSettings


def arxiv_search(job_settings: JobSettings, search_terms, repository):
    """
    Search and download papers from arXiv or ChemRxiv repositories, with optional concurrent extraction.
    """
    if job_settings.concurrent and any([job_settings.files.schema is None,
                                        job_settings.extract.user_instructions is None,
                                        job_settings.model_name_version is None]):
        raise ValueError(
            "schema_file, user_instructions, and model_name_version must be provided when concurrent is True")

    print(f"Starting {repository} search with terms: {search_terms}")
    query = "+AND+".join(search_terms).replace(' ', '%20')
    print(f"Constructed query: {query}")

    MAX_RETRIES = 3
    SEARCH_MAX_RETRIES = 10
    scraped_files = []
    processed_papers = set()

    # Initial API call to get total count
    if repository == 'arxiv':
        initial_url = f"https://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1"
    elif repository == 'chemrxiv':
        initial_url = f"https://chemrxiv.org/engage/chemrxiv/public-api/v1/items?term={query}&skip=0&limit=1"
    else:
        print(f"Unsupported repository: {repository}")
        return []

    try:
        response = requests.get(initial_url)
        response.raise_for_status()
        if repository == 'arxiv':
            soup = BeautifulSoup(response.text, "xml")
            total_results = int(soup.find('opensearch:totalResults').text)
        elif repository == 'chemrxiv':
            json_data = response.json()
            total_results = json_data['totalCount']
    except Exception as e:
        print(f"Error fetching initial results: {e}")
        return []

    print(f"Total results found: {total_results}")

    start = 0
    while start < min(total_results, job_settings.scrape.retmax):
        current_max = min(50, job_settings.scrape.retmax - start)

        if repository == 'arxiv':
            api_url = f"https://export.arxiv.org/api/query?search_query=all:{query}&start={start}&max_results={current_max}"
        elif repository == 'chemrxiv':
            api_url = f"https://chemrxiv.org/engage/chemrxiv/public-api/v1/items?term={query}&skip={start}&limit={current_max}"

        print(f"Fetching results from: {api_url}")

        search_retry_count = 0
        while search_retry_count < SEARCH_MAX_RETRIES:
            try:
                print(f"Sending request to API... (Retry: {search_retry_count})")
                response = requests.get(api_url)
                print(f"API response status code: {response.status_code}")
                response.raise_for_status()

                if repository == 'arxiv':
                    entries = BeautifulSoup(response.text, "xml").find_all("entry")
                elif repository == 'chemrxiv':
                    json_data = response.json()
                    entries = json_data.get('itemHits', [])

                print(f"Number of entries found: {len(entries)}")

                if not entries:
                    print("No more results found. Exiting.")
                    return scraped_files

                for entry in entries:
                    if repository == 'arxiv':
                        pdf_link = entry.find('link', {'title': 'pdf'})['href']
                        doi = entry.find('id').text.split('/')[-1]
                    elif repository == 'chemrxiv':
                        pdf_link = entry['item']['asset']['original']['url']
                        doi = entry['item'].get('doi', '')

                    if doi in processed_papers:
                        continue

                    processed_papers.add(doi)
                    filename = f"{repository}_{doi_to_filename(doi)}.pdf"
                    file_path = os.path.join(os.getcwd(), 'scraped_docs', filename)

                    if os.path.exists(file_path):
                        print(f"{filename} already downloaded.")
                        if job_settings.concurrent and not is_file_processed(job_settings.files.csv, filename):
                            print(f"{filename} not extracted for this task; performing extraction...")
                            restart_tries = 0
                            while restart_tries < 2:
                                try:
                                    extracted_data = extract(file_path, job_settings)
                                    if extracted_data:
                                        print(f"Successfully extracted data from {filename}")
                                        restart_tries = 2
                                        break
                                    else:
                                        print(f"Failed to extract data from {filename}")
                                        failed_result = [["failed" for _ in range(job_settings.extract.num_columns)] + [
                                            os.path.splitext(os.path.basename(file_path))[0]]]
                                        write_to_csv(failed_result, job_settings.extract.headers,
                                                     filename=job_settings.files.csv)
                                        restart_tries = 2
                                        break
                                except Exception as e:
                                    if '500' in str(e):
                                        restart_tries += 1
                                        
                                        print("Ollama either crashed, or the model you are trying to use is too large, trying to restart...")
                                        
                                        # Start ollama server
                                        begin_ollama_server()
                                        
                                    else:
                                        print(f"Error extracting data from {filename}: {e}")
                                        break
                        continue

                    retry_count = 0
                    while retry_count < MAX_RETRIES:
                        try:
                            print(f"PDF link found: {pdf_link}")
                            if pdf_link:
                                pdf_response = requests.get(pdf_link)
                                pdf_response.raise_for_status()
                                with open(file_path, 'wb') as f:
                                    f.write(pdf_response.content)
                                scraped_files.append(filename)
                                print(f"Successfully downloaded PDF for {filename}.")

                                if job_settings.concurrent:
                                    concurrent_tries = 0
                                    while concurrent_tries < 2:
                                        try:
                                            extracted_data = extract(file_path, job_settings)
                                            if extracted_data:
                                                print(f"Successfully extracted data from {filename}")
                                                concurrent_tries = 2
                                                break
                                            else:
                                                print(f"Failed to extract data from {filename}")
                                                failed_result = [
                                                    ["failed" for _ in range(job_settings.extract.num_columns)] + [
                                                        os.path.splitext(os.path.basename(file_path))[0]]]
                                                write_to_csv(failed_result, job_settings.extract.headers,
                                                             filename=job_settings.files.csv)
                                                concurrent_tries = 2
                                                break
                                        except Exception as e:
                                            if '500' in str(e):
                                                print("Ollama either crashed, or the model you are trying to use is too large, trying to restart...")
                                                
                                                # Start ollama server
                                                begin_ollama_server()
                                                
                                                concurrent_tries += 1
                                                
                                            else:
                                                print(f"Error extracting data from {filename}: {e}")
                                                break
                                break
                            else:
                                print(f"No PDF link found for {filename}. Skipping.")
                                break
                        except requests.exceptions.RequestException as e:
                            retry_count += 1
                            print(f"Error downloading PDF. Retry attempt {retry_count}/{MAX_RETRIES}. Error: {e}")
                            time.sleep(10)
                    else:
                        print(f"Failed to download {filename} after {MAX_RETRIES} attempts. Skipping.")

                    if len(processed_papers) % 10 == 0:
                        print(f"Progress: Processed {len(processed_papers)} unique papers")

                    if len(scraped_files) >= job_settings.scrape.retmax:
                        print(f"Reached retmax of {job_settings.scrape.retmax}. Stopping search.")
                        return scraped_files

                start += len(entries)
                break  # Break out of the retry loop if successful

            except (requests.exceptions.RequestException, Exception) as e:
                search_retry_count += 1
                print(
                    f"Failed to fetch search results. Retry attempt {search_retry_count}/{SEARCH_MAX_RETRIES}. Error: {e}")
                time.sleep(10)
        else:
            print(f"Failed to fetch results after {SEARCH_MAX_RETRIES} attempts. Moving to next batch.")

    print(f"{repository} search completed. Total unique papers processed: {len(processed_papers)}")
    print(f"Total files scraped: {len(scraped_files)}")
    return scraped_files