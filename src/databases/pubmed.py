import os
import requests
import time
import subprocess
import xml.etree.ElementTree as ET
from src.extract import extract
from src.utils import (is_file_processed, write_to_csv, begin_ollama_server)
from src.classes import JobSettings

# Constants
ESEARCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
EFETCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'

def pubmed_search(job_settings: JobSettings, search_terms): # concurrent=False, schema_file=None, user_instructions=None, model_name_version="mistral:7b-instruct-v0.2-q8_0"  ):
    """
    Search and download papers from PubMed Central, with optional concurrent extraction.

    Args:
    search_terms (list): List of search terms.
    retmax (int): Maximum number of results to retrieve.
    concurrent (bool): Whether to perform extraction concurrently with downloading.
    schema_file (str): Path to the schema file for extraction (required if concurrent is True).
    user_instructions (str): Instructions for the extraction model (required if concurrent is True).
    model_name_version (str): Name and version of the model to use for extraction (required if concurrent is True).

    Returns:
    list: List of filenames of downloaded papers (and extracted data if concurrent is True).
    """
    if job_settings.concurrent and (job_settings.files.schema is None or job_settings.extract.user_instructions is None or job_settings.model_name_version is None):
        raise ValueError("schema_file, user_instructions, and model_name_version must be provided when concurrent is True")

    # csv_file = os.path.join(os.getcwd(), 'results', f"{model_name}_{model_version}_{os.path.splitext(schema_file)[0].split('/')[-1]}.csv")

    esearch_params = {
        'db': 'pmc',
        'term': " AND ".join(search_terms),
        'retmode': 'json',
        'retmax': job_settings.scrape.retmax
    }

    print("Now performing esearch...")
    try:
        esearch_response = requests.get(ESEARCH_URL, params=esearch_params)
        esearch_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []

    esearch_data = esearch_response.json()

    if 'esearchresult' in esearch_data:
        uid_list = esearch_data['esearchresult']['idlist']

        if not uid_list:
            print("No search results found.")
            return []

        # downloaded_files = os.listdir(os.path.join(os.getcwd(), 'scraped_docs'))
        # downloaded_files = [file.replace("pubmed_", "").replace(".xml", "") for file in downloaded_files if
        #                     "pubmed_" in file]
        downloaded_files = [file.replace("pubmed_", "").replace(".xml", "") for file in os.listdir(os.path.join(os.getcwd(), 'scraped_docs')) if "pubmed_" in file] # one-liner to avoid storing excess data, even temporarily.

        num_downloaded = len([uid for uid in uid_list if uid in downloaded_files])

        print(f"{num_downloaded} files already downloaded for this search.")

        scraped_files = []

        for uid in uid_list:
            filename = f"pubmed_{uid}.xml"
            file_path = os.path.join(os.getcwd(), 'scraped_docs', filename)
            if uid not in downloaded_files:
                if num_downloaded >= job_settings.scrape.retmax:
                    print("Reached maximum number of downloads for this search. Stopping.")
                    return scraped_files

                efetch_params = {
                    'db': 'pmc',
                    'id': uid,
                    'retmode': 'xml',
                    'rettype': 'full'
                }

                print(f"Now performing efetch for UID {uid}...")
                try:
                    efetch_response = requests.get(EFETCH_URL, params=efetch_params)
                    efetch_response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}")
                    continue

                xml_data = efetch_response.text
                root = ET.fromstring(xml_data)

                if root.find(".//body"):
                    with open(file_path, 'w') as f:
                        f.write(xml_data)
                    num_downloaded += 1
                    scraped_files.append(filename)

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
                                    failed_result = [["failed" for _ in range(job_settings.extract.num_columns)] + [
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
                else:
                    print(f"Full text not available for UID {uid}. Skipping.")

                time.sleep(1 / 2)

            elif (not is_file_processed(job_settings.files.csv, filename)) and job_settings.concurrent:
                print(f"{filename} already downloaded, but not extracted from for this task; performing extraction...")
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
                            write_to_csv(failed_result, job_settings.extract.headers, filename=job_settings.files.csv)
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

        return scraped_files
    else:
        print("No 'esearchresult' in the API response.")
        return []
