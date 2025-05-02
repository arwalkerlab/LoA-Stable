import subprocess
import sys
import os
import csv
import requests
from src.utils import download_ollama
from src.classes import JobSettings
from itertools import combinations

# URLs for PubMed Central API
ESEARCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
EFETCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'

def scrape_and_extract_concurrent(job_settings: JobSettings):
    """
    Concurrently scrape papers from multiple sources and extract information based on a given schema.

    Args:
    job_settings (JobSettings): A class containing all configuration parameters.

    Returns:
    None
    """

    # Prepare search terms
    # Filter out 'none' from the definite search terms
    def_terms = [term for term in job_settings.def_search_terms if term.lower() != 'none']
    # Filter out 'none' from the maybe search terms
    maybe_terms = [term for term in job_settings.maybe_search_terms if term.lower() != 'none']

    # This will hold a list of lists (each sub-list is one combination of terms)
    all_search_terms = []

    # Generate every possible combination of the maybe-terms (including the empty subset)
    # and prepend the definite terms to that subset
    for r in range(len(maybe_terms) + 1):
        for combo in combinations(maybe_terms, r):
            all_search_terms.append(def_terms + list(combo))

    # Set up output directory
    output_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Ping ollama port to see if it is running
    try:
        response = requests.get("http://localhost:11434")
    
    except:
        try:
            response.status_code = 404
        except:
            is_ollama_running = False
        
    # If so, cool, if not, start it!
    try:
        if response.status_code == 200:
            is_ollama_running = True
        else:
            is_ollama_running = False
    except:
        is_ollama_running = False

    # Check for Ollama binary and start server if not running
    if not is_ollama_running:
        if not os.path.isfile('ollama'):
            print("ollama binary not found. Downloading the latest release...")
            download_ollama()

        # Start Ollama server
        subprocess.Popen(["./ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Count processed papers for each source
    source_counts = {
        'pubmed': 0,
        'arxiv': 0,
        'chemrxiv': 0,
        'SO': 0,
        'unpaywall': 0
    }

    if os.path.exists(job_settings.files.csv):
        with open(job_settings.files.csv, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                if row:  # Check if row is not empty
                    paper_id = row[-1]
                    if paper_id.startswith('pubmed_'):
                        source_counts['pubmed'] += 1
                    elif paper_id.startswith('arxiv_'):
                        source_counts['arxiv'] += 1
                    elif paper_id.startswith('chemrxiv_'):
                        source_counts['chemrxiv'] += 1
                    elif paper_id.startswith('SO_'):
                        source_counts['SO'] += 1
                    elif paper_id.startswith('unpaywall_'):
                        source_counts['unpaywall'] += 1

    # Adjust retmax for each source
    original_retmax = job_settings.scrape.retmax
    pubmed_retmax = max(0, original_retmax - source_counts['pubmed'])
    arxiv_retmax = max(0, original_retmax - source_counts['arxiv'])
    chemrxiv_retmax = max(0, original_retmax - source_counts['chemrxiv'])
    scienceopen_retmax = max(0, original_retmax - source_counts['SO'])
    unpaywall_retmax = max(0, original_retmax - source_counts['unpaywall'])

    for search_terms in all_search_terms:
        # Perform searches and extractions
        if job_settings.scrape.scrape_pubmed and pubmed_retmax > 0:
            job_settings.scrape.retmax = pubmed_retmax
            print(f"Searching PubMed for {pubmed_retmax} papers...")
            from src.databases.pubmed import pubmed_search
            pubmed_search(job_settings, search_terms)
            job_settings.scrape.retmax = original_retmax

        if job_settings.scrape.scrape_arxiv and arxiv_retmax > 0:
            job_settings.scrape.retmax = arxiv_retmax
            print(f"Searching arXiv for {arxiv_retmax} papers...")
            from src.databases.arxiv import arxiv_search
            arxiv_search(job_settings, search_terms, 'arxiv')
            job_settings.scrape.retmax = original_retmax

        if job_settings.scrape.scrape_arxiv and chemrxiv_retmax > 0:
            job_settings.scrape.retmax = chemrxiv_retmax
            print(f"Searching ChemRxiv for {chemrxiv_retmax} papers...")
            from src.databases.arxiv import arxiv_search
            arxiv_search(job_settings, search_terms, 'chemrxiv')
            job_settings.scrape.retmax = original_retmax

        if job_settings.scrape.scrape_scienceopen and scienceopen_retmax > 0:
            job_settings.scrape.retmax = scienceopen_retmax
            print(f"Searching ScienceOpen for {scienceopen_retmax} papers...")
            from src.databases.science_open import scrape_scienceopen
            scrape_scienceopen(job_settings, search_terms)
            job_settings.scrape.retmax = original_retmax

        if job_settings.scrape.scrape_unpaywall and unpaywall_retmax > 0:
            job_settings.scrape.retmax = unpaywall_retmax
            if job_settings.scrape.email is None:
                print("Email is required for Unpaywall search. Skipping Unpaywall.")
            else:
                print(f"Searching Unpaywall for {unpaywall_retmax} papers...")
                from src.databases.unpaywall import unpaywall_search
                job_settings.query_chunks = [search_terms]
                unpaywall_search(job_settings)
            job_settings.scrape.retmax = original_retmax

    print("Concurrent scraping and extraction completed.")

    # If not in auto mode, restart the script
    if job_settings.auto is None:
        python = sys.executable
        os.execl(python, python, *sys.argv)