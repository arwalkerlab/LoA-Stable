import sqlite3
import sys
import os
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
from src.classes import JobSettings

# Constants
ESEARCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
EFETCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'

def main_scrape_pubmed(job_settings: JobSettings):
    if job_settings.scrape.scrape_pubmed:
        from src.databases.pubmed import pubmed_search
        for chunk in job_settings.query_chunks:
            print("Current search: " + str(chunk))
            scraped_files = pubmed_search(job_settings, chunk)
            with open(job_settings.files.search_info_file, 'a') as f:
                f.write('\n'.join(scraped_files) + '\n')

def main_scrape_arxiv(job_settings: JobSettings):
    if job_settings.scrape.scrape_arxiv:
        from src.databases.arxiv import arxiv_search
        repositories = ['arxiv', 'chemrxiv']  # Decided to remove bio and med, as their api's are not very good
        # I could be convinced to add them back, but because the api doesn't allow for search terms, I would need to write code
        # to build a local database and search that, which would be time-consuming and a hassle for the end user.
        for repository in repositories:
            for chunk in job_settings.query_chunks:
                print("Current search: " + str(chunk))
                scraped_files = arxiv_search(job_settings, chunk, repository)
                if scraped_files is not None:
                    with open(job_settings.files.search_info_file, 'a') as f:
                        f.write('\n'.join(scraped_files) + '\n')

def main_scrape_science_open(job_settings: JobSettings):
    if job_settings.scrape.scrape_scienceopen:
        from src.databases.science_open import scrape_scienceopen
        for chunk in job_settings.query_chunks:
            print(f"Now running ScienceOpen scrape for {chunk}")
            scraped_files = scrape_scienceopen(job_settings, chunk)
            with open(job_settings.files.search_info_file, 'a') as f:
                f.write('\n'.join(scraped_files) + '\n')

def main_scrape_unpaywall(job_settings: JobSettings):
    if job_settings.scrape.scrape_unpaywall:
        from src.databases.unpaywall import unpaywall_search
        scraped_files = unpaywall_search(job_settings)
        with open(job_settings.files.search_info_file, 'a') as f:
            f.write('\n'.join(scraped_files) + '\n')

def main_scrape_custom_db(job_settings: JobSettings):
    if job_settings.scrape.scrape_custom_db:
        # Create a connection to the SQLite database
        conn = sqlite3.connect(str(os.getcwd()) + '/customdb/metadata.db')
        c = conn.cursor()

        scraped_files = []

        # Iterate over all search terms
        for chunk in job_settings.query_chunks:
            print(f"Current search: {chunk}")

            # Create the SQL query
            query = 'SELECT * FROM metadata WHERE '
            query += ' AND '.join([f'title LIKE ?' for _ in chunk])
            query += ' ORDER BY title LIMIT ?'

            # Create the parameters for the SQL query
            params = tuple([f'%{term}%' for term in chunk] + [job_settings.scrape.retmax])

            # Execute the SQL query
            c.execute(query, params)

            # Fetch all the results
            results = c.fetchall()
            print(f"Found {len(results)} results for search terms: {chunk}")

            # Iterate over the results
            for result in results:
                # Extract and format the DOI
                doi = result[0]
                encoded_doi = quote(doi, safe='')
                print(f"Processing DOI: {doi}")

                # Create the URL for the paper
                url = job_settings.scrape.base_url + quote(doi)
                print(f"URL: {url}")

                # Get the webpage content
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}")
                    continue

                # Check if the request was successful
                if response.status_code == 200:
                    # Parse the HTML
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Find the PDF link
                    pdf_link = None
                    for button in soup.find_all('button'):
                        if '.pdf' in button.get('onclick', ''):
                            pdf_link = 'https:' + button['onclick'].split("'")[1]
                            break

                    if pdf_link is not None:
                        # Download the PDF
                        try:
                            pdf_response = requests.get(pdf_link)
                            pdf_response.raise_for_status()
                        except requests.exceptions.RequestException as e:
                            print(f"Request failed: {e}")
                            continue

                        if pdf_response.status_code == 200:
                            # Extract the PDF from the response
                            pdf = pdf_response.content

                            # Create the path for the PDF file
                            pdf_filename = f'{encoded_doi}.pdf'
                            pdf_path = os.path.join(os.getcwd(), 'scraped_docs', pdf_filename)
                            print(f"Saving PDF to: {pdf_path}")

                            # Write the PDF to a file
                            with open(pdf_path, 'wb') as f:
                                f.write(pdf)

                            scraped_files.append(pdf_filename)
                        else:
                            print(f"Failed to download PDF. Status code: {pdf_response.status_code}")
                    else:
                        print("No PDF link found on the page. Saving the webpage as HTML.")

                        # Create the path for the HTML file
                        html_filename = f'{doi}.html'
                        html_path = os.path.join(os.getcwd(), 'scraped_docs', html_filename)
                        print(f"Saving HTML to: {html_path}")

                        # Write the HTML to a file
                        with open(html_path, 'w') as f:
                            f.write(response.text)

                        scraped_files.append(html_filename)
                else:
                    print(f"Failed to access the webpage. Status code: {response.status_code}")

        # Write the scraped files to the search info file
        with open(job_settings.files.search_info_file, 'a') as f:
            f.write('\n'.join(scraped_files) + '\n')

        # Close the connection to the database
        conn.close()

def scrape(job_settings: JobSettings):
    """
    Main function to scrape papers from various sources based on user input or provided arguments.

    This function coordinates the scraping process across multiple repositories including
    PubMed, arXiv, ChemRxiv, ScienceOpen, Unpaywall, and a custom database if specified.
    It handles user input for search terms and options when not in automatic mode.

    Args:
    args (dict): A dictionary containing scraping parameters. If empty or None, user input is requested.

    Returns:
    None: The function writes scraped file information to a text file and doesn't return any value.
    """

    # Scrape from PubMed if selected
    main_scrape_pubmed(job_settings)

    # Scrape from arXiv and ChemRxiv if selected
    main_scrape_arxiv(job_settings)

    # Scrape from ScienceOpen if selected
    main_scrape_science_open(job_settings)

    # Scrape from Unpaywall if selected
    main_scrape_unpaywall(job_settings)

    # Scrape from custom database if selected
    main_scrape_custom_db(job_settings)

    # If not in automatic mode, restart the script
    if job_settings.auto is None:
        python = sys.executable
        os.execl(python, python, *sys.argv)

    # If in automatic mode, return None
    else:
        return None
