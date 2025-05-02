import os
import requests
from tqdm import tqdm
import time
import logging
from urllib.parse import quote
from bs4 import BeautifulSoup
from selenium.common import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from src.extract import extract
from src.utils import (get_chrome_driver, is_file_processed, write_to_csv)
from src.classes import JobSettings

def scrape_scienceopen(job_settings:JobSettings, search_terms):  # retmax, concurrent=False, schema_file=None, user_instructions=None, model_name_version=None
    """
    Scrape articles from ScienceOpen based on search terms, with optional concurrent extraction.

    Args:
    search_terms (list): A list of search terms to use on ScienceOpen.
    retmax (int): The maximum number of articles to scrape.
    concurrent (bool): Whether to perform extraction concurrently with downloading.
    schema_file (str): Path to the schema file for extraction (required if concurrent is True).
    user_instructions (str): Instructions for the extraction model (required if concurrent is True).
    model_name_version (str): Name and version of the model to use for extraction (required if concurrent is True).

    Returns:
    list: A list of filenames of the scraped PDFs.
    """

    # First things first, if this whole function is dependent on Chrome being installed, we have to make sure it's installed and bail out if not.
    if os.popen("which google-chrome").read().strip() == "":
        print("Google Chrome is not installed on this system.  Aborting use of ScienceOpen database.")
        return []

    if job_settings.concurrent and (job_settings.files.schema is None or job_settings.extract.user_instructions is None or job_settings.model_name_version is None):
        raise ValueError("schema_file, user_instructions, and model_name_version must be provided when concurrent is True")

    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.WARNING)

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-crash-reporter")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-in-process-stack-traces")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--output=/dev/null")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])

    try:
        driver = get_chrome_driver()
        driver.set_page_load_timeout(60)  # Set timeout to 60 seconds
        wait = WebDriverWait(driver, 10)

        url = (
            f"https://www.scienceopen.com/search#('v'~4_'id'~''_'queryType'~1_'context'~null_'kind'~77_'order'~0_"
            f"'orderLowestFirst'~false_'query'~'{' '.join(search_terms)}'_'filters'~!("
            f"'kind'~84_'openAccess'~true)*_'hideOthers'~false)")

        print("Attempting to load ScienceOpen URL...")
        driver.get(url)
        print("ScienceOpen URL loaded successfully")

        scraped_links_dir = os.path.join(os.getcwd(), 'search_info', 'SO_searches')
        os.makedirs(scraped_links_dir, exist_ok=True)

        scraped_links_file_path = os.path.join(scraped_links_dir, f"{'_'.join(search_terms)}.txt")

        if os.path.exists(scraped_links_file_path):
            with open(scraped_links_file_path, 'r') as file:
                scraped_links = file.readlines()
        else:
            scraped_links = []

        article_links = []
        while len(article_links) < job_settings.scrape.retmax:
            new_links = driver.find_elements(By.CSS_SELECTOR, 'div.so-article-list-item > div > h3 > a')
            new_links = [link.get_attribute('href') for link in new_links if
                         link.get_attribute('href') not in scraped_links]

            article_links.extend(new_links)

            try:
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.so--tall'))
                )
                driver.execute_script("arguments[0].click();", load_more_button)
                time.sleep(2)
            except TimeoutException:
                print("No more results to load or couldn't find 'Load More' button.")
                break
            except StaleElementReferenceException:
                print("No more results to load.")
                break
            except Exception as other:
                print(f"An unknown exception occurred, please let the dev know: {other}")
                break

        start_time = time.time()
        pbar = tqdm(total=job_settings.scrape.retmax, dynamic_ncols=True)
        count = 0
        scraped_files = []
        failed_articles = []

        for link in article_links:
            if count >= job_settings.scrape.retmax:
                break

            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[-1])

            try:
                driver.get(link)

                try:
                    pdf_link_element = wait.until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, '#id2e > li:nth-child(1) > a:nth-child(1)')))
                    pdf_link = pdf_link_element.get_attribute('href')
                except (TimeoutException, NoSuchElementException):
                    print(f"PDF link not found for article: {link}")
                    failed_articles.append(link)
                    continue

                try:
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    doi_element = soup.find('meta', attrs={'name': 'citation_doi'})
                    if doi_element is not None:
                        doi = doi_element.get('content')
                        encoded_doi = quote(doi, safe='')
                    else:
                        print(f"DOI not found for article: {link}")
                        failed_articles.append(link)
                        continue
                except Exception as e:
                    print(f"Error occurred while extracting DOI for article: {link}")
                    print(f"Error: {e}")
                    failed_articles.append(link)
                    continue

                filename = f"SO_{encoded_doi}.pdf"
                file_path = os.path.join(os.getcwd(), 'scraped_docs', filename)

                if os.path.exists(file_path):
                    print(f"{filename} already exists.")
                    if job_settings.concurrent and not is_file_processed(job_settings.files.csv, filename):
                        print(f"{filename} not extracted for this task; performing extraction...")
                        try:
                            extracted_data = extract(file_path, job_settings)
                            if extracted_data:
                                print(f"Successfully extracted data from {filename}")
                            else:
                                print(f"Failed to extract data from {filename}")
                                failed_result = [["failed" for _ in range(job_settings.extract.num_columns)] + [
                                    os.path.splitext(os.path.basename(file_path))[0]]]
                                write_to_csv(failed_result, job_settings.extract.headers,
                                             filename=job_settings.files.csv)
                        except Exception as e:
                            print(f"Error extracting data from {filename}: {e}")
                    continue

                try:
                    pdf_response = requests.get(pdf_link)
                    with open(file_path, 'wb') as f:
                        f.write(pdf_response.content)

                    count += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_pdf = elapsed_time / count
                    est_time_remaining = avg_time_per_pdf * (job_settings.scrape.retmax - count)
                    pbar.set_description(
                        f"DOI: {doi}, Count: {count}/{job_settings.scrape.retmax}, Avg time per PDF: {avg_time_per_pdf:.2f}s, Est. time remaining: {est_time_remaining:.2f}s")
                    pbar.update(1)

                    scraped_files.append(filename)

                    with open(scraped_links_file_path, 'a') as file:
                        file.write(f"{link}\n")

                    if job_settings.concurrent:
                        try:
                            extracted_data = extract(file_path, job_settings)
                            if extracted_data:
                                print(f"Successfully extracted data from {filename}")
                            else:
                                print(f"Failed to extract data from {filename}")
                                failed_result = [["failed" for _ in range(job_settings.extract.num_columns)] + [
                                    os.path.splitext(os.path.basename(file_path))[0]]]
                                write_to_csv(failed_result, job_settings.extract.headers,
                                             filename=job_settings.files.csv)
                        except Exception as e:
                            print(f"Error extracting data from {filename}: {e}")

                except Exception as e:
                    print(f"Error occurred while downloading PDF for article: {link}")
                    print(f"Error: {e}")
                    failed_articles.append(link)
            except Exception as e:
                print(f"Error occurred while processing article: {link}")
                print(f"Error: {e}")
                failed_articles.append(link)
            finally:
                driver.close()
                driver.switch_to.window(driver.window_handles[0])

        driver.quit()
        pbar.close()

        print(f"Scraping completed. Successfully scraped {count} articles.")
        print(f"Failed to scrape {len(failed_articles)} articles.")
        print("Failed articles:")
        for article in failed_articles:
            print(article)

        return scraped_files

    except TimeoutException:
        print(
            "Timeout occurred while loading ScienceOpen. This could be due to slow internet connection or the website being down.")
        print("Skipping ScienceOpen scraping for this search term.")
        if driver:
            driver.quit()
        return []
    except Exception as e:
        print(f"An unexpected error occurred while scraping ScienceOpen: {e}")
        print("Skipping ScienceOpen scraping for this search term.")
        if driver:
            driver.quit()
        return []
