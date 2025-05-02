# Imports
import itertools
import os
import re
import hashlib
import csv
import numpy as np
from pdf2image import convert_from_path
import builtins
import datetime
import random
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
import zipfile
import io
from selenium.webdriver.chrome.service import Service
from tqdm import tqdm
from datetime import datetime
import xml.etree.ElementTree as ET
import requests
from pathlib import Path
import subprocess
import tarfile


CONVERT_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={}&format=json"
EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={}"
ESEARCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
EFETCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'

def splashbanner():
    print("""

              _____           _______                   _____          
             /\    \         /::\    \                 /\    \         
            /::\____\       /::::\    \               /::\    \        
           /:::/    /      /::::::\    \             /::::\    \       
          /:::/    /      /::::::::\    \           /::::::\    \      
         /:::/    /      /:::/~~\:::\    \         /:::/\:::\    \     
        /:::/    /      /:::/    \:::\    \       /:::/__\:::\    \    
       /:::/    /      /:::/    / \:::\    \     /::::\   \:::\    \   
      /:::/    /      /:::/____/   \:::\____\   /::::::\   \:::\    \  
     /:::/    /      |:::|    |     |:::|    | /:::/\:::\   \:::\    \ 
    /:::/____/       |:::|____|     |:::|    |/:::/  \:::\   \:::\____\

    \:::\    \        \:::\    \   /:::/    / \::/    \:::\  /:::/    /
     \:::\    \        \:::\    \ /:::/    /   \/____/ \:::\/:::/    / 
      \:::\    \        \:::\    /:::/    /             \::::::/    /  
       \:::\    \        \:::\__/:::/    /               \::::/    /   
        \:::\    \        \::::::::/    /                /:::/    /    
         \:::\    \        \::::::/    /                /:::/    /     
          \:::\    \        \::::/    /                /:::/    /      
           \:::\____\        \::/____/                /:::/    /       
            \::/    /         ~~                      \::/    /        
             \/____/                                   \/____/         
    """)

def print(text):
    """Prints the given text to the console and to a log file, with a timestamp."""
    with open(builtins.a, 'a+') as file:
        builtins.print(f"{datetime.now()}: {text}", file=file)
    builtins.print(text)


def truncate_filename(directory, filename, max_path_length=255):
    """
    Truncate a filename to ensure the entire path stays within a given maximum length 
    and append a short hash to ensure uniqueness.
    """
    # Calculate the remaining allowable length for the filename
    remaining_length = max_path_length - len(directory) - len("arXiv_") - len(
        ".pdf") - 1  # account for separators and extensions

    if len(filename) <= remaining_length:
        return filename

    # Create a short hash of the filename
    short_hash = hashlib.sha1(filename.encode()).hexdigest()[:5]
    truncated_name = filename[:remaining_length - len(short_hash) - 1]
    return f"{truncated_name}_{short_hash}"


def select_search_info_file():
    """
    Allows the user to select a search info file from the 'search_info' directory.

    Returns:
    str: Path to the selected search info file or 'All' if all files should be used.
    """
    # Get the path to the search_info directory
    search_info_dir = os.path.join(os.getcwd(), 'search_info')
    # List all .txt files in the directory
    search_info_files = [file for file in os.listdir(search_info_dir) if file.endswith('.txt') and "no_fulltext" not in file]
    # Add 'All' option at the beginning of the list3

    search_info_files.insert(0, 'All')

    # Display available files to the user
    print("Available search info files:")
    for i, file in enumerate(search_info_files):
        print(f"{i + 1}. {file}")

    # Get user's choice
    while True:
        choice = input("Enter the number of the search info file you want to use: ")
        if int(choice) == 1:
            return 'All'  # User chose to use all files
        if choice.isdigit() and 2 <= int(choice) <= len(search_info_files):
            # Return the full path of the selected file
            return os.path.join(search_info_dir, search_info_files[int(choice) - 1])
        else:
            print("Invalid choice. Please try again.")


def select_data_model_file():
    """
    Allows the user to select a data model file from the 'dataModels' directory.

    Returns:
    str: Name of the selected data model file.
    """
    # Get the path to the dataModels directory
    data_models_dir = os.path.join(os.getcwd(), 'dataModels')
    # List all .pkl files in the directory
    data_model_files = [file for file in os.listdir(data_models_dir) if file.endswith('.pkl')]

    if len(data_model_files) == 0:
        print("No data model files found!")
        return "NO_DATA_MODELS_FOUND"
    
    # Display available files to the user
    print("Available data model files:")
    for i, file in enumerate(data_model_files):
        print(f"{i + 1}. {file}")

    # Get user's choice
    while True:
        choice = input("Enter the number of the data model file you want to use: ")
        if choice.isdigit() and 1 <= int(choice) <= len(data_model_files):
            return data_model_files[int(choice) - 1]
        else:
            print("Invalid choice. Please try again.")


def get_out_id(def_search_terms_input, maybe_search_terms_input):
    """
    Processes definite and maybe search terms to generate an output directory ID and query chunks.
    The output directory ID is used in a lot of places, and generally links scrape results to extraction results.
    The query chunks are a list of combinations of search terms to iterate over.

    Args:
    def_search_terms_input (str or list): Definite search terms.
    maybe_search_terms_input (str or list): Maybe search terms.

    Returns:
    tuple: (output_directory_id, query_chunks)
    """
    # Process definite search terms
    if def_search_terms_input[0].lower() == "none" or def_search_terms_input[0] == '':
        print("No definite search terms selected.")
        def_search_terms = None
    elif isinstance(def_search_terms_input, str):
        print("String input detected for def search terms")
        def_search_terms = [term.strip() for term in def_search_terms_input.split(",")]
        def_search_terms.sort()
    elif isinstance(def_search_terms_input, list):
        print("List input detected for def search terms")
        print(f"def_search_terms_input = {def_search_terms_input}")
        def_search_terms = def_search_terms_input
        def_search_terms.sort()
        print(f"def_search_terms = {def_search_terms}")
    else:
        print(f"def search terms should be str or list, but it is instead {type(def_search_terms_input)}")

    # Process maybe search terms
    if maybe_search_terms_input[0].lower() == "none" or maybe_search_terms_input[0] == '':
        print("No maybe search terms selected, only using definite search terms.")
        maybe_search_terms = None
    elif isinstance(maybe_search_terms_input, str):
        print("String input detected for maybe search terms")
        maybe_search_terms = [term.strip() for term in maybe_search_terms_input.split(",")]
        maybe_search_terms.sort()
    elif isinstance(maybe_search_terms_input, list):
        print("List input detected for maybe search terms")
        print(f"maybe_search_terms_input = {maybe_search_terms_input}")
        maybe_search_terms = maybe_search_terms_input
        maybe_search_terms.sort()
        print(f"maybe_search_terms = {maybe_search_terms}")
    else:
        print(f"maybe search terms should be str or list, but it is instead {type(maybe_search_terms_input)}")

    # Ensure at least one type of search term is provided
    if def_search_terms is None and maybe_search_terms is None:
        print("Error: Both definite and maybe search terms cannot be None.")
        return

    # Generate output directory ID and query chunks
    if maybe_search_terms is not None:
        if def_search_terms is not None:
            # Combine definite and maybe search terms
            output_directory_id = f"{'_'.join(['def'] + def_search_terms + ['maybe'] + maybe_search_terms).replace(' ', '')}"

            # Generate all combinations of maybe search terms
            combinations = list(itertools.chain.from_iterable(
                itertools.combinations(maybe_search_terms, r) for r in range(0, len(maybe_search_terms) + 1)))
            queries = [def_search_terms + list(comb) for comb in combinations]
        else:
            # Use only maybe search terms
            output_directory_id = f"{'_'.join(['maybe'] + maybe_search_terms).replace(' ', '')}"

            # Generate combinations of maybe search terms (excluding single-term combinations)
            combinations = list(itertools.chain.from_iterable(
                itertools.combinations(maybe_search_terms, r) for r in range(1, len(maybe_search_terms) + 1)))
            queries = [list(comb) for comb in combinations]
            queries = [q for q in queries if len(q) > 1]

        print(" Ok! your adjusted searches are: " + str(queries))
        print("That's " + str(len(queries)) + " total combinations")
        if len(queries) > 100:
            print("This could take a while...")

        query_chunks = queries

    else:
        # Use only definite search terms
        output_directory_id = f"{'_'.join(['def'] + def_search_terms).replace(' ', '')}"
        query_chunks = [def_search_terms]

    return output_directory_id, query_chunks


def doi_to_filename(doi: str) -> str:
    """
    Convert a DOI to a valid filename by replacing invalid characters.

    Args:
    doi (str): The DOI to be converted.

    Returns:
    str: A filename-safe version of the DOI.
    """
    # Define replacements for invalid filename characters
    replacements = {
        '/': '_SLASH_',
        ':': '_COLON_'
    }

    # Replace each invalid character with its replacement
    filename = doi
    for char, replacement in replacements.items():
        filename = filename.replace(char, replacement)

    # Replace any other invalid characters (e.g., non-alphanumeric) with '_OTHER_'
    filename = re.sub(r'[^\w\-\.]', '_OTHER_', filename)

    return filename


def filename_to_doi(filename: str) -> str:
    """
    Convert a filename back to a DOI by replacing the placeholders with original characters.

    Args:
    filename (str): The filename to be converted back to a DOI.

    Returns:
    str: The original DOI.
    """
    # Define replacements for invalid filename characters
    replacements = {
        '_SLASH_': '/',
        '_COLON_': ':'
    }

    # Replace each replacement with its original character
    doi = filename
    for replacement, char in replacements.items():
        doi = doi.replace(replacement, char)

    # Remove any '_OTHER_' placeholders
    doi = re.sub(r'_OTHER_', '', doi)

    return doi


def list_files_in_directory(directory):
    """
    List all files in the given directory.

    Args:
    directory (str): Path to the directory.

    Returns:
    list: A list of filenames in the directory.
    """
    return [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]


def has_multiple_columns(pdf_path):
    """
    Check if a PDF has multiple columns by analyzing its first page.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    bool: True if the PDF likely has multiple columns, False otherwise.
    """
    # Convert the first page of the PDF to an image
    images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=1)
    image = images[0]

    # Convert the image to grayscale and binarize it
    image = image.convert('L')
    image = image.point(lambda x: 0 if x < 128 else 255, '1')

    # Convert the image to a NumPy array for easier processing
    image_array = np.array(image)

    # Get the dimensions of the image
    height, width = image_array.shape

    # Scan the middle of the image vertically
    for x in range(width // 2 - 5, width // 2 + 5):
        white_pixels = 0
        for y in range(height):
            if image_array[y, x] == 255:
                white_pixels += 1
                if white_pixels > 50:
                    return True
            else:
                white_pixels = 0

    # If no tall line of white pixels was found, return False
    return False

def xml_to_string(xml_string):
    """
    Convert an XML string to a formatted string representation. This is what we use to get the XML documents into a
    format the LLM will be able to interpret better, with fewer artifacts.

    Args:
        xml_string (str): The XML content as a string.

    Returns:
        str: A formatted string representation of the XML content.
    """
    formatted_output = ""
    root = ET.fromstring(xml_string)

    def process_element(element, level=0):
        nonlocal formatted_output
        tag = element.tag.split('}')[-1]  # Remove namespace prefix
        text = (element.text or "").strip()

        # Format different types of elements
        if tag == "article-title":
            formatted_output += text + "\n" + "=" * len(text) + "\n\n"
        elif tag in ["p", "sec"]:
            formatted_output += text + "\n\n"
        elif tag == "title":
            formatted_output += "#" * (level + 1) + " " + text + "\n\n"
        elif tag == "abstract":
            formatted_output += "Abstract: " + text + "\n\n"
        elif tag in ["fig", "table-wrap"]:
            formatted_output += f"{tag.capitalize()}: {text}\n\n"
        elif tag == "caption":
            formatted_output += f"Caption: {text}\n\n"
        elif tag == "table":
            formatted_output += f"Table: {text}\n\n"
        elif tag not in [
            "ref", "element-citation", "person-group", "name", "surname",
            "given-names", "article-title", "source", "year", "volume",
            "fpage", "lpage", "pub-id"
        ]:
            # For other tags, just append the text
            formatted_output += text

        # Process each child element recursively
        for child in element:
            process_element(child, level + 1)
            # IMPORTANT: capture tail text (the text after the child's closing tag)
            tail_text = (child.tail or "").strip()
            if tail_text:
                formatted_output += tail_text + " "

    process_element(root)
    return formatted_output



def elements_to_string(elements_list):
    """
    Convert a list of document elements to a formatted string representation. Document elements are what we get when we
    process a document using unstructured (pdfs), this function converts this into formatted plaintext so the LLM can
    interpret it better.

    Args:
    elements_list (list): A list of document elements.

    Returns:
    str: A formatted string representation of the document elements.
    """
    formatted_output = ""

    for element in elements_list:
        element_type = element.get("type")
        text = element.get("text", "")

        # Format different types of elements
        if element_type == "Title":
            formatted_output += text + "\n" + "=" * len(text) + "\n\n"
        elif element_type in ["Text", "NarrativeText", "UncategorizedText"]:
            formatted_output += text + "\n\n"
        elif element_type == "BulletedText":
            formatted_output += "* " + text + "\n"
        elif element_type == "Abstract":
            formatted_output += "Abstract: " + text + "\n\n"
        elif element_type in ["Form", "Field-Name", "Value", "Link", "CompositeElement"]:
            formatted_output += text + "\n"
        elif element_type in ["Image", "Picture"]:
            formatted_output += "[Image: " + text + "]\n"
        elif element_type in ["FigureCaption", "Caption", "Footnote"]:
            formatted_output += "(" + text + ")\n"
        elif element_type == "Figure":
            formatted_output += "Figure: " + text + "\n"
        elif element_type in ["List", "List-item", "ListItem"]:
            formatted_output += "\n- " + text + "\n"
        elif element_type == "Checked":
            formatted_output += "[x] " + text + "\n"
        elif element_type == "Unchecked":
            formatted_output += "[ ] " + text + "\n"
        elif element_type in ["Address", "EmailAddress"]:
            formatted_output += text + "\n"
        elif element_type == "PageBreak":
            formatted_output += "\n--- Page Break ---\n\n"
        elif element_type == "Formula":
            formatted_output += "Formula: " + text + "\n"
        elif element_type == "Table":
            formatted_output += "Table: " + text + "\n"
        elif element_type in ["Header", "Headline", "Subheadline", "Page-header", "Section-header"]:
            formatted_output += text.upper() + "\n"
        elif element_type in ["Footer", "Page-footer"]:
            formatted_output += "-- " + text + " --\n"
        else:
            formatted_output += f"Other: {text}\n\n"

    return formatted_output



def select_schema_file():
    """
    Allows the user to select a schema file from the 'dataModels' directory.

    Returns:
    str: Full path to the selected schema file.
    """
    schema_dir = os.path.join(os.getcwd(), 'dataModels')
    schema_files = [file for file in os.listdir(schema_dir) if file.endswith('.pkl')]

    print("Available schema files:")
    for i, file in enumerate(schema_files):
        print(f"{i + 1}. {file}")

    while True:
        choice = input("Enter the number of the schema file you want to use: ")
        if choice.isdigit() and 1 <= int(choice) <= len(schema_files):
            return os.path.join(schema_dir, schema_files[int(choice) - 1])
        else:
            print("Invalid choice. Please try again.")


def load_schema_file(schema_file):
    """
    Loads and parses a schema file.

    Args:
    schema_file (str): Path to the schema file.

    Returns:
    tuple: (schema_data, key_columns) where schema_data is a dictionary containing the schema information
           and key_columns is a list of column numbers used as keys.
    """
    with open(schema_file, 'r') as f:
        lines = f.readlines()

    key_columns = []
    if lines[0].startswith("Key Columns:"):
        key_columns = [int(column.strip()) for column in lines[0].split(':')[1].strip().split(',') if column.strip()]
        lines = lines[1:]  # Skip the key columns line

    schema_data = {}
    current_column = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split('-')
        if len(parts) == 2:
            column_number = int(parts[0].strip())
            info_type, info_value = parts[1].split(':', 1)
            info_type = info_type.strip()
            info_value = info_value.strip().strip("'")

            if info_type == "Type":
                current_column = column_number
                schema_data[current_column] = {'type': info_value}
            elif current_column is not None:
                if info_type == "Name":
                    schema_data[current_column]['name'] = info_value
                elif info_type == "Description":
                    schema_data[current_column]['description'] = info_value
                elif info_type == "Allowed Values":
                    schema_data[current_column]['allowed_values'] = [value.strip() for value in info_value.split(',') if
                                                                     value.strip()]
                elif info_type == "Min Length":
                    schema_data[current_column]['min_length'] = int(info_value) if info_value else None
                elif info_type == "Max Length":
                    schema_data[current_column]['max_length'] = int(info_value) if info_value else None
                elif info_type == "Min Value":
                    schema_data[current_column]['min_value'] = float(info_value) if info_value else None
                elif info_type == "Max Value":
                    schema_data[current_column]['max_value'] = float(info_value) if info_value else None
                elif info_type == "Required Substrings":
                    schema_data[current_column]['required_substrings'] = [substring.strip() for substring in
                                                                          info_value.split(',') if substring.strip()]
                elif info_type == "Blacklisted Substrings":
                    schema_data[current_column]['blacklisted_substrings'] = [substring.strip() for substring in
                                                                             info_value.split(',') if substring.strip()]

    # Fill in missing 'description' keys with empty strings
    for column_data in schema_data.values():
        if 'description' not in column_data:
            column_data['description'] = ''

    return schema_data, key_columns


def generate_prompt(schema_data, user_instructions, key_columns=None):
    """
    Generates a prompt for the AI model based on the schema data and user instructions. Uses a few other helper
    functions to generate text describing the schema, examples of good responses, and key column info.

    Args:
    schema_data (dict): Dictionary containing the schema information.
    user_instructions (str): Additional instructions provided by the user.
    key_columns (list): List of column numbers used as keys for checking duplicates.

    Returns:
    str: A formatted prompt for the AI model.
    """
    if key_columns is None:
        key_columns = []
    num_columns = len(schema_data)
    schema_info = ""
    schema_diagram = ""

    # Generate schema information and diagram
    for column_number, column_data in schema_data.items():
        schema_info += f"Column {column_number}: {column_data['name']} ({column_data['type']})\n"
        schema_info += f"Description: {column_data['description']}\n\n"
        schema_diagram += f"{column_data['name']}, "
    schema_diagram = schema_diagram[:-2]

    # Generate key column information
    key_column_names = [schema_data[int(column)]['name'] for column in key_columns]
    key_column_info = (f"The column(s) {', '.join(key_column_names)} will be used as a key to check for duplicates "
                       f"within each paper. Ensure that the values in these columns are unique for each row extracted.")

    # Construct the prompt
    prompt = f"""
Please extract information from the provided research paper that fits into the following CSV schema:

{schema_info}
{key_column_info if key_columns else ''}

Instructions:
- Extract relevant information and provide it as comma-separated values.
- This paper has been flagged as containing relevant information, and should have data to be extracted.
- Each line should contain {num_columns} values, corresponding to the {num_columns} columns in the schema.
- If information is missing for a column, use 'null' as a placeholder.
- Do not use anything other than 'null' as a placeholder.
- Enclose all string values in double-quotes.
- Never use natural language outside of a string enclosed in double-quotes.
- For range values, use the format "min-max", do not use this format for field expecting integer or float values, if a range is expected it will be labelled explicitly as a range.
- Do not include headers, explanations, summaries, or any additional formatting.
- Invalid responses will result in retries, thus causing significant time and money loss per paper.
- Ignore any information in references that may be included at the end of the paper.

Below I shall provide a few examples to help you understand the desired output format.

Here is an example with just the names of the columns instead of actual values, and just a single entry:
{schema_diagram}

Here are a few examples with randomly generated values where appropriate (be sure to recognize that these values mean nothing, 
should be ignored, and definitely not included in your output, this is just to show the proper structure):

Example where the paper contains a single piece of information to extract:
{generate_examples(schema_data, 1)}

Example where the paper contains two pieces of information to extract:
{generate_examples(schema_data, 2)}

Example where the paper contains three pieces of information to extract:
{generate_examples(schema_data, 3)}

Hopefully that is enough examples for you to see the desired output format.

User Instructions:
{user_instructions}

Paper Contents:
    """

    return prompt
    

def generate_check_prompt(schema_data, user_instructions):
    """
    Generates a prompt for the AI model to check if the paper is relevant and contains extractable information.
    
    Args:
    schema_data (dict): Dictionary containing the schema information.
    user_instructions (str): Instructions on what information to look for.
    
    Returns:
    str: A formatted prompt for the AI model.
    """
    # Generate schema information
    schema_info = ""
    for column_number, column_data in schema_data.items():
        schema_info += f"- {column_data['name']}: {column_data['description']}\n"
    
    # Construct the prompt
    prompt = f"""
Please read the following paper and determine whether it contains information relevant to the following schema and instructions:

Schema:
{schema_info}

User Instructions:
{user_instructions}

Answer "yes" if the paper contains information that can be extracted according to the schema and instructions.
Answer "no" if the paper does not contain enough relevant information to fill out a single row of the defined schema.

Your answer should be only "yes" or "no".

Understand that if you answer "yes" a somewhat costly call will be made to an api to extract relevant information. 
So, it is important to only answer "yes" if the paper contains enough relevant information to fill out at least one row of the defined schema.
Otherwise, answering "yes" when the paper does not contain the needed information will result in wasted money.

Paper Contents:
    """
    return prompt
    

def parse_llm_response(response, num_columns):
    """
    Parse the response from a language model into structured data, filtering out thought sections.

    This function removes any lines between <think> and </think>, then converts the remaining text
    into a list of rows, where each row represents a set of extracted information.

    Args:
    response (str): The raw text response from the language model.
    num_columns (int): The expected number of columns in each row.

    Returns:
    list: A list of unique rows, where each row is a list of column values.
    """
    filtered_lines = []
    in_think_section = False
    
    # Process response line by line, removing <think> sections
    for line in response.strip().split('\n'):
        if "<think>" in line:
            in_think_section = True
        elif "</think>" in line:
            in_think_section = False
            continue
        
        if not in_think_section:
            filtered_lines.append(line)
    
    parsed_data = []
    
    # Use csv reader to properly handle quoted values and commas within fields
    reader = csv.reader(filtered_lines, quotechar='"', skipinitialspace=True)
    for row in reader:
        # Only include rows that have the correct number of columns
        if len(row) == num_columns:
            parsed_data.append(row)
    
    # Remove duplicate entries to ensure uniqueness
    unique_data = []
    for row in parsed_data:
        if row not in unique_data:
            unique_data.append(row)
    
    return unique_data


def normalize_numeric_value(value):
    """
    Normalize a numeric value, handling scientific notation.

    This function removes non-numeric characters (except those used in scientific notation)
    and converts scientific notation to a standard decimal format with up to 4 decimal places.

    Args:
    value (str): The numeric value to normalize.

    Returns:
    str: The normalized numeric value as a string.
    """
    # Remove any special characters and spaces, keeping only digits, decimal point, and scientific notation characters
    value = re.sub(r'[^\d.eE+-]', '', value)

    # Handle scientific notation
    if 'e' in value.lower():
        try:
            # Convert scientific notation to float
            value = float(value)
            # Convert float to string with a maximum of 4 decimal places
            value = f"{value:.4f}"
        except ValueError:
            # If conversion fails, return the original cleaned string
            pass

    return value


def process_value(value, column_data):
    """
    Process and validate a value based on its column type and constraints.

    This function takes a raw value and processes it according to the column type
    (e.g., int, str, float, complex, range, boolean). It also applies any constraints
    specified in the column_data (e.g., min/max values, allowed values, substrings).

    Args:
    value (str): The raw value to process.
    column_data (dict): A dictionary containing the column's type and constraints.

    Returns:
    The processed and validated value, type-casted according to the column type.

    Raises:
    ValueError: If the value doesn't meet the specified constraints.
    """
    column_type = column_data['type']

    if column_type == 'int':
        # Process integer values
        processed_value = int(''.join(filter(str.isdigit, value)))
        min_value = column_data.get('min_value')
        max_value = column_data.get('max_value')
        allowed_values = column_data.get('allowed_values')

        # Validate against constraints
        if allowed_values and str(processed_value) not in allowed_values:
            raise ValueError(f"Value {processed_value} is not in the list of allowed values: {allowed_values}")
        if min_value is not None and processed_value < min_value:
            raise ValueError(f"Value {processed_value} is less than the minimum allowed value {min_value}")
        if max_value is not None and processed_value > max_value:
            raise ValueError(f"Value {processed_value} is greater than the maximum allowed value {max_value}")

        return processed_value

    elif column_type == 'str':
        # Process string values
        processed_value = value.strip()
        min_length = column_data.get('min_length')
        max_length = column_data.get('max_length')
        whitelist_substrings = column_data.get('whitelist_substrings')
        blacklist_substrings = column_data.get('blacklist_substrings')
        allowed_values = column_data.get('allowed_values')

        # Validate against constraints
        if allowed_values and str(processed_value) not in allowed_values:
            raise ValueError(f"Value {processed_value} is not in the list of allowed values: {allowed_values}")
        if min_length is not None and len(processed_value) < min_length:
            raise ValueError(f"String '{processed_value}' is shorter than the minimum allowed length {min_length}")
        if max_length is not None and len(processed_value) > max_length:
            raise ValueError(f"String '{processed_value}' is longer than the maximum allowed length {max_length}")
        if whitelist_substrings:
            for substring in whitelist_substrings:
                if substring not in processed_value:
                    raise ValueError(
                        f"String '{processed_value}' does not contain the required substring '{substring}'")
        if blacklist_substrings:
            for substring in blacklist_substrings:
                if substring in processed_value:
                    raise ValueError(f"String '{processed_value}' contains the blacklisted substring '{substring}'")

        # Remove leading and trailing spaces
        while processed_value and processed_value[0] == " ":
            processed_value = processed_value[1:]
        while processed_value and processed_value[-1] == " ":
            processed_value = processed_value[:-1]

        return processed_value

    elif column_type == 'float':
        # Process float values
        processed_value = float(''.join(filter(lambda x: x.isdigit() or x == '.', value)))
        min_value = column_data.get('min_value')
        max_value = column_data.get('max_value')
        allowed_values = column_data.get('allowed_values')

        # Validate against constraints
        if allowed_values and str(processed_value) not in allowed_values:
            raise ValueError(f"Value {processed_value} is not in the list of allowed values: {allowed_values}")
        if min_value is not None and processed_value < min_value:
            raise ValueError(f"Value {processed_value} is less than the minimum allowed value {min_value}")
        if max_value is not None and processed_value > max_value:
            raise ValueError(f"Value {processed_value} is greater than the maximum allowed value {max_value}")

        return processed_value

    elif column_type == 'complex':
        # Process complex numbers
        return complex(''.join(filter(lambda x: x.isdigit() or x in ['+', '-', 'j', '.'], value)))

    elif column_type == 'range':
        # Process range values (e.g., "10-20")
        range_parts = value.replace(' ', '').split('-')
        if len(range_parts) == 2:
            minimum = float(re.sub("[^0-9]", "", range_parts[0]))
            maximum = float(re.sub("[^0-9]", "", range_parts[1]))
            return f"{minimum}-{maximum}"
        else:
            return value

    elif column_type == 'boolean':
        # Process boolean values
        lower_value = value.lower().strip()
        if 'true' in lower_value:
            return True
        elif 'false' in lower_value:
            return False
        else:
            return value

    else:
        # For unrecognized types, return the value as-is
        return value


def validate_result(parsed_result, schema_data, examples, key_columns=None):
    """
    Validate the parsed result against the schema and remove any invalid or example rows.

    This function processes each row in the parsed result, validating it against the schema
    and removing any rows that don't meet the criteria or match example data.

    Args:
    parsed_result (list): The parsed data from the language model response.
    schema_data (dict): The schema defining the structure and constraints of the data.
    examples (str): A string containing example rows to be excluded from the result.
    key_columns (list): A list of column numbers to be used as keys for checking duplicates.

    Returns:
    list: A list of validated rows that meet all the schema requirements.
    """
    num_columns = len(schema_data)
    validated_result = []
    example_rows = examples.split('\n')

    # Check if headers are present in the parsed result
    headers_present = False
    if parsed_result and len(parsed_result[0]) == num_columns:
        header_row = parsed_result[0]
        for i in range(num_columns):
            if schema_data[i + 1]['name'] == header_row[i]:
                headers_present = True
                break

    if headers_present:
        print("Headers found in response, removing...")
        parsed_result = parsed_result[1:]  # Remove the header row from parsed_result

    for row in parsed_result:
        # Skip rows with incorrect number of columns
        if len(row) != num_columns:
            print(f"Skipping row with invalid number of columns: {row}")
            continue

        # Skip rows that match example data
        if any(example_row == ','.join(row) for example_row in example_rows):
            print(f"Skipping row containing example strings: {row}")
            continue

        validated_row = []
        row_valid = True
        all_null = True

        # Process and validate each value in the row
        for i, value in enumerate(row):
            if value.replace("'","").replace('"','').strip().lower() != 'null' and 'example_string' not in value.replace("'","").replace('"','').strip().lower():
                all_null = False
                column_data = schema_data[i + 1]
                try:
                    processed_value = process_value(value, column_data)
                    validated_row.append(processed_value)
                except Exception as e:
                    print(
                        f"Error processing value '{value}' for column {i + 1} (type: {column_data['type']}): {type(e).__name__} - {str(e)}")
                    row_valid = False
                    break
            else:
                validated_row.append('null')

        # Check if at least one key column has a non-null value, but only if the row is not all nulls
        if row_valid and key_columns and not all_null:
            key_values = [validated_row[i-1] for i in key_columns]
            if all(value.lower() == 'null' for value in key_values):
                print(f"Skipping row with all null key columns: {row}")
                row_valid = False

        if row_valid:
            validated_result.append(validated_row)

    if not validated_result:
        print("No valid rows found in the result.")

    return validated_result


def is_float(value):
    """
    Check if a given value can be converted to a float.

    Args:
    value: The value to check.

    Returns:
    bool: True if the value can be converted to a float, False otherwise.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def write_to_csv(data, headers, filename="extracted_data.csv"):
    """
    Write data to a CSV file, appending if the file exists,
    and adding headers then appending if it does not.

    Args:
    data (list): A list of rows to write to the CSV.
    headers (list): The column headers for the CSV.
    filename (str): The name of the CSV file to write to.
    """
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL, quotechar='"')
        if not file_exists:
            writer.writerow(headers)
        writer.writerows(data)


def generate_examples(schema_data, num_examples=3):
    """
    Generate example data based on the provided schema.

    This function creates random example data that conforms to the schema,
    which can be used to illustrate the expected format of the data.

    Args:
    schema_data (dict): The schema defining the structure and constraints of the data.
    num_examples (int): The number of example rows to generate.

    Returns:
    str: A string containing the generated examples, with each row separated by a newline.
    """
    examples = []
    for _ in range(num_examples):
        example_row = []
        for column_number, column_data in schema_data.items():
            column_type = column_data['type']
            allowed_values = column_data.get('allowed_values', [])

            if allowed_values:
                example_value = random.choice(allowed_values)
            elif column_type == 'str':
                example_value = f'"example_string_{column_number}"'
            elif column_type == 'int':
                min_value = column_data.get('min_value', column_number)
                max_value = column_data.get('max_value', column_number + 10)
                example_value = random.randint(min_value, max_value)
            elif column_type == 'float':
                min_value = column_data.get('min_value', float(column_number))
                max_value = column_data.get('max_value', float(column_number) + 1.0)
                example_value = round(random.uniform(min_value, max_value), 2)
            elif column_type == 'range':
                min_value = column_data.get('min_value', column_number)
                max_value = column_data.get('max_value', column_number + 10)

                # Generate two random integers for the range
                random.seed(datetime.now().timestamp())
                rand1 = random.randint(min_value, max_value)
                random.seed(datetime.now().timestamp())
                rand2 = random.randint(min_value, max_value)

                # Ensure the first number is smaller
                if rand1 > rand2:
                    rand1, rand2 = rand2, rand1

                example_value = f"{rand1}-{rand2}"
            else:
                example_value = f"(example_{column_type}_{column_number})"

            example_row.append(str(example_value))
        examples.append(','.join(example_row))
    return '\n'.join(examples)


def estimate_tokens(text):
    """
    Estimate the number of tokens in a given text.

    This function uses a simple heuristic to estimate the number of tokens,
    based on the assumption that 1 token is approximately 0.75 words.

    Args:
    text (str): The input text to estimate tokens for.

    Returns:
    int: The estimated number of tokens in the text.
    """
    # Split the text into words
    words = re.findall(r'\w+', text)

    # Count the number of words
    word_count = len(words)

    # Estimate the number of tokens
    # A common rule of thumb is that 1 token ≈ 0.75 words
    estimated_tokens = int(word_count / 0.75)

    return estimated_tokens


def truncate_text(text, max_tokens=32000, buffer=3500):
    """
    Truncate text to fit within a specified token limit.

    This function estimates the number of tokens in the text and truncates it
    if it exceeds the specified maximum, leaving a buffer for additional content.
    This is necessary as often times the references at the end of a peper are so long
    that they push the prompt outside the context. This function will remove from the
    end of the text to make sure this does not happen.

    Args:
    text (str): The input text to truncate.
    max_tokens (int): The maximum number of tokens allowed.
    buffer (int): A buffer of tokens to reserve for additional content.

    Returns:
    str: The truncated text if it exceeded the limit, or the original text if not.
    """
    estimated_total_tokens = estimate_tokens(text)

    if estimated_total_tokens <= max_tokens:
        return text

    # Calculate the proportion of text to keep
    keep_ratio = (max_tokens - buffer) / estimated_total_tokens

    # Split the text into words
    words = text.split()

    # Calculate how many words to keep
    words_to_keep = int(len(words) * keep_ratio)

    # Join the words back together
    truncated_text = ' '.join(words[:words_to_keep])

    return truncated_text


def get_processed_pmids(csv_file):
    """
    Retrieve sets of processed PMIDs and PMIDs with no full text from files.

    This function reads a CSV file to get processed PMIDs and a separate file
    for PMIDs that have no full text available.

    Used for the concurrent scraping with pubmed specifically.

    Args:
    csv_file (str): Path to the CSV file containing processed PMIDs.

    Returns:
    tuple: Two sets - processed PMIDs and PMIDs with no full text.
    """
    processed_pmids = set()
    no_fulltext_pmids = set()

    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row:  # Check if row is not empty
                    processed_pmids.add(row[-1])  # Last column is the PMID

    no_fulltext_file = os.path.join(os.getcwd(), 'search_info', 'no_fulltext.txt')
    if os.path.exists(no_fulltext_file):
        with open(no_fulltext_file, 'r') as f:
            no_fulltext_pmids = set(f.read().splitlines())

    return processed_pmids, no_fulltext_pmids
    

def download_ollama():
    """
    Download the latest release of the Ollama binary for AMD64 architecture.

    This function fetches the latest release information from GitHub,
    downloads the AMD64 binary in a .tgz file, extracts the 'ollama' binary,
    makes it executable, and deletes the .tgz file. It provides a progress bar
    during download and handles interruptions.
    """
    # GitHub API URL for the latest releases of ollama
    url = "https://api.github.com/repos/ollama/ollama/releases/latest"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Find the download URL for the amd64 .tgz file
        download_url = None
        for asset in data['assets']:
            if re.search(r'amd64\.tgz$', asset['name'], re.IGNORECASE):
                download_url = asset['browser_download_url']
                break  # Found the desired asset

        if not download_url:
            print("No amd64 .tgz file found in the latest release.")
            return

        # Download the .tgz file with progress bar
        print(f"Downloading ollama .tgz from {download_url}")
        binary_response = requests.get(download_url, stream=True)
        binary_response.raise_for_status()

        total_size = int(binary_response.headers.get('content-length', 0))
        block_size = 8192  # 8 KB

        with open('ollama.tgz', 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc='ollama'
        ) as progress_bar:
            for chunk in binary_response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        # Extract the 'ollama' binary from the .tgz file
        with tarfile.open('ollama.tgz', 'r:gz') as tar:
            member = tar.getmember('bin/ollama')
            member.name = os.path.basename(member.name)  # Rename to 'ollama'
            tar.extract(member, path='.')

        # Delete the .tgz file
        os.remove('ollama.tgz')

        # Make the binary executable
        os.chmod('ollama', 0o755)

        print("ollama binary downloaded, extracted, and saved successfully.")

    except requests.RequestException as e:
        print(f"Failed to download the binary: {e}")
    except KeyboardInterrupt:
        print("You interrupted before ollama finished downloading, cleaning up files...")
        if os.path.isfile('ollama.tgz'):
            os.remove('ollama.tgz')
            print("ollama.tgz file deleted.")
        if os.path.isfile('ollama'):
            os.remove('ollama')
            print("ollama binary deleted.")
        print("Cleanup completed. Next time, please let the download finish!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_chrome_driver():
    """
    Set up and return a Chrome WebDriver instance.

    This function gets the Chrome version, downloads the appropriate ChromeDriver if necessary,
    and sets up a Chrome WebDriver with specific options for headless browsing.

    Returns:
    webdriver.Chrome: A configured Chrome WebDriver instance.

    Raises:
    WebDriverException: If there's an issue with the WebDriver executable permissions.
    """
    chrome_version = get_chrome_version()
    print(f"Chrome Version: {chrome_version}")
    driver_path = get_or_download_chromedriver(chrome_version)

    service = Service(driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    try:
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except WebDriverException as e:
        if 'executable may have wrong permissions' in str(e):
            print(f"Changing permissions for {driver_path}")
            os.chmod(driver_path, 0o755)  # Add execute permissions
            driver = webdriver.Chrome(service=service, options=options)
            return driver
        else:
            raise


def get_chrome_version():
    """
    Get the installed Chrome version on a Linux system.

    Returns:
    str: The version number of the installed Chrome browser.
    """
    # This is specifically for linux
    try:
        vers = os.popen('google-chrome --version').read().strip().split()[-1]
    except:
        raise
    return vers


def get_or_download_chromedriver(version):
    """
    Get the path to ChromeDriver, downloading it if necessary.

    This function checks if ChromeDriver is already available, and if not,
    downloads the appropriate version based on the installed Chrome version.

    Args:
    version (str): The version of Chrome installed on the system.

    Returns:
    str: The path to the ChromeDriver executable.

    Raises:
    Exception: If unable to fetch version information or download ChromeDriver.
    """
    driver_path = os.path.join(os.getcwd(), 'chromedriver-linux64', 'chromedriver')

    # Check if chromedriver already exists
    if os.path.exists(driver_path):
        print("ChromeDriver already exists. Skipping download.")
        return driver_path

    version_base = '.'.join(version.split('.')[:3])

    # Use the JSON API to get the latest version information
    json_url = "https://googlechromelabs.github.io/chrome-for-testing/latest-versions-per-milestone.json"
    response = requests.get(json_url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch version information. Status code: {response.status_code}")

    version_data = response.json()
    milestone = version_base.split('.')[0]

    if milestone not in version_data['milestones']:
        raise Exception(f"No ChromeDriver version found for Chrome {version_base}")

    driver_version = version_data['milestones'][milestone]['version']
    print(f"Latest compatible ChromeDriver version: {driver_version}")

    download_url = f"https://storage.googleapis.com/chrome-for-testing-public/{driver_version}/linux64/chromedriver-linux64.zip"

    print(f"Downloading ChromeDriver from: {download_url}")
    response = requests.get(download_url)

    if response.status_code != 200:
        raise Exception(f"Failed to download ChromeDriver. Status code: {response.status_code}")

    try:
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    except zipfile.BadZipFile:
        raise Exception("Downloaded file is not a valid zip file. The ChromeDriver download might have failed.")

    zip_file.extract('chromedriver-linux64/chromedriver', path=os.getcwd())
    os.rename(os.path.join(os.getcwd(), 'chromedriver-linux64', 'chromedriver'), driver_path)

    os.chmod(driver_path, 0o755)
    print(f"ChromeDriver downloaded and saved to: {driver_path}")
    return driver_path

def is_file_processed(csv_file, filename):
    if not os.path.exists(csv_file):
        return False
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        processed_files = [row[-1] for row in reader]
    return os.path.splitext(filename)[0] in processed_files

def get_yn_response(prompt, attempts=5):
        response = input(prompt).lower()
        attempt_count = 0
        while response not in ["y","n"]:
            if attempt_count > attempts:
                print("Sorry you're having difficulty.  Setting response to 'n' and continuing onward.")
                return "n"
            print("Please enter either 'y' or 'n'. ")
            attempt_count += 1
            response = input(prompt).lower()
        return response


def begin_ollama_server():
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

    if is_ollama_running:
        print("Ollama is running.")
    else:
        # print("Ollama is not running, starting...")
    
        # Check for ollama binary and download if not present
        if not os.path.isfile('ollama'):
            print("ollama binary not found. Downloading the latest release...")
            download_ollama()
        else:
            # print("ollama binary already exists in the current directory.")
            None
        
        try:
            # Start ollama server
            subprocess.Popen(["./ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except:
            # print("Failed to start ollama using portable install, trying with full install")
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def check_model_file(model_name_version):
    model_name, model_version = model_name_version.split(":")
    model_file = os.path.join("/", str(Path.home()), ".ollama", "models", "manifests", "registry.ollama.ai", "library", model_name, model_version)
    model_file_installed = os.path.join("/usr", "share", "ollama", ".ollama", "models", "manifests", "registry.ollama.ai", "library", model_name, model_version)
    print(f"Checking for model {model_name_version} in both:")
    print(model_file)
    print(model_file_installed)
    if os.path.isfile(model_file):
        print("Portable install version found for model")
        begin_ollama_server()
        return False
    elif os.path.isfile(model_file_installed):
        print("Fully installed ollama model file found, nice!")
        return False
    else:
        begin_ollama_server()
        print(f"Model file not found. Pulling the model...")
        try:
            subprocess.run(["./ollama", "pull", model_name_version], check=True)
        except Exception as e:
            print(f"Failed to pull the model using local install: {e}")
            print("Trying using global install...")
            try: 
                subprocess.run(["ollama", "pull", model_name_version], check=True)
            except Exception as e:
                print(f"Global install failed: {e}")
                return True
        return False
