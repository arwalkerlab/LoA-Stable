# Imports
import itertools
import glob
import base64
import os
import re
import hashlib
import csv
import tempfile
import shutil
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from openai import OpenAI
import builtins
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
from rdkit import Chem
from rdkit import RDLogger
import cirpy
import pubchempy as pcp
import json

RDLogger.DisableLog('rdApp.error')


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
        elif tag in ["graphic", "media"]:
            href = element.attrib.get('{http://www.w3.org/1999/xlink}href') or element.attrib.get('href')
            if href:
                formatted_output += f"[{href}]"
            else:
                formatted_output += "[Image]"
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


# Built-in column definitions for different chemical targets
BUILTIN_TARGET_COLUMNS = {
    "small_molecule": {
        "type": "str",
        "name": "molecule_name",
        "description": (
            "SMILES string or common name of the small molecule. "
            "Names will be resolved to SMILES using RDKit, Cirpy, and PubChem."
        ),
    },
    "protein": {
        "type": "str",
        "name": "protein_name",
        "description": (
            "Amino acid sequence or common name of the protein. "
            "Names will be resolved to sequences via PyPept, UniProt, and the PDB."
        ),
    },
    "peptide": {
        "type": "str",
        "name": "peptide_name",
        "description": (
            "Amino acid sequence or common name of the peptide. "
            "Names will be resolved to sequences via PyPept, UniProt, and the PDB."
        ),
    },
    "polymer": {
        "type": "str",
        "name": "polymer_bigsmiles",
        "description": (
            "BigSMILES representation of the polymer. "
            "Infer and report BigSMILES from the paper's structures/text when needed."
        ),
    },
    "reaction": [
        {
            "type": "str",
            "name": "reactants",
            "description": (
                "Reactant SMILES string(s). Use '.' to separate multiple reactants "
                "in the same reaction entry."
            ),
        },
        {
            "type": "str",
            "name": "products",
            "description": (
                "Product SMILES string(s). Use '.' to separate multiple products "
                "in the same reaction entry."
            ),
        },
    ],
    "general": [],
}

SUPPORTED_TARGET_TYPES = tuple(BUILTIN_TARGET_COLUMNS.keys())


def normalize_target_type(target_type: str) -> str:
    """Normalize and validate target type values."""
    normalized = str(target_type).strip().lower()
    if normalized not in SUPPORTED_TARGET_TYPES:
        supported = ", ".join(SUPPORTED_TARGET_TYPES)
        raise ValueError(
            f"Unknown target_type '{target_type}'. Supported values are: {supported}."
        )
    return normalized


def normalize_validation_mode(validation_mode: str) -> str:
    """Normalize schema validation mode values."""
    normalized = normalize_target_type(validation_mode)
    if normalized == "general":
        raise ValueError(
            "Validation Mode 'general' is not supported for column validation."
        )
    if normalized == "reaction":
        return "small_molecule"
    return normalized

# Column added to all schemas to capture additional notes
COMMENTS_COLUMN = {
    "type": "str",
    "name": "comments",
    "description": "Any additional relevant details about the molecule or its measurements."
}

# Column added optionally to capture solvent information
SOLVENT_COLUMN = {
    "type": "str",
    "name": "solvent",
    "description": "SMILES string or common name of the solvent; treated like a small molecule."
}

# Common solvents mapped to canonical SMILES to avoid network lookups
SOLVENT_SMILES_LOOKUP = {
    "water": "O",
    "h2o": "O",
    "methanol": "CO",
    "ethanol": "CCO",
    "propanol": "CCCO",
    "isopropanol": "CC(C)O",
    "acetonitrile": "CC#N",
    "dichloromethane": "ClCCl",
    "chloroform": "ClC(Cl)Cl",
    "dmso": "CS(=O)C",
    "dimethyl sulfoxide": "CS(=O)C",
    "hexane": "CCCCCC",
    "dioxane": "O1CCOCC1",
}


def _target_descriptor(target_type: str) -> str:
    """Return a simplified descriptor for the target type."""
    t = normalize_target_type(target_type)
    if t in ["protein", "peptide"]:
        return t
    if t == "polymer":
        return "polymer"
    if t == "reaction":
        return "reactions"
    if t == "general":
        return "records"
    return "small molecule"


def _first_column_instruction(target_type: str, col_name: str) -> str:
    """Return instructions describing the first column based on target type."""
    descriptor = _target_descriptor(target_type)
    if descriptor in ["protein", "peptide"]:
        return (
            f"The first column '{col_name}' uniquely identifies each {descriptor}. "
            "Provide an amino acid sequence in one-letter code or a common name. "
            "Common names will be resolved to sequences."
        )
    if descriptor == "polymer":
        return (
            f"The first column '{col_name}' uniquely identifies each polymer. "
            "Always provide a BigSMILES string. "
            "Do not return polymer/common names in this column."
        )
    if descriptor == "reactions":
        return (
            f"The first column '{col_name}' contains reaction reactant SMILES. "
            "Always provide SMILES directly whenever available."
        )
    return (
        f"The first column '{col_name}' uniquely identifies each {descriptor}. "
        "Always provide a SMILES string directly if it is available. "
        "Only fall back to an IUPAC or common name when no SMILES is given. "
        "Common names will be resolved to SMILES."
    )


def prepend_target_column(schema_data, target_type):
    """Prepend a built-in target column to the schema."""
    info = BUILTIN_TARGET_COLUMNS[normalize_target_type(target_type)]
    info_list = info if isinstance(info, list) else [info]
    new_schema = {}
    for idx, column in enumerate(info_list, start=1):
        new_schema[idx] = column
    shift = len(info_list)
    for idx in sorted(schema_data.keys()):
        new_schema[idx + shift] = schema_data[idx]
    return new_schema


def append_comments_column(schema_data):
    """Append the built-in comments column to the schema."""
    new_schema = schema_data.copy()
    new_schema[len(schema_data) + 1] = COMMENTS_COLUMN
    return new_schema


def insert_solvent_column(schema_data, target_type="small_molecule"):
    """Insert the solvent column after the target column."""
    t = normalize_target_type(target_type)
    if not schema_data:
        return {1: SOLVENT_COLUMN}
    insertion_index = 1
    if t == "reaction":
        insertion_index = 2
    elif t == "general":
        insertion_index = 0

    new_schema = {}
    for idx in sorted(schema_data.keys()):
        new_idx = idx + 1 if idx > insertion_index else idx
        new_schema[new_idx] = schema_data[idx]
    new_schema[insertion_index + 1] = SOLVENT_COLUMN
    return new_schema


def has_solvent_column(schema_data):
    """Return True if the schema includes the solvent column."""
    return any(col.get("name") == "solvent" for col in schema_data.values())


def has_comments_column(schema_data):
    """Return True if the schema includes the default comments column."""
    if not schema_data:
        return False
    last_col = schema_data.get(len(schema_data), {})
    return last_col.get("name") == "comments"


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

    # Allow empty schema files (e.g., reaction-only extractions that rely only on
    # built-in columns like reactants/products).
    if not lines:
        return {}, []

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

        parts = line.split('-', 1)
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
                elif info_type == "Validation Mode":
                    schema_data[current_column]['validation_mode'] = normalize_validation_mode(info_value)

    # Fill in missing 'description' keys with empty strings
    for column_data in schema_data.values():
        if 'description' not in column_data:
            column_data['description'] = ''

    return schema_data, key_columns


def generate_prompt(schema_data, user_instructions, key_columns=None, target_type="small_molecule"):
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
        validation_info = ""
        if column_data.get("validation_mode"):
            validation_info = f" [validated as {column_data['validation_mode']}]"
        schema_info += f"Column {column_number}: {column_data['name']} ({column_data['type']})\n"
        schema_info += f"Description: {column_data['description']}{validation_info}\n\n"
        schema_diagram += f"{column_data['name']}, "
    schema_diagram = schema_diagram[:-2]

    # Generate key column information
    key_column_names = [schema_data[int(column)]['name'] for column in key_columns]
    first_col_instruction = (
        _first_column_instruction(target_type, key_column_names[0])
        if key_columns and normalize_target_type(target_type) != "general"
        else ""
    )
    if key_columns:
        uniqueness_instruction = (
            "Duplicate target values are allowed when other extracted properties differ."
            if normalize_target_type(target_type) in {"polymer", "reaction", "general"}
            else "Ensure that the values in this column are unique within each paper."
        )
        key_column_info = f"{first_col_instruction} {uniqueness_instruction}"
    else:
        key_column_info = ""

    # Construct the prompt
    normalized_target_type = normalize_target_type(target_type)
    descriptor = _target_descriptor(normalized_target_type)
    if normalized_target_type == "polymer":
        target_output_instruction = (
            "- For polymer targets, output BigSMILES exactly as written in the source text.\n"
            "- Do not convert BigSMILES polymer notation to standard SMILES.\n"
            "- Keep all brackets/braces/parentheses and bond descriptors (e.g., $, <, >) unchanged.\n"
            "- If no valid BigSMILES-like target is present, use 'null' for the target column.\n"
            "- If BigSMILES is not explicitly written, infer it from the paper's text/figures.\n"
            "- Build BigSMILES from constitutional repeat unit(s), not full-chain drawings.\n"
            "- Choose descriptors by connectivity: use [$] for equivalent AA-type links; use [<]/[>] for directional/complementary AB-type links.\n"
            "- Mark repeat-unit continuation atoms with bonding descriptors and keep bond order consistent for matching descriptor patterns.\n"
            "- Use the smallest repeat-unit set that captures architecture (homo/random/alternating/block/branched/graft).\n"
            "- Use stochastic-object form { [left] repeat_unit_1,repeat_unit_2 ; optional_end_groups [right] } with [] terminals when outside connections are unspecified.\n"
            "- Include descriptor IDs (e.g., [$1], [<1], [>1]) only when multiple orthogonal connection types are required.\n"
            "- Encode only connectivity supported by the paper; do not invent unsupported stereochemistry/sequence statistics/end-group populations.\n"
            "- Keep output format identical: comma-separated CSV rows only."
        )
    elif normalized_target_type == "reaction":
        target_output_instruction = (
            "- For reactions, both reactants and products columns must contain SMILES.\n"
            "- Use '.' to separate multiple species within reactants or products."
        )
    elif normalized_target_type == "general":
        target_output_instruction = (
            "- Follow each schema column's description carefully.\n"
            "- If a column has a validation mode, format values to satisfy that validator."
        )
    else:
        target_output_instruction = (
            "- Provide SMILES strings directly whenever possible. This has the highest priority over IUPAC or common names."
        )
    prompt = f"""
Using the research paper text provided above, extract information about {descriptor}s that fits into the following CSV schema:

{schema_info}
{key_column_info if key_columns else ''}

Extraction Instructions:
- Extract relevant information and provide it as comma-separated values.
- This paper has been flagged as containing relevant information and should have data to be extracted.
- Each line must contain {num_columns} values, corresponding to the {num_columns} columns in the schema.
- If information is missing for a column, use 'null' as a placeholder.
- Do not use anything other than 'null' as a placeholder.
- Enclose all string values in double-quotes.
- Never use natural language outside of a string enclosed in double-quotes.
- For range values, use the format "min-max" when a range is explicitly expected.
{target_output_instruction}
- Do not include headers, explanations, summaries, or any additional formatting.
- Invalid responses will result in retries, causing significant time and money loss per paper.
- Ignore any information in references that may be included at the end of the paper.

Below are a few examples that demonstrate the correct output format.

Example showing only the column names:
{schema_diagram}

Example where the paper contains a single piece of information:
{generate_examples(schema_data, 1, normalized_target_type)}

Example where the paper contains two pieces of information:
{generate_examples(schema_data, 2, normalized_target_type)}

Example where the paper contains three pieces of information:
{generate_examples(schema_data, 3, normalized_target_type)}

User Instructions:
{user_instructions}
"""

    return prompt
    

def generate_check_prompt(schema_data, user_instructions, target_type="small_molecule"):
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
    normalized_target_type = normalize_target_type(target_type)
    descriptor = _target_descriptor(normalized_target_type)
    polymer_check_instruction = ""
    if normalized_target_type == "polymer":
        polymer_check_instruction = (
            "\nPolymer-specific checks:\n"
            "- Consider BigSMILES extractable when polymer repeat-unit/connectivity information in text or figures is sufficient to construct it.\n"
            "- Output BigSMILES exactly as written in source text when explicit notation is present.\n"
            "- Otherwise infer BigSMILES from repeat units and connectivity shown in the paper.\n"
            "- Do not convert polymer notation to standard SMILES.\n"
            "- Keep brackets/braces/parentheses and bond descriptors unchanged.\n"
            "- If no valid BigSMILES can be found or constructed from the available evidence, answer \"no\"."
        )
    prompt = f"""
Using the research paper text provided above, determine whether it contains information about {descriptor}s relevant to the following schema and instructions.

Schema:
{schema_info}

User Instructions:
{user_instructions}
{polymer_check_instruction}

Answer "yes" if the paper contains enough information to fill out at least one row of the defined schema.
Answer "no" if the required information is missing.

Your answer must be exactly "yes" or "no" with no additional text.

Understand that answering "yes" will result in a costly extraction step, so please be certain.
"""
    return prompt


def generate_double_check_prompt():
    """
    Generate a strict binary prompt for row-level verification.

    Returns:
    str: Prompt template that expects a single yes/no answer.
    """
    return """
You are performing a strict binary classification task.

You will be given:
1) The full text of a scientific paper.
2) The paper images (and segmented sub-images when available).
3) A single CSV row that was previously extracted from this paper.

Task:
- Decide whether the provided row is supported by the paper contents and images.
- Return "yes" only if the row clearly appears in or is directly supported by the paper.
- If there is any uncertainty, inconsistency, or missing support, return "no".
- Be strict: partial matches, guesses, or weak implications should be "no".

Output requirements:
- Respond with exactly one token: yes or no
- Any answer other than exact "yes" is treated as "no"
"""
    

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


def _is_fasta_sequence(seq):
    """Check if a string is a valid FASTA-style amino acid sequence."""
    return re.fullmatch(r"[A-Za-z*]+", seq.strip()) is not None


def _try_pypept(seq):
    """Attempt to generate a peptide structure using pypept."""
    try:
        from pypept import Peptide
        Peptide(seq)
        return True
    except Exception:
        return False


def _fetch_uniprot_sequence(name):
    """Fetch a protein sequence from UniProt using the REST API."""
    try:
        search_url = (
            "https://rest.uniprot.org/uniprotkb/search?query="
            f"{requests.utils.quote(name)}&format=json&size=1"
        )
        r = requests.get(search_url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results:
                acc = results[0].get("primaryAccession")
                if acc:
                    fasta = requests.get(
                        f"https://rest.uniprot.org/uniprotkb/{acc}.fasta",
                        timeout=10,
                    )
                    if fasta.status_code == 200:
                        seq = "".join(
                            line.strip()
                            for line in fasta.text.splitlines()
                            if not line.startswith(">")
                        )
                        if seq:
                            return seq
    except Exception:
        pass
    return None


def _fetch_pdb_sequence(name):
    """Fetch a protein sequence from the RCSB PDB Data API."""
    try:
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {"value": name},
            },
            "return_type": "polymer_entity",
            "request_options": {"pager": {"start": 0, "rows": 1}},
        }
        r = requests.post(
            "https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=10
        )
        if r.status_code == 200:
            results = r.json().get("result_set", [])
            if results:
                identifier = results[0].get("identifier")
                if identifier:
                    r2 = requests.get(
                        f"https://data.rcsb.org/rest/v1/core/polymer_entity/{identifier}",
                        timeout=10,
                    )
                    if r2.status_code == 200:
                        seq = (
                            r2.json()
                            .get("entity_poly", {})
                            .get("pdbx_seq_one_letter_code_can")
                        )
                        if seq:
                            return seq.replace("\n", "").strip()
    except Exception:
        pass
    return None


def _smiles_from_string(value):
    """Return a canonical SMILES string from input using RDKit, Cirpy or PubChem.

    This function now falls back to synonym and related compound searches in
    PubChem when a direct name lookup does not yield a valid structure.
    """
    val_lower = value.strip().lower()
    if val_lower in SOLVENT_SMILES_LOOKUP:
        return SOLVENT_SMILES_LOOKUP[val_lower]
    try:
        mol = Chem.MolFromSmiles(value)
        if mol:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass

    try:
        res = cirpy.resolve(value, "smiles")
        if res:
            return res
    except Exception:
        pass

    def _smiles_from_pubchem(name):
        """Return a canonical SMILES for a name via PubChem, if available."""
        try:
            comps = pcp.get_compounds(name, "name")
            for comp in comps:
                smi = getattr(comp, "canonical_smiles", None)
                if smi:
                    return smi, comp
        except Exception:
            pass
        return None, None

    smi, comp = _smiles_from_pubchem(value)
    if smi:
        return smi

    # Try synonyms and related compounds when a direct lookup fails
    syns: set[str] = set()
    try:
        syns.update(pcp.get_synonyms(value, "name") or [])
    except Exception:
        pass

    try:
        for comp in pcp.get_compounds(value, "name"):
            syns.update(getattr(comp, "synonyms", []) or [])
            rec = getattr(comp, "record", {})
            for rel in rec.get("compound", []):
                cid = rel.get("id", {}).get("id", {}).get("cid")
                if cid:
                    syns.add(f"CID {cid}")
    except Exception:
        pass

    seen: set[str] = {value.lower()}
    while syns:
        syn = syns.pop()
        if not syn or syn.lower() in seen:
            continue
        seen.add(syn.lower())
        smi, comp = _smiles_from_pubchem(syn)
        if smi:
            return smi
        if comp:
            syns.update(getattr(comp, "synonyms", []) or [])
            rec = getattr(comp, "record", {})
            for rel in rec.get("compound", []):
                cid2 = rel.get("id", {}).get("id", {}).get("cid")
                if cid2:
                    syns.add(f"CID {cid2}")
        m = re.search(r"\bCID\s+(\d+)\b", syn, re.IGNORECASE)
        if m:
            cid = m.group(1)
            try:
                comp = pcp.Compound.from_cid(cid)
                smi = getattr(comp, "canonical_smiles", None)
                if smi:
                    return smi
                syns.update(getattr(comp, "synonyms", []) or [])
                rec = getattr(comp, "record", {})
                for rel in rec.get("compound", []):
                    cid2 = rel.get("id", {}).get("id", {}).get("cid")
                    if cid2:
                        syns.add(f"CID {cid2}")
            except Exception:
                pass
        else:
            try:
                for comp in pcp.get_compounds(syn, "name"):
                    syns.update(getattr(comp, "synonyms", []) or [])
                    rec = getattr(comp, "record", {})
                    for rel in rec.get("compound", []):
                        cid = rel.get("id", {}).get("id", {}).get("cid")
                        if cid:
                            syns.add(f"CID {cid}")
            except Exception:
                pass

    return None


def _reaction_side_from_string(value):
    """Validate a dot-separated reaction side and canonicalize each component.

    Accepts strings like ``A.B.C`` and validates each molecule token
    independently using the standard small-molecule resolver.
    """
    if value is None:
        return None
    side = str(value).strip()
    if not side:
        return None

    parts = [part.strip() for part in side.split(".")]
    if not parts or any(not part for part in parts):
        return None

    canonical_parts = []
    for part in parts:
        canonical = _smiles_from_string(part)
        if canonical is None:
            return None
        canonical_parts.append(canonical)
    return ".".join(canonical_parts)


def _bigsmiles_from_string(value):
    """Validate and normalize a BigSMILES string using lightweight heuristics."""
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None

    lowered = candidate.lower()
    invalid_literals = {
        "null",
        "none",
        "n/a",
        "na",
        "unknown",
        "example",
        "example_string",
        "example_bigsmiles",
        "polymer",
        "bigsmiles",
        "-",
        "--",
    }
    if lowered in invalid_literals or lowered.startswith("example_"):
        return None

    def _balanced(text, opener, closer):
        depth = 0
        for ch in text:
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

    if not _balanced(candidate, "{", "}"):
        return None
    if not _balanced(candidate, "[", "]"):
        return None
    if not _balanced(candidate, "(", ")"):
        return None

    # BigSMILES-like structural heuristics:
    # - stochastic object braces are expected
    # - and either explicit descriptors ($, <, >) or repeat/connectivity hints
    has_braces = "{" in candidate and "}" in candidate
    has_descriptor = any(token in candidate for token in ("$", "<", ">"))
    has_repeat_pattern = bool(re.search(r"\[[^\]]+\]", candidate))
    if not has_braces:
        return None
    if not (has_descriptor or has_repeat_pattern):
        return None

    return candidate


def _protein_sequence_from_string(value):
    """Resolve a protein name or sequence to a valid amino acid sequence."""
    seq = value.strip()
    if _is_fasta_sequence(seq) and _try_pypept(seq):
        return seq
    seq2 = _fetch_uniprot_sequence(value)
    if seq2:
        return seq2
    seq3 = _fetch_pdb_sequence(value)
    return seq3


def _fetch_pubchem_sequence(name):
    """Retrieve a peptide sequence from PubChem given a compound name."""
    try:
        compounds = pcp.get_compounds(name, "name")
        if compounds:
            comp = compounds[0]
            seq = getattr(comp, "peptide_sequence", None)
            if seq:
                return seq
            smiles = getattr(comp, "canonical_smiles", None)
            if smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        seq = Chem.MolToFASTA(mol)
                        if seq:
                            return seq.strip()
                except Exception:
                    pass
    except Exception:
        pass
    return None


def _peptide_sequence_from_string(value):
    """Resolve a peptide name or sequence to a valid amino acid sequence."""
    seq = value.strip()
    if _is_fasta_sequence(seq) and _try_pypept(seq):
        return seq
    return _fetch_pubchem_sequence(value)


def validate_target_value(value, target_type):
    """Validate and normalize the first column based on the target type."""
    t = normalize_target_type(target_type)
    if t == "protein":
        return _protein_sequence_from_string(value)
    if t == "peptide":
        return _peptide_sequence_from_string(value)
    if t == "polymer":
        return _bigsmiles_from_string(value)
    if t == "reaction":
        return _reaction_side_from_string(value)
    return _smiles_from_string(value)


def _get_column_index_by_name(schema_data, column_name):
    """Return 0-based index for a named column, or None if absent."""
    for idx, col in schema_data.items():
        if col.get("name") == column_name:
            return idx - 1
    return None


def _build_validation_modes(schema_data, target_type, verify_target):
    """Build a map of 0-based column indexes to validation modes."""
    mode_map = {}
    normalized_target_type = normalize_target_type(target_type)
    if verify_target and normalized_target_type not in {"general"}:
        if normalized_target_type == "reaction":
            mode_map[0] = "small_molecule"
            mode_map[1] = "small_molecule"
        else:
            mode_map[0] = normalized_target_type

    solvent_idx = _get_column_index_by_name(schema_data, "solvent")
    if solvent_idx is not None:
        mode_map[solvent_idx] = "small_molecule"

    for idx, col_data in schema_data.items():
        validation_mode = col_data.get("validation_mode")
        if validation_mode:
            mode_map[idx - 1] = validation_mode
    return mode_map


def validate_result(parsed_result, schema_data, examples, key_columns=None, target_type="small_molecule", verify_target=True, assume_water=False):
    """
    Validate the parsed result against the schema and remove any invalid or example rows.

    This function processes each row in the parsed result, validating it against the schema
    and removing any rows that don't meet the criteria or match example data.

    Args:
    parsed_result (list): The parsed data from the language model response.
    schema_data (dict): The schema defining the structure and constraints of the data.
    examples (str): A string containing example rows to be excluded from the result.
    key_columns (list): A list of column numbers to be used as keys for checking duplicates.

    verify_target (bool): Whether to verify the first column according to the
    target type.
    assume_water (bool): If True, replace 'null' in the solvent column with the
        SMILES for water and skip validation for that column.

    Returns:
    list: A list of validated rows that meet all the schema requirements.
    """
    num_columns = len(schema_data)
    validated_result = []
    # Parse examples using csv reader so comparisons match parsed_result rows even
    # when quote characters differ from the raw prompt text.
    example_rows = []
    for ex_row in csv.reader(examples.splitlines(), quotechar='"', skipinitialspace=True):
        if len(ex_row) == num_columns:
            example_rows.append(tuple(cell.strip() for cell in ex_row))
    example_rows = set(example_rows)

    validation_modes = _build_validation_modes(schema_data, target_type, verify_target)
    solvent_idx = _get_column_index_by_name(schema_data, "solvent")

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
        normalized_row = tuple(cell.strip() for cell in row)
        if normalized_row in example_rows:
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

        # Check if at least one key column has a non-null value, but only if the row is not all nulls.
        # This guards against empty rows; deduplication (if any) happens in extract.py.
        if row_valid and key_columns and not all_null:
            key_values = [validated_row[i-1] for i in key_columns]
            if all(value.lower() == 'null' for value in key_values):
                print(f"Skipping row with all null key columns: {row}")
                row_valid = False

        if row_valid:
            for idx, mode in sorted(validation_modes.items()):
                if idx >= len(validated_row):
                    continue
                value = validated_row[idx]
                if str(value).lower() == "null":
                    if solvent_idx == idx:
                        if assume_water:
                            validated_row[idx] = SOLVENT_SMILES_LOOKUP.get('water', 'O')
                        continue
                    row_valid = False
                    break
                else:
                    canonical = validate_target_value(value, mode)
                    if canonical is None:
                        print(
                            f"Warning: Unable to resolve '{value}' to a valid {mode}."
                        )
                        row_valid = False
                        break
                    validated_row[idx] = canonical

        # Require at least one data field (excluding comments) when a target is present
        if row_valid:
            normalized_target_type = normalize_target_type(target_type)
            start_idx = 0
            if normalized_target_type in {"small_molecule", "protein", "peptide", "polymer"}:
                start_idx = 1
            elif normalized_target_type == "reaction":
                start_idx = 2
            if has_solvent_column(schema_data):
                start_idx += 1
            if has_comments_column(schema_data):
                end_idx = -1
            else:
                end_idx = None
            data_fields = validated_row[start_idx:end_idx] if end_idx is not None else validated_row[start_idx:]
            if data_fields and all(str(v).lower() == 'null' for v in data_fields):
                print(f"Skipping row with no data fields: {row}")
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


def generate_examples(schema_data, num_examples=3, target_type="small_molecule"):
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
    normalized_target_type = normalize_target_type(target_type)
    examples = []
    polymer_targets = [
        '"{[$]CC[$]}"',
        '"{[>][<]CCO[>][<]}"',
        '"{[$]CC(c1ccccc1)[$]}"',
    ]
    reaction_examples = ['"CCO.CC(=O)O"', '"CC(=O)O"', '"CCN.CCO"']
    for _ in range(num_examples):
        example_row = []
        for column_number, column_data in schema_data.items():
            column_type = column_data['type']
            allowed_values = column_data.get('allowed_values', [])

            if (
                normalized_target_type == "polymer"
                and column_number == 1
                and column_type == 'str'
            ):
                example_value = polymer_targets[(_ + column_number) % len(polymer_targets)]
            elif (
                normalized_target_type == "reaction"
                and column_number in {1, 2}
                and column_type == "str"
            ):
                example_value = reaction_examples[(_ + column_number) % len(reaction_examples)]
            elif allowed_values:
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


def _supports_feature(page_text, label):
    active = re.search(rf'<div[^>]*class="text-sm font-medium"[^>]*>\s*{label}\s*</div>', page_text, re.IGNORECASE)
    inactive = re.search(rf'<div[^>]*class="text-sm font-medium[^>]*text-gray-400[^>]*>\s*{label}\s*</div>', page_text, re.IGNORECASE)
    if inactive:
        return False
    if active:
        return True
    return None


def get_model_info(model_name_version, ollama_url="http://localhost:11434", use_openai=False, api_key=None):
    """Return context length and capabilities for a model.

    Attempts to run ``ollama show`` via subprocess. If that fails or the needed
    information cannot be parsed, this function falls back to the Ollama HTTP
    API. The result is a dictionary with ``context_length`` and ``capabilities``
    (a set of capability strings). If the context length cannot be determined,
    a default of 32768 is used.
    """

    if use_openai:
        ctx = 64000
        caps = {"text", "responses", "chat", "vision"}
        model_id = model_name_version.split(":", 1)[0]

        try:
            client = OpenAI(api_key=api_key)
            models = client.models.list()
            if not any(m.id == model_id for m in models.data):
                print(f"Model {model_id} not available for this API key")
        except Exception as err:
            print(f"Failed to query OpenAI for model list: {err}")

        return {"context_length": ctx, "capabilities": caps}

    output = ""

    try:
        result = subprocess.run(
            ["ollama", "show", model_name_version], capture_output=True, text=True, check=True
        )
        output = result.stdout
    except Exception as e:
        print(f"Subprocess call failed: {e}. Trying API fallback...")

    ctx = None
    capabilities = set()

    if output:
        match = re.search(r"context length\s+(\d+)", output, re.IGNORECASE)
        if match:
            ctx = int(match.group(1))

        cap_match = re.search(r"Capabilities\s+([\s\S]+?)\n\s*\n", output, re.IGNORECASE)
        if cap_match:
            for line in cap_match.group(1).splitlines():
                line = line.strip().lower()
                if line:
                    capabilities.add(line)

    if ctx is None or not capabilities:
        try:
            response = requests.post(
                f"{ollama_url}/api/show", json={"model": model_name_version, "verbose": True}
            )
            response.raise_for_status()
            data = response.json()
            if ctx is None:
                info = data.get("model_info", {})
                for key in [
                    "llama.context_length",
                    "qwen.context_length",
                    "general.context_length",
                ]:
                    if key in info:
                        ctx = int(info[key])
                        break
            if not capabilities:
                caps = data.get("capabilities", [])
                if isinstance(caps, list):
                    capabilities.update([c.lower() for c in caps])
        except Exception as api_err:
            print(f"Failed to query Ollama API for model info: {api_err}")

    if ctx is None:
        print("Unable to determine context length. Using default of 32768.")
        ctx = 32768

    if ctx < 32000:
        print(
            f"WARNING: Model context length {ctx} is below the recommended 32000 tokens."
        )

    return {"context_length": ctx, "capabilities": capabilities}


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

    except Exception:
        try:
            response.status_code = 404
        except Exception:
            is_ollama_running = False
        
    # If so, cool, if not, start it!
    try:
        if response.status_code == 200:
            is_ollama_running = True
        else:
            is_ollama_running = False
    except Exception:
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
            subprocess.Popen(
                ["./ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            # print("Failed to start ollama using portable install, trying with full install")
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

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
        print("Model file not found. Pulling the model...")
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


def check_openai_model(model_name, api_key=None):
    """Return True if the given model is unavailable for the provided API key."""
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        for m in models.data:
            if m.id == model_name:
                return False
        print(f"OpenAI model {model_name} not found for this API key")
        return True
    except Exception as err:
        print(f"Failed to fetch OpenAI models: {err}")
        return True


def _run_decimer_segmentation(path):
    """Execute DECIMER image segmentation in dedicated decimer-seg env."""
    script = os.path.join(os.path.dirname(__file__), "decimer_segment_runner.py")
    abs_path = path if os.path.isabs(path) else os.path.join(os.getcwd(), 'scraped_docs', path)
    segment_dir = os.path.join(
        os.path.dirname(abs_path),
        os.path.splitext(os.path.basename(abs_path))[0],
    )
    if os.path.isdir(segment_dir):
        shutil.rmtree(segment_dir, ignore_errors=True)
    os.makedirs(segment_dir, exist_ok=True)
    metadata_path = None
    cmd = [
        "conda",
        "run",
        "-n",
        "decimer-seg",
        "python",
        script,
        abs_path,
        "--output-dir",
        segment_dir,
        "--metadata-out",
        "",  # filled below
    ]
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            metadata_path = tf.name
        cmd[-1] = metadata_path
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as err:
        print(f"Failed to invoke DECIMER segmentation: {err}")
        return []
    finally:
        pass

    if result.returncode != 0:
        print(f"DECIMER segmentation error: {result.stderr}")
        return []

    try:
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        print(f"Could not decode DECIMER segmentation output: {result.stdout}")
    finally:
        if metadata_path and os.path.exists(metadata_path):
            try:
                os.remove(metadata_path)
            except Exception:
                pass
    return []


def _run_decimer_smiles_for_segments(segment_paths):
    """Predict SMILES for a list of segment image file paths in decimer-gpu env."""
    if not segment_paths:
        return []

    script = os.path.join(os.path.dirname(__file__), "decimer_smiles_runner.py")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(segment_paths, tf)
            tmp_path = tf.name

        cmd = [
            "conda",
            "run",
            "-n",
            "decimer-gpu",
            "python",
            script,
            "--segment-paths-json",
            tmp_path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as err:
        print(f"Failed to invoke DECIMER SMILES runner: {err}")
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if result.returncode != 0:
        print(f"DECIMER SMILES error: {result.stderr}")
        return []

    try:
        data = json.loads(result.stdout.strip())
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        print(f"Could not decode DECIMER SMILES output: {result.stdout}")
    return []


def _run_decimer_segments(path, predict_smiles=True):
    """Run DECIMER segmentation and optionally augment with SMILES predictions."""
    data = _run_decimer_segmentation(path)
    if not isinstance(data, list) or not data:
        return []
    if predict_smiles:
        segment_paths = [row.get("segment_path") for row in data if row.get("segment_path")]
        smiles_predictions = _run_decimer_smiles_for_segments(segment_paths)
        for idx, smi in enumerate(smiles_predictions):
            if idx < len(data) and smi:
                data[idx]["smiles"] = smi
    return data


def get_segmented_multimodal_images(images_dir):
    """Return DECIMER-segmented image crops and location metadata for prompts."""
    if not os.path.isdir(images_dir):
        return [], []

    segment_images = []
    segment_notes = []
    for img_path in sorted(glob.glob(os.path.join(images_dir, "*"))):
        if not os.path.isfile(img_path):
            continue
        records = _run_decimer_segments(img_path, predict_smiles=False)
        image_name = os.path.basename(img_path)
        for rec in records:
            seg_path = rec.get("segment_path")
            bbox = rec.get("bbox")
            rel = rec.get("bbox_relative")
            seg_idx = rec.get("segment_index")
            if not seg_path or not os.path.exists(seg_path):
                continue
            with open(seg_path, "rb") as seg_f:
                segment_images.append(base64.b64encode(seg_f.read()).decode("utf-8"))
            bbox_str = bbox if bbox else "unavailable"
            rel_str = [round(float(x), 4) for x in rel] if rel else "unavailable"
            segment_notes.append(
                "source_image={name}, segment={seg}, bbox(y0,x0,y1,x1)={bbox}, "
                "relative_bbox={rel}".format(
                    name=image_name,
                    seg=seg_idx,
                    bbox=bbox_str,
                    rel=rel_str,
                )
            )
    return segment_images, segment_notes


def _rms_diff(arr1, arr2):
    """Return the normalized RMS difference between two image arrays."""
    arr1 = arr1.astype("float32")
    # Ensure arr2 is uint8 before converting to an image
    if arr2.dtype != np.uint8:
        scaled = arr2.copy()
        if scaled.max() <= 1.0:
            scaled *= 255.0
        scaled = np.clip(scaled, 0, 255)
        arr2 = scaled.astype("uint8")
    if arr1.shape != arr2.shape:
        arr2 = np.array(
            Image.fromarray(arr2).resize((arr1.shape[1], arr1.shape[0]))
        ).astype("float32")
    else:
        arr2 = arr2.astype("float32")
    diff = np.sqrt(np.mean((arr1 - arr2) ** 2))
    return diff / 255.0


def _load_pdf_pages(pdf_path, dpi=300):
    """Return list of page images as numpy arrays."""
    pages = convert_from_path(pdf_path, dpi)
    return [np.array(p.convert("RGB")) for p in pages]


def extract_smiles_for_paper(file_path, text, match_tolerance=0.1):
    """Insert SMILES strings predicted from figures directly into the text.

    Parameters
    ----------
    file_path : str
        Path to the source PDF or XML file.
    text : str
        Body text extracted from the paper that may contain placeholders like
        ``[image.jpg]`` where figures were located.

    Returns
    -------
    tuple
        Updated text with SMILES strings inserted and a list of tuples
        ``(smiles, snippet)`` describing where each SMILES string was placed.
    """

    if not text:
        return text, []

    paper_id = os.path.splitext(os.path.basename(file_path))[0]

    images_dir = os.path.join(os.getcwd(), 'images', paper_id)
    pattern = re.compile(r"\[([^\[\]]+\.(?:png|jpg|jpeg|gif|tif|tiff))\]")
    smiles_cache = {}
    for match in pattern.finditer(text):
        img_name = os.path.basename(match.group(1))
        if img_name in smiles_cache:
            continue
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            preds = _run_decimer_segments(img_path, predict_smiles=True)
            smiles = []
            for pred in preds:
                smi = pred.get("smiles")
                if smi and smi not in smiles:
                    smiles.append(smi)
            if smiles:
                smiles_cache[img_name] = smiles

    updated_text = text
    offset = 0
    locations = []
    for match in pattern.finditer(text):
        img_name = os.path.basename(match.group(1))
        smiles = smiles_cache.get(img_name)
        if smiles:
            smi = ", ".join(smiles)
            start, end = match.span()
            start += offset
            end += offset
            updated_text = updated_text[:start] + smi + updated_text[end:]
            offset += len(smi) - (end - start)
            context_start = max(0, start - 30)
            context_end = min(len(updated_text), start + len(smi) + 30)
            snippet = updated_text[context_start:context_end]
            locations.append((smi, snippet))

    return updated_text, locations
