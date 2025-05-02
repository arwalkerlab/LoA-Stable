# Import necessary libraries and modules
from unstructured.staging.base import convert_to_dict
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from pdf2image.exceptions import PDFSyntaxError
import logging
import os
from src.utils import has_multiple_columns, xml_to_string, elements_to_string
from src.utils import print  # Custom print function for logging

# Configure logging to minimize output from the 'unstructured' library
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.CRITICAL)


def doc_to_elements(file, use_hi_res=False):
    """
    Convert a document file to a structured text format.

    Args:
    file (str): Path to the input file.
    use_hi_res (bool): Flag to use high-resolution processing for PDFs.

    Returns:
    str: Formatted text content of the document.
    """
    # Create a directory for processed documents if it doesn't exist
    processed_docs_dir = os.path.join(os.getcwd(), 'processed_docs')
    os.makedirs(processed_docs_dir, exist_ok=True)

    # Generate the path for the processed file
    processed_file = os.path.splitext(os.path.basename(file))[0] + '.txt'
    processed_file_path = os.path.join(processed_docs_dir, processed_file)

    # If the file has already been processed, return its content
    if os.path.exists(processed_file_path):
        with open(processed_file_path, 'r') as f:
            return f.read()

    print(f"Now processing {file}")
    elements = None

    # Process PDF files
    if file.endswith('.pdf'):
        if not use_hi_res:
            try:
                # Use default strategy for PDF partitioning
                elements = partition_pdf(filename=file, strategy='auto')
            except (PDFSyntaxError, TypeError) as er:
                print(f"Failed to process {file} due to '{er}'.")
                return None
        else:
            # Check if the PDF has multiple columns
            if has_multiple_columns(file):
                print("Multiple columns detected, running default strategy")
                try:
                    elements = partition_pdf(filename=file, strategy='auto')
                except (PDFSyntaxError, TypeError) as er:
                    print(f"Failed to process {file} due to '{er}'.")
                    return None
            else:
                print("Multiple columns not detected, running advanced strategy")
                try:
                    # Use high-resolution strategy with table structure inference
                    elements = partition_pdf(filename=file, strategy='hi_res', infer_table_structure=True)
                except (PDFSyntaxError, TypeError) as er:
                    print(f"Failed to process {file} due to '{er}'.")
                    return None

    # Process XML files
    elif file.endswith('.xml'):
        with open(file, 'r') as f:
            xml_content = f.read()
        formatted_output = xml_to_string(xml_content)
        with open(processed_file_path, 'w') as f:
            f.write(formatted_output)
        return formatted_output

    # Process other file types
    else:
        f = open(file, 'rb')
        try:
            elements = partition(file)
        except Exception as e:
            print(f"Unstructured failed because of {e}")

    # Convert elements to a formatted string output
    formatted_output = elements_to_string(convert_to_dict(elements))

    # Save the processed output
    with open(processed_file_path, 'w') as f:
        f.write(formatted_output)

    return formatted_output