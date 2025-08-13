# Import necessary libraries and modules
from unstructured.staging.base import convert_to_dict
from unstructured.partition.auto import partition
import logging
import os
import subprocess
from src.utils import xml_to_string, elements_to_string
from src.utils import print  # Custom print function for logging

from xml.etree import ElementTree as ET
import requests
import tarfile
import io
import re
from html import unescape
from PIL import Image

# Configure logging to minimize output from the 'unstructured' library
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.CRITICAL)


def pdf2txt_extract(pdf_path, output_dir):
    """Run pdf2txt.py in HTML mode, returning text and filtering out small images."""
    os.makedirs(output_dir, exist_ok=True)

    html_file = os.path.join(
        output_dir, os.path.splitext(os.path.basename(pdf_path))[0] + ".html"
    )

    try:
        result = subprocess.run(
            [
                "pdf2txt.py",
                pdf_path,
                "--output-dir",
                output_dir,
                "--output_type",
                "html",
                "--outfile",
                html_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print(f"pdf2txt.py failed for {pdf_path}: {result.stderr.strip()}")
    except FileNotFoundError:
        print("pdf2txt.py not found; ensure pdfminer.six is installed.")
        return ""
    except Exception as err:
        print(f"Error running pdf2txt.py on {pdf_path}: {err}")
        return ""

    # Filter images by size
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if not os.path.isfile(fpath) or fname.endswith(".html"):
            continue
        try:
            with Image.open(fpath) as im:
                w, h = im.size
        except Exception as err:
            print(f"Failed to read {fpath}: {err}; deleting")
            os.remove(fpath)
            continue
        if min(w, h) < 150:
            print(f"Deleting {fname}: {w}x{h} < 150px threshold")
            os.remove(fpath)

    text_content = ""
    if os.path.exists(html_file):
        with open(html_file, "r") as hf:
            html_data = hf.read()
        # Replace image tags with file names
        html_data = re.sub(r'<img[^>]*src="([^">]+)"[^>]*>', r' [\1] ', html_data)
        html_data = re.sub(r"<[^>]+>", " ", html_data)
        text_content = unescape(re.sub(r"\s+", " ", html_data)).strip()
        os.remove(html_file)

    return text_content


def extract_images_from_pubmed_xml(xml_string, output_dir):
    """Download images for a PubMed Central article using the OA package.

    Returns the number of images downloaded.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        root = ET.fromstring(xml_string)
    except Exception as err:
        print(f"Unable to parse XML for image extraction: {err}")
        return 0

    pmcid_elem = root.find(".//article-id[@pub-id-type='pmcid']")
    if pmcid_elem is None or not pmcid_elem.text:
        print("PMCID not found in XML; cannot download images")
        return 0

    pmcid = pmcid_elem.text.strip()
    oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
    try:
        response = requests.get(oa_url)
        response.raise_for_status()
        oa_root = ET.fromstring(response.text)
    except Exception as err:
        print(f"Failed to fetch OA package link for {pmcid}: {err}")
        return 0

    link_elem = oa_root.find(".//record/link[@format='tgz']")
    if link_elem is None:
        link_elem = oa_root.find(".//record/link[@format='tar.gz']")
    if link_elem is None or not link_elem.attrib.get("href"):
        print(f"No OA package available for {pmcid}")
        return 0

    tar_url = link_elem.attrib["href"].replace("ftp://", "https://")
    try:
        tar_resp = requests.get(tar_url)
        tar_resp.raise_for_status()
    except Exception as err:
        print(f"Failed to download OA package for {pmcid}: {err}")
        return 0

    img_count = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_resp.content), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                if not re.search(r"\.(jpg|jpeg|png|gif|tif|tiff)$",
                                 member.name, re.IGNORECASE):
                    continue

                extracted = tar.extractfile(member)
                if extracted is None:
                    continue

                # ── read the raw bytes once ───────────────────────────────────────────
                img_bytes = extracted.read()

                # ── probe the image size in memory (no temp file needed) ─────────────
                try:
                    with Image.open(io.BytesIO(img_bytes)) as im:
                        w, h = im.size
                except Exception as err:
                    print(f"Skipped {member.name}: could not read image ({err})")
                    continue

                # ── keep only “real” figures ─────────────────────────────────────────
                if min(w, h) < 150:
                    # optionally: also check area, dpi, or aspect ratio here
                    # e.g. if w * h < 40_000:  continue
                    print(f"Skipped {member.name}: {w}×{h} < 150 px threshold")
                    continue

                # ── write out the image; naming is unchanged ─────────────────────────
                out_path = os.path.join(output_dir, os.path.basename(member.name))
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                img_count += 1
    except Exception as err:
        print(f"Failed to extract images from OA package for {pmcid}: {err}")
        return img_count

    if img_count == 0:
        print(f"No images extracted from OA package for {pmcid}")
    else:
        print(f"Extracted {img_count} images from OA package for {pmcid}")
    return img_count


def doc_to_elements(file, use_hi_res=False, use_multimodal=False):
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

    # Base name used for image directory
    paper_id = os.path.splitext(os.path.basename(file))[0]
    images_dir = os.path.join(os.getcwd(), 'images', paper_id)

    # If the file has already been processed, load its content
    formatted_output = None
    if os.path.exists(processed_file_path):
        with open(processed_file_path, 'r') as f:
            formatted_output = f.read()

    xml_content = None

    if formatted_output is None:
        print(f"Now processing {file}")
        elements = None

        # Process PDF files
        if file.endswith('.pdf'):
            formatted_output = pdf2txt_extract(file, images_dir)

        # Process XML files
        elif file.endswith('.xml'):
            with open(file, 'r') as f:
                xml_content = f.read()
            formatted_output = xml_to_string(xml_content)
            with open(processed_file_path, 'w') as f:
                f.write(formatted_output)
        else:
            f = open(file, 'rb')
            try:
                elements = partition(file)
            except Exception as e:
                print(f"Unstructured failed because of {e}")

        if elements is not None and not formatted_output:
            formatted_output = elements_to_string(convert_to_dict(elements))

        if formatted_output:
            with open(processed_file_path, 'w') as f:
                f.write(formatted_output)

    # If multimodal, ensure images are available
    if use_multimodal:
        if file.endswith('.xml'):
            if xml_content is None:
                with open(file, 'r') as f:
                    xml_content = f.read()
            count = extract_images_from_pubmed_xml(xml_content, images_dir)
            print(f"Image extraction complete for {file}: {count} images saved to {images_dir}")

    return formatted_output
