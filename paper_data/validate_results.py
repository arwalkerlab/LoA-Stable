import csv
import os
import logging
import sys
import time
from rdkit import Chem
from rdkit.Chem import AllChem
import cirpy
import requests
# from STOUT import translate_reverse

# Set up logging
logging.basicConfig(filename='validation_errors.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Create a handler that logs to both the log file and console (stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the root logger so that print statements will be logged
logging.getLogger().addHandler(console_handler)

def wait_for_cactus_service(wait_seconds=300):
    """
    Indefinitely check CACTUS to see if it's responding.
    If not responding, wait 'wait_seconds' and try again.
    """
    cactus_url = "https://cactus.nci.nih.gov/chemical/structure/Water/smiles"
    while True:
        try:
            resp = requests.get(cactus_url, timeout=10)
            # If we get a 2xx or 3xx status, we assume it's up
            if resp.ok:
                logging.info("CACTUS (Cirpy) is up.")
                return
            else:
                logging.warning(f"CACTUS ping returned non-OK status ({resp.status_code}). Waiting {wait_seconds} seconds...")
        except Exception as e:
            logging.warning(f"Error checking CACTUS availability: {e}. Will retry in {wait_seconds} seconds.")
        time.sleep(wait_seconds)

def resolve_with_cirpy(identifier):
    """
    Attempt to resolve the 'identifier' using cirpy.resolve.
    If we see a 504 Gateway Timeout or any network-related error,
    we wait for CACTUS to come back online and then retry indefinitely.
    """
    while True:
        try:
            smiles = cirpy.resolve(identifier, 'smiles')
            return smiles
        except Exception as e:
            # If we specifically see a 504 in the error text or a requests error
            # that suggests CACTUS is down, we wait.
            error_str = str(e)
            if "504" in error_str or "timeout" in error_str.lower() or "failed to establish a new connection" in error_str.lower():
                logging.error(f"CIRpy error (possibly CACTUS down) for '{identifier}': {error_str}")
                logging.info("Waiting for CACTUS to come back up...")
                wait_for_cactus_service(300)  # wait 5 minutes, then try again
            else:
                # Some other error, just raise it
                raise

def validate_identifier(identifier, identifier_type):
    if identifier == 'null' or not identifier:
        return False, None

    try:
        if identifier_type == 'smiles':
            # Add check to ensure SMILES isn't empty or malformed before parsing
            if identifier.startswith('['):
                mol = Chem.MolFromSmiles(identifier)
                if mol is None:
                    logging.error(f"Invalid SMILES: {identifier}")
                return mol is not None, mol
            else:
                logging.warning(f"Skipping invalid SMILES input: {identifier}")
                return False, None

        elif identifier_type == 'inchi':
            mol = Chem.MolFromInchi(identifier)
            if mol is None:
                logging.error(f"Invalid InChI: {identifier}")
            return mol is not None, mol
        
        elif identifier_type == 'iupac':
            # Try Cirpy first (in a loop that waits if CACTUS is down)
            try:
                smiles = resolve_with_cirpy(identifier)
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        return True, mol
            except Exception as e:
                logging.error(f"Cirpy error for IUPAC: {identifier}. Error: {str(e)}")

            # If Cirpy fails or returns None, check PubChem
            try:
                pubchem_smiles = fetch_smiles_from_pubchem(identifier)
                if pubchem_smiles:
                    mol = Chem.MolFromSmiles(pubchem_smiles)
                    if mol:
                        return True, mol
            except Exception as e:
                logging.error(f"PubChem error for IUPAC: {identifier}. Error: {str(e)}")

            # If PubChem fails, try STOUT
            """
            try:
                smiles = translate_reverse(identifier)
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        return True, mol
            except KeyError as e:
                logging.error(f"STOUT KeyError for IUPAC: {identifier}. Error: {str(e)}")
            except Exception as e:
                logging.error(f"STOUT error for IUPAC: {identifier}. Error: {str(e)}")
            """
    
    except Exception as e:
        logging.error(f"Unexpected error for {identifier_type}: {identifier}. Error: {str(e)}")
    
    return False, None

def fetch_smiles_from_pubchem(name):
    """Fetch SMILES string from PubChem using the PUG REST API."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'PropertyTable' in data and 'Properties' in data['PropertyTable'] and len(data['PropertyTable']['Properties']) > 0:
            smiles = data['PropertyTable']['Properties'][0].get('CanonicalSMILES')
            if smiles:
                logging.info(f"Successfully fetched SMILES from PubChem for {name}: {smiles}")
                return smiles
            else:
                logging.warning(f"No SMILES found for {name} from PubChem.")
        else:
            logging.warning(f"PubChem returned no valid SMILES for {name}.")
    except Exception as e:
        logging.error(f"Error fetching SMILES from PubChem for {name}: {str(e)}")
    
    return None

def molecule_has_carbon(mol):
    """Returns True if the RDKit molecule has at least one carbon atom."""
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:  # 6 = Carbon
            return True
    return False

def process_results(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    valid_file = os.path.join(output_dir, 'valid_results.csv')
    invalid_file = os.path.join(output_dir, 'invalid_results.csv')
    no_info_file = os.path.join(output_dir, 'no_info_results.csv')

    with open(input_file, 'r') as infile, \
         open(valid_file, 'w', newline='') as valid_out, \
         open(invalid_file, 'w', newline='') as invalid_out, \
         open(no_info_file, 'w', newline='') as no_info_out:

        reader = csv.reader(infile)
        valid_writer = csv.writer(valid_out)
        invalid_writer = csv.writer(invalid_out)
        no_info_writer = csv.writer(no_info_out)

        # Write headers
        headers = next(reader) + ['iupac_resolved', 'smiles_resolved', 'inchi_resolved']
        headers.insert(0, 'canonical_smiles')
        valid_writer.writerow(headers)
        invalid_writer.writerow(headers)
        no_info_writer.writerow(headers[:10])  # Original headers only

        for row in reader:
            try:
                iupac = row[0]

                if iupac == 'null' or iupac == 'failed':
                    no_info_writer.writerow(row[:10])
                    continue
                    
                # Check if there's any "useful" info in columns 2..6
                i = 2
                useful_info = False
                while i < 7:
                    if row[i] != 'null':
                        useful_info = True
                        break
                    else:
                        i += 1
                    
                if not useful_info:
                    logging.info(f"\nSkipping row due to no useful information: \n{row}\n\n")
                    no_info_writer.writerow(row[:10])
                    continue
                    
                logging.info(f"\n\nWorking on row: \n{row}")

                iupac_valid, iupac_mol = validate_identifier(iupac, 'iupac')
                smiles_valid = False
                smiles_mol = None
                inchi_valid = False
                inchi_mol = None

                if not iupac_valid:
                    smiles_valid, smiles_mol = validate_identifier(iupac, 'smiles')
                    if not smiles_valid:
                        inchi_valid, inchi_mol = validate_identifier(iupac, 'inchi')

                resolution_info = [
                    'Yes' if iupac_valid else 'No',
                    'Yes' if smiles_valid else 'No',
                    'Yes' if inchi_valid else 'No'
                ]

                # If any form was valid, proceed
                if iupac_valid or smiles_valid or inchi_valid:
                    # Use the first valid mol object
                    mol = iupac_mol or smiles_mol or inchi_mol
                    
                    # Generate canonical SMILES
                    canonical_smiles = Chem.MolToSmiles(mol)
                    
                    # Now check for at least one carbon atom
                    if molecule_has_carbon(mol):
                        # Insert canonical SMILES into the row
                        row.insert(0, canonical_smiles)
                        valid_writer.writerow(row + resolution_info)
                        logging.info(f"Valid canonical SMILES found: {canonical_smiles}")
                    else:
                        # No carbon => treat as invalid for your dataset
                        row.insert(0, canonical_smiles)
                        invalid_writer.writerow(row + resolution_info)
                        logging.info(f"Invalid canonical SMILES found: {canonical_smiles}")
                else:
                    # None were valid
                    invalid_writer.writerow(row + resolution_info)
            except Exception as e:
                logging.error(f"Error processing row: {row}. Error: {str(e)}")
                invalid_writer.writerow(row + ['No', 'No', 'No'])

if __name__ == "__main__":
    # 1. First, wait for CACTUS/CIRpy to be up. This may wait indefinitely if the service is down.
    wait_for_cactus_service(wait_seconds=300)  # checks every 5 minutes

    # 2. Then proceed with your normal processing.
    input_file = "spec.csv"
    output_dir = "validated_results_1"
    process_results(input_file, output_dir)

    logging.info("Processing complete. Results are in the 'validated_results_1' directory.")
    logging.info("Check 'validation_errors.log' for detailed error information.")

