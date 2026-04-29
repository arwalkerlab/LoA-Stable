# LoA (Librarian of Alexandria)

## Overview

LoA (Librarian of Alexandria) is a comprehensive tool designed for researchers, data scientists, and information professionals. It automates the process of searching, scraping, and analyzing scientific papers from various preprint servers and databases. The tool combines web scraping, natural language processing, and data extraction to efficiently gather and process large volumes of academic literature.

## Features

- **Multi-Source Scraping**: Supports scraping from multiple sources including PubMed Central, arXiv, ChemRxiv, ScienceOpen, and Unpaywall.
- **Customizable Search**: Allows users to define search terms and filters for targeted paper retrieval.
- **Full-Text Download**: Automatically downloads full-text PDFs and XMLs when available.
- **Intelligent Data Extraction**: Uses advanced NLP models to extract specific information from papers based on user-defined schemas.
- **Flexible Schema Definition**: Enables users to create custom data extraction schemas for various research needs.
- **Concurrent Processing**: Implements concurrent scraping and extraction for improved efficiency.
- **Resume Capability**: Ability to resume interrupted scraping or extraction processes.
- **Automatic Model Management**: Handles the download and management of required NLP models.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/MorganRO8/LoA.git
   cd LoA
   ```

2. Install required dependencies listen in install_commands.txt (will be creating requirements.txt soon, as well as a dockerized version)

3. Install conda environments for DECIMER SMILES and DECIMER segmentation if
   DECIMER features are desired (`DECIMER-GPU.yml` and `DECIMER-SEG.yml`)

4. Simply run "python main.py"

## Usage

LoA can be used in two run styles:
- **Interactive mode** (`python main.py`) for prompt-driven setup.
- **Automatic mode** (`python main.py -auto <job.json>`) for reproducible jobs.

### Interactive Mode

Run the main script without any arguments:

```
python main.py
```

Follow the prompts to:
1. Choose between scraping, defining a CSV structure, or extracting data.
2. Enter search terms and select data sources.
3. Define or select a schema for data extraction.
4. Specify the NLP model to use.

### Automatic Mode

Prepare a JSON configuration file with your desired settings, then run:

```
python main.py -auto ./job_scripts/example_small_molecule.json
```

Be sure to replace example_small_molecule.json with the actual file you want to use.

If the `def_search_terms` list contains only `"local"` and `maybe_search_terms`
is set to `"none"`, LoA will skip the scraping step and instead process every
file found in the `scraped_docs` directory. This mode is useful when you have
already downloaded all the papers you want to analyze. Any directories inside
`scraped_docs` are ignored.

Example json files for common workflows can be found in `job_scripts/`.
Recommended starter templates:
- `example_small_molecule.json` + `example_small_molecule.pkl`
- `example_protein.json` + `example_protein.pkl`
- `example_peptide.json` + `example_peptide.pkl`
- `example_reaction.json` + `example_reaction.pkl`
- `example_general.json` + `example_general.pkl`

The `example_general` pair demonstrates schema-level `Validation Mode` usage
for mixed-type extraction in a single schema.
The decimer_synthesis.json configuration focuses on synthesis papers rich in
chemical figures so DECIMER insertion can be tested easily.
The `example_double_check.json` configuration runs a strict row-level
verification pass over an existing extraction CSV and writes a new CSV with an
additional `double_check` yes/no column.
Each configuration can optionally include a `use_comments` setting (`"y"` or `"n"`) to control
whether a trailing comments column is added automatically. A similar `use_solvent` setting
(`"y"` or `"n"`) toggles a built-in solvent column that expects a SMILES string or common name.
If `use_solvent` is enabled, you can also set `assume_water` (`"y"` or `"n"`). When `assume_water`
is on and the model outputs `null` for the solvent column, the value will automatically be filled
with the SMILES string for water (`O`) without further validation.
Common solvents like water and ethanol are resolved through a built-in lookup
table before any online database queries.
The `use_decimer` option (`"y"` or `"n"`) controls whether images are processed
with DECIMER to extract additional SMILES strings from figures. When enabled,
any predicted SMILES are inserted back into the text at the location of the
corresponding figure instead of being appended separately.
The `use_decimer_segmentation` option (`"y"` or `"n"`) controls whether figure
images are first segmented with DECIMER image segmentation into sub-images.
When enabled, LoA will:
- run DECIMER SMILES prediction at the segmented sub-image level (instead of on
  whole extracted figure images),
- still include the original extracted figures as multimodal inputs, and
- also attach all segmented sub-images as additional multimodal inputs.
The prompt additionally includes each sub-image's source figure filename and
bounding-box metadata (absolute pixel coordinates and normalized relative
coordinates) so reaction flow can be inferred in context.
This uses a separate segmentation environment (`decimer-seg`) plus a SMILES
environment (`decimer-gpu`) so segmentation and SMILES inference can be tuned
independently (`DECIMER-SEG.yml` and `DECIMER-GPU.yml`).
`use_openai` can be set to `"y"` if you want to send prompts to the OpenAI API
instead of using local Ollama models. Provide your API key with the `api_key`
setting. You can list available models for your key using:
```
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```
When `use_openai` is enabled, model files are not downloaded through Ollama.

## Target Modes and Column Injection

The `target_type` setting controls which built-in leading columns are injected
before your schema columns.

- `small_molecule` *(default)*: prepends `molecule_name`, validated as SMILES/name.
- `protein`: prepends `protein_name`, validated as sequence/name.
- `peptide`: prepends `peptide_name`, validated as sequence/name.
- `polymer`: prepends `polymer_bigsmiles`, validated as BigSMILES.
- `reaction`: prepends **two** columns:
  1. `reactants` (validated as SMILES)
  2. `products` (validated as SMILES)
  - You can use an empty schema file if you only want these two built-in columns.
- `general`: **no built-in target columns are injected**.

Optional columns:
- `use_solvent: "y"` inserts a solvent column after built-in target columns
  (for `reaction`, it is inserted after `products`).
- `use_comments: "y"` appends a trailing `comments` column.

## Per-Column Validation in General Mode (and Any Mode)

LoA now supports schema-level column validation via an optional
`Validation Mode` field inside schema files. This lets you validate specific
columns with different validators, regardless of the selected `target_type`.

Supported validation modes:
- `small_molecule`
- `protein`
- `peptide`
- `polymer`
- `reaction` (treated as small-molecule SMILES validation per column)

### Example schema fragment

```
1 - Type: str
1 - Name: catalyst
1 - Description: catalyst identity
1 - Validation Mode: small_molecule
2 - Type: str
2 - Name: enzyme
2 - Description: enzyme used
2 - Validation Mode: protein
```

With `target_type: "general"`, these two columns stay exactly where you define
them and are validated using their respective modes.

## Key Components

1. **Scraping Module** (`scrape.py`): Handles the retrieval of papers from various sources.
2. **Extraction Module** (`extract.py`): Manages the process of extracting information from downloaded papers.
3. **Schema Creator** (`meta_model.py`): Allows users to define custom schemas for data extraction.
4. **Document Reader** (`document_reader.py`): Converts various document formats into processable text.
5. **Utilities** (`utils.py`): Contains helper functions used throughout the project.
6. **Concurrent Mode** ('single_paper.py')

## Customization

- **Adding New Sources**: Extend the `scrape.py` file to include new paper sources.
- **Custom Extraction Logic**: Modify the extraction prompts in `extract.py` to suit specific research needs.
- **Schema Definitions**: Use the interactive schema creator to define new data extraction templates.

## Best Practices

1. Respect the terms of service and rate limits of the sources you're scraping from.
2. Ensure you have the necessary permissions to access and use the papers you're downloading.
3. Regularly update your NLP models to benefit from the latest improvements.
4. For large-scale scraping, consider using a distributed system to avoid overloading single sources.

## SMILES Extraction

LoA can optionally extract SMILES strings from PDFs and images using the
DECIMER project. Because DECIMER relies on a separate set of dependencies,
the recommended approach is to install it in its own conda environment. The
`install_commands.txt` file includes the commands to create an environment
named `DECIMER` and install the required packages. LoA will invoke this
environment via `conda run` whenever SMILES extraction is requested. SMILES
strings predicted from figures are inserted directly into the extracted text
where the corresponding images were located.

## Troubleshooting

- If you encounter issues with model downloads, ensure you have a stable internet connection and sufficient disk space.
- For API-related errors, check your API keys and ensure they have the necessary permissions.
- If extraction results are unsatisfactory, try adjusting the schema or providing more specific user instructions.

## Contributing

Contributions to LoA are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes and push to your fork.
4. Submit a pull request with a clear description of your changes.

## Acknowledgements

This project uses several open-source libraries and APIs. We're grateful to the maintainers of these projects for their work, such as:
Unstructured
Ollama

For more detailed information on each module and function, please refer to the inline documentation in the source code.
