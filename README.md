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

3. Simply run "python main.py"

## Usage

LoA can be used in two modes: Interactive (UI) mode and Automatic mode. 

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
python main.py -auto ./job_scripts/example.json
```

Be sure to replace example.json with the actual file you want to use.

Example json files for various kinds of jobs can be found in job_scripts.

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



