import os
from src.utils import load_schema_file, generate_examples, generate_prompt, generate_check_prompt, truncate_text, get_out_id
from src.document_reader import doc_to_elements


class ScrapeSettings():
    def __init__(self):
        self.scrape_pubmed      = False
        self.scrape_arxiv       = False
        self.scrape_scienceopen = False
        self.scrape_unpaywall   = False
        self.scrape_custom_db   = False
        self.retmax             = 100
        self.base_url           = ""
        self.email              = ""
    def _parse_from_json(self,json):
        for key, val in json.items():
            if key.lower() == "pubmed":
                self.scrape_pubmed = bool(val.lower() == "y")
            elif key.lower() == "arxiv":
                self.scrape_arxiv = bool(val.lower() == "y")
            elif key.lower() == "scienceopen":
                self.scrape_scienceopen = bool(val.lower() == "y")
            elif key.lower() == "unpaywall":
                self.scrape_unpaywall = bool(val.lower() == "y")
            elif key.lower() == "customdb":
                self.scrape_custom_db = bool(val.lower() == "y")
            elif key.lower() == "retmax":
                self.retmax = int(val)
            elif key.lower() == "base_url":
                self.base_url = str(val)
            elif key.lower() == "email":
                self.email = str(val)
            else:
                print(f"Scrape setting '{key}' not recognized. \n")

class ExtractSettings():
    def __init__(self):
        self.max_retries = 3
        self.ollama_url  = "http://localhost:11434"
        self.user_instructions = "Explain the extraction task here"
        self.schema_data = None
        self.key_columns = None
        self.num_columns = None
        self.headers = None
        self.prompt = None
        self.examples = None
    def _parse_from_json(self,json):
        for key, val in json.items():
            if key.lower() == "max_retries":
                self.max_retries = int(val)
            elif key.lower() == "ollama_url":
                self.ollama_url = str(val)
            elif key.lower() == "user_instructions":
                self.user_instructions = str(val)
            else:
                print(f"Extract setting '{key}' not recognized. \n")

class FileSettings():
    def __init__(self):
        self.json = "automatic.json"
        self.schema = "schema.pkl"
        self.csv = "results_output.csv"
        self.log = "LoA.log"
        self.search_info_file = "search_info.txt"
    def _parse_from_json(self,json):
        for key,val in json.items():
            if key.lower() == "schema_file":
                self.schema = os.path.join(os.getcwd(), 'dataModels', str(val) )
            elif key.lower() == "log":
                self.log = str(val)
            elif key.lower() == "results_csv":
                self.csv = str(val)
            else:
                print(f"Files setting '{key}' not recognized. \n")

class JobSettings(): ## Contains subsettings as well for each of the job types.
    def __init__(self):
        self.scrape = ScrapeSettings()
        self.extract = ExtractSettings()
        self.files = FileSettings()
        self.auto = False
        self.run_scrape = False
        self.run_extract= False
        self.concurrent = False
        self.use_hi_res = False
        self.def_search_terms = []
        self.maybe_search_terms = []
        self.query_chunks = []
        self.model_name_version = "nemotron:latest"
        self.model_name = "nemotron"
        self.model_version = "latest"
        self.check_model_name_version = "nemotron:latest"
        self.check_model_name = "nemotron"
        self.check_model_version = "latest"
        self.use_openai = False
        self.api_key = None
        self.check_prompt = ""


    def _update_model_name_version(self, model_name_version):
        if ":" not in model_name_version:
            model_name_version += ":latest"
        self.model_name, self.model_version = model_name_version.split(":", 1)
        self.model_name_version = f"{self.model_name}:{self.model_version}"

        # Check if model_name indicates OpenAI API usage
        if self.model_name.startswith("o1-") or self.model_name.startswith("gpt-"):
            self.use_openai = True
            self.api_key = self.model_version  # The part after the first colon is the API key
            os.environ["OPENAI_API_KEY"] = self.api_key # To use the current version of openai api, must set as environment variable
        else:
            self.use_openai = False
            self.api_key = None

    def _parse_from_json(self,json):
        for key,val in json.items():
            if key.lower() == "def_search_terms":
                if type(val) == list:
                    self.def_search_terms = val
                elif type(val) == str:
                    self.def_search_terms = val.split(",")
            elif key.lower() == "maybe_search_terms":
                if type(val) == list:
                    self.maybe_search_terms = val
                elif type(val) == str:
                    self.maybe_search_terms = val.split(",")
            elif key.lower() == "model_name_version":
                self._update_model_name_version(val)
            elif key.lower() == "check_model_name_version":
                if ":" not in val:
                    val += ":latest"
                self.check_model_name, self.check_model_version = val.split(":", 1)
                self.check_model_name_version = f"{self.check_model_name}:{self.check_model_version}"
            elif key.lower() == "concurrent":
                self.concurrent = bool(val.lower() == "y")
            elif key.lower() == "use_hi_res":
                self.use_hi_res = bool(val.lower() == "y")
            else:
                print(f"JSON key '{key}' not recognized.")
    
    def _finalize(self):
        # Check for necessary file information, generate if missing.
        if any([self.files.csv == "results_output.csv",self.files.csv == "", self.files.csv is None]):
            self.files.csv = os.path.join(os.getcwd(), 'results', f"{self.model_name}_{self.model_version}_{os.path.splitext(self.files.schema)[0].split('/')[-1]}.csv")
        
        # Process Secondary extraction parameters.
        ## Set up extraction parameters
        self.extract.schema_data, self.extract.key_columns = load_schema_file(self.files.schema)
        self.extract.num_columns = len(self.extract.schema_data)
        self.extract.headers = [self.extract.schema_data[column_number]['name'] for column_number in range(1, self.extract.num_columns + 1)] + ['paper']
        self.extract.prompt = generate_prompt(self.extract.schema_data, self.extract.user_instructions, self.extract.key_columns)
        self.extract.examples = generate_examples(self.extract.schema_data)
        # Generate check prompt to reduce cost
        self.check_prompt = generate_check_prompt(self.extract.schema_data, self.extract.user_instructions)
        
        # Generate output directory ID and query chunks
        output_directory_id, self.query_chunks = get_out_id(self.def_search_terms, self.maybe_search_terms)
        # Define the search info file path
        self.files.search_info_file = os.path.join(os.getcwd(), 'search_info', f"{output_directory_id}.txt")


class PromptData():
    def __init__(self, model_name_version, check_model_name_version, use_openai=False, use_hi_res=False):
        self.model = model_name_version
        self.check_model_name_version = check_model_name_version
        self.use_openai = use_openai  # Track if using OpenAI API
        self.stream = False
        self.options = {
                        "num_ctx": 32768,
                        "num_predict": 2048,
                        "mirostat": 0,
                        "mirostat_tau": 0.5,
                        "mirostat_eta": 1,
                        "tfs_z": 1,
                        "top_p": 1,
                        "top_k": 5,
                        "stop": ["|||"],
                        "temperature": 0.7,
                        }
        self.prompt = ""
        self.paper_content = ""
        self.check_prompt = ""
        self.use_hi_res = use_hi_res
        self.first_print = True

    def _refresh_paper_content(self,file,prompt,check_prompt):
        file_path = os.path.join(os.getcwd(), 'scraped_docs', file)
        processed_file_path = os.path.join(os.getcwd(), 'processed_docs', os.path.splitext(file)[0] + '.txt')
        
        """ Supposed to only go once, doesn't...
        if self.first_print:
            print("Base prompt:")
            print(prompt)
            self.first_print = False
        """

        # Load paper content
        if os.path.exists(processed_file_path):
            with open(processed_file_path, 'r') as f:
                self.paper_content = truncate_text(f.read())
        else:
            try:
                self.paper_content = truncate_text(doc_to_elements(file_path), self.use_hi_res)
            except Exception as err:
                print(f"Unable to process {file} into plaintext due to {err}")
                return True
        self.prompt = f"{prompt}\n\n{self.paper_content}\n\nAgain, please make sure to respond only in the specified format exactly as described, or you will cause errors.\nResponse:"
        self.check_prompt = f"{check_prompt}\n\n{self.paper_content}\n\nAgain, please only answer 'yes' or 'no' (without quotes) to let me know if we should extract information from this paper using the costly api call"
        return False
    
    def _refresh_data(self, retry_count):
        if self.use_openai:
            self.options["temperature"] = min(0.7 + 0.1 * retry_count, 1.0)
        else:
            self.options["temperature"] = 0.35 * retry_count
            self.options["repeat_penalty"] = 1.1 + 0.1 * retry_count

    def __dict__(self):
        return {"model": self.model,
                "stream": self.stream,
                "options":self.options,
                "think": True,
                "prompt":self.prompt}
                
    def __check__(self):
        return {"model": self.check_model_name_version,
                "stream": self.stream,
                "options": {
                        "num_ctx": 32768,
                        "num_predict": 1,
                        "mirostat": 0,
                        "mirostat_tau": 0.5,
                        "mirostat_eta": 1,
                        "tfs_z": 1,
                        "top_p": 1,
                        "top_k": 5,
                        "stop": ["|||"],
                        "temperature": 0,
                        },
                "think": False,
                "prompt": self.check_prompt}

