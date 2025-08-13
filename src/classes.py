import os
import base64
import glob
from io import BytesIO
from PIL import Image
from src.utils import (
    load_schema_file,
    generate_examples,
    generate_prompt,
    generate_check_prompt,
    truncate_text,
    get_out_id,
    get_model_info,
    prepend_target_column,
    insert_solvent_column,
    append_comments_column,
)
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
        self.use_multimodal = False
        self.use_thinking = False
        self.use_decimer = False
        self.use_comments = True
        self.use_solvent = False
        self.assume_water = False
        self.skip_check = False
        self.target_type = "small_molecule"
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
        """Set ``model_name_version`` respecting OpenAI naming."""
        if self.use_openai:
            name = model_name_version.split(":", 1)[0]
            self.model_name = name
            self.model_version = ""
            self.model_name_version = name
        else:
            if ":" not in model_name_version:
                model_name_version += ":latest"
            self.model_name, self.model_version = model_name_version.split(":", 1)
            self.model_name_version = f"{self.model_name}:{self.model_version}"

    def _parse_from_json(self, json):
        # Handle OpenAI settings first so model parsing knows the mode
        if "use_openai" in json:
            self.use_openai = str(json["use_openai"]).lower() == "y"
        if "api_key" in json:
            self.api_key = str(json["api_key"])

        for key, val in json.items():
            if key.lower() in {"use_openai", "api_key"}:
                continue
            if key.lower() == "def_search_terms":
                if isinstance(val, list):
                    self.def_search_terms = val
                elif isinstance(val, str):
                    self.def_search_terms = val.split(",")
            elif key.lower() == "maybe_search_terms":
                if isinstance(val, list):
                    self.maybe_search_terms = val
                elif isinstance(val, str):
                    self.maybe_search_terms = val.split(",")
            elif key.lower() == "model_name_version":
                self._update_model_name_version(val)
            elif key.lower() == "check_model_name_version":
                if self.use_openai:
                    self.check_model_name = val.split(":", 1)[0]
                    self.check_model_version = ""
                    self.check_model_name_version = self.check_model_name
                else:
                    if ":" not in val:
                        val += ":latest"
                    self.check_model_name, self.check_model_version = val.split(":", 1)
                    self.check_model_name_version = f"{self.check_model_name}:{self.check_model_version}"
            elif key.lower() == "concurrent":
                self.concurrent = bool(val.lower() == "y")
            elif key.lower() == "use_hi_res":
                self.use_hi_res = bool(val.lower() == "y")
            elif key.lower() == "use_multimodal":
                self.use_multimodal = bool(val.lower() == "y")
            elif key.lower() == "use_thinking":
                self.use_thinking = bool(val.lower() == "y")
            elif key.lower() == "use_decimer":
                self.use_decimer = bool(val.lower() == "y")
            elif key.lower() == "use_comments":
                self.use_comments = bool(val.lower() == "y")
            elif key.lower() == "use_solvent":
                self.use_solvent = bool(val.lower() == "y")
            elif key.lower() == "assume_water":
                self.assume_water = bool(val.lower() == "y")
            elif key.lower() == "use_openai":
                self.use_openai = bool(val.lower() == "y")
            elif key.lower() == "api_key":
                self.api_key = str(val)
            elif key.lower() == "skip_check":
                self.skip_check = bool(val.lower() == "y")
            elif key.lower() == "target_type":
                self.target_type = val
            else:
                print(f"JSON key '{key}' not recognized.")
    
    def _finalize(self):
        # Check for necessary file information, generate if missing.
        if any([self.files.csv == "results_output.csv",self.files.csv == "", self.files.csv is None]):
            self.files.csv = os.path.join(os.getcwd(), 'results', f"{self.model_name}_{self.model_version}_{os.path.splitext(self.files.schema)[0].split('/')[-1]}.csv")
        
        # Process Secondary extraction parameters.
        ## Set up extraction parameters
        self.extract.schema_data, _ = load_schema_file(self.files.schema)
        self.extract.schema_data = prepend_target_column(
            self.extract.schema_data, self.target_type
        )
        if self.use_solvent:
            self.extract.schema_data = insert_solvent_column(self.extract.schema_data)
        if self.use_comments:
            self.extract.schema_data = append_comments_column(self.extract.schema_data)
        self.extract.key_columns = [1]
        if self.use_solvent:
            self.extract.key_columns.append(2)
        self.extract.num_columns = len(self.extract.schema_data)
        self.extract.headers = [self.extract.schema_data[column_number]['name'] for column_number in range(1, self.extract.num_columns + 1)] + ['paper']
        self.extract.prompt = generate_prompt(
            self.extract.schema_data,
            self.extract.user_instructions,
            self.extract.key_columns,
            self.target_type,
        )
        self.extract.examples = generate_examples(self.extract.schema_data)
        # Generate check prompt to reduce cost
        self.check_prompt = generate_check_prompt(
            self.extract.schema_data,
            self.extract.user_instructions,
            self.target_type,
        )
        
        # Generate output directory ID and query chunks
        output_directory_id, self.query_chunks = get_out_id(
            self.def_search_terms, self.maybe_search_terms
        )
        # Define the search info file path
        self.files.search_info_file = os.path.join(
            os.getcwd(), 'search_info', f"{output_directory_id}.txt"
        )

        if self.use_openai:
            # Ensure model names are plain IDs without version suffixes
            self.model_name_version = self.model_name_version.split(":", 1)[0]
            self.model_name = self.model_name_version
            self.check_model_name_version = self.check_model_name_version.split(":", 1)[0]
            self.check_model_name = self.check_model_name_version
        if self.use_openai and self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key

        # Special case: use all locally downloaded papers
        if (
            len(self.def_search_terms) == 1
            and self.def_search_terms[0].lower() == "local"
            and (not self.maybe_search_terms or self.maybe_search_terms[0].lower() == "none")
        ):
            self.run_scrape = False
            self.files.search_info_file = "All"


class PromptData():
    def __init__(self, model_name_version, check_model_name_version, use_openai=False, api_key=None, use_hi_res=False, use_multimodal=False, use_thinking=False):
        self.model = model_name_version
        self.check_model_name_version = check_model_name_version
        self.use_openai = use_openai  # Track if using OpenAI API
        self.stream = False
        info = get_model_info(model_name_version, use_openai=use_openai, api_key=api_key)
        ctx_len = info["context_length"]
        self.supports_thinking = "thinking" in info["capabilities"]
        if use_openai:
            self.supports_vision = bool(use_multimodal)
            self.supports_responses = True
        else:
            self.supports_vision = any(cap in info["capabilities"] for cap in ["vision", "images"])
            self.supports_responses = "responses" in info["capabilities"]
            if use_multimodal and not self.supports_vision:
                print("Model does not support vision; disabling multimodal features.")
                use_multimodal = False
        self.options = {
                        "num_ctx": ctx_len,
                        "num_predict": 2048,
                        "mirostat": 0,
                        "mirostat_tau": 0.5,
                        "mirostat_eta": 1,
                        "tfs_z": 1,
                        "top_p": 0.1,
                        "top_k": 5,
                        "temperature": 0.2,
                        }
        self.prompt = ""
        self.paper_content = ""
        self.check_prompt = ""
        self.use_hi_res = use_hi_res
        self.use_multimodal = use_multimodal
        self.use_thinking = use_thinking
        self.first_print = True
        self.images = []
        self.si_images = []

    def _load_images_from_dir(self, directory):
        imgs = []
        for img_file in sorted(glob.glob(os.path.join(directory, '*'))):
            if os.path.isfile(img_file):
                ext = os.path.splitext(img_file)[1].lower()
                data = None
                if ext not in ['.png', '.jpg', '.jpeg']:
                    try:
                        with Image.open(img_file) as im:
                            im = im.convert('RGB')
                            with BytesIO() as buf:
                                im.save(buf, format='PNG')
                                data = buf.getvalue()
                    except Exception as err:
                        print(f"Failed to convert {img_file}: {err}")
                        continue
                if data is None:
                    with open(img_file, 'rb') as img_f:
                        data = img_f.read()
                imgs.append(base64.b64encode(data).decode('utf-8'))
        if not imgs:
            print(f"No images found in {directory}")
        else:
            print(f"Found {len(imgs)} images in {directory}")
        return imgs

    def _refresh_paper_content(self, file, prompt, check_prompt, check_only=False):
        file_path = os.path.join(os.getcwd(), 'scraped_docs', file)
        
        """ Supposed to only go once, doesn't...
        if self.first_print:
            print("Base prompt:")
            print(prompt)
            self.first_print = False
        """

        # Load text first; skip image extraction when only checking
        multimodal = False if check_only else self.use_multimodal
        try:
            self.paper_content = truncate_text(
                doc_to_elements(file_path, self.use_hi_res, multimodal),
                max_tokens=self.options["num_ctx"],
            )
        except Exception as err:
            print(f"Unable to process {file} into plaintext due to {err}")
            return True

        if not check_only and self.use_multimodal and self.supports_vision:
            paper_id = os.path.splitext(os.path.basename(file))[0]
            main_img_dir = os.path.join(os.getcwd(), 'images', paper_id)
            self.images = self._load_images_from_dir(main_img_dir)
            print(f"Loaded {len(self.images)} images from {main_img_dir}")

            si_files = sorted(glob.glob(os.path.join(os.getcwd(), 'scraped_docs', f"{paper_id}_SI*")))
            self.si_images = []
            for si_file in si_files:
                try:
                    doc_to_elements(si_file, self.use_hi_res, self.use_multimodal)
                except Exception as err:
                    print(f"Unable to process {si_file} for images due to {err}")
                    continue
                si_id = os.path.splitext(os.path.basename(si_file))[0]
                si_dir = os.path.join(os.getcwd(), 'images', si_id)
                self.si_images.extend(self._load_images_from_dir(si_dir))
            if si_files:
                print(f"Loaded {len(self.si_images)} images from {len(si_files)} SI files")
        else:
            self.images = []
            self.si_images = []
        note = ""
        if check_only and self.use_multimodal and self.supports_vision:
            note = (
                "\nNote: you are not being shown images at this stage, but they "
                "will be provided if extraction proceeds. Consider this when "
                "deciding if relevant information is present."
            )
        self.prompt = (
            f"Paper Contents:\n{self.paper_content}\n\n{prompt}\n\nAgain, please make sure to respond only in the specified format exactly as described, or you will cause errors.\nResponse:"
        )
        self.check_prompt = (
            f"Paper Contents:\n{self.paper_content}{note}\n\n{check_prompt}\n\nAgain, please only answer 'yes' or 'no' (without quotes) to let me know if we should extract information from this paper using the costly api call"
        )
        return False
    
    def _refresh_data(self, retry_count):
        if self.use_openai:
            self.options["temperature"] = min(0.7 + 0.1 * retry_count, 1.0)
        else:
            self.options["temperature"] = 0.35 * retry_count
            self.options["repeat_penalty"] = 1.1 + 0.1 * retry_count

    def __dict__(self):
        data = {
            "model": self.model,
            "stream": self.stream,
            "options": self.options,
            "think": self.use_thinking and self.supports_thinking,
            "prompt": self.prompt,
        }
        if self.use_multimodal and self.supports_vision:
            data["images"] = self.images + self.si_images
        return data
                
    def __check__(self):
        ctx_len = self.options.get("num_ctx", 32768)
        data = {
            "model": self.check_model_name_version,
            "stream": self.stream,
            "options": {
                "num_ctx": ctx_len,
                "num_predict": 1,
                "mirostat": 0,
                "mirostat_tau": 0.5,
                "mirostat_eta": 1,
                "tfs_z": 1,
                "top_p": 0.1,
                "top_k": 5,
                "temperature": 0,
            },
            "think": False,
            "prompt": self.check_prompt,
        }
        if self.use_multimodal and self.supports_vision:
            data["images"] = self.images
        return data

