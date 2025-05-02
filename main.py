import os
import sys
import json
import datetime
import builtins

# Create necessary directories if they don't exist
os.makedirs(os.path.join(os.getcwd(), 'scraped_docs'), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), 'dataModels'), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), 'logs'), exist_ok=True)

# Set up custom logging
# This creates a log file with the current timestamp in the filename
builtins.a = os.path.join(os.getcwd(), "logs", f"{str(datetime.datetime.now()).replace(' ', '_')}.txt")
from src.utils import print  # Custom print function for logging
from src.utils import splashbanner, get_yn_response, select_data_model_file, get_out_id, check_model_file
from src.classes import JobSettings

def write_json_jobfile(job_settings: JobSettings):
    with open(job_settings.files.json,"w") as f:
        f.write('{\n')
        f.write('  "settings":{\n')
        f.write('    "def_search_terms":"' + ",".join([x for x in job_settings.def_search_terms]) + '",\n')
        f.write('    "maybe_search_terms":"' + ",".join([x for x in job_settings.maybe_search_terms]) + '",\n')
        f.write('    "auto": true,\n')
        f.write('    "model_name_version":"' + job_settings.model_name_version + '",\n')
        f.write('    "concurrent":"' + job_settings.concurrent + '"\n')
        f.write('  },\n')
        f.write('  "files":{\n')
        f.write('    "schema_file":"' + job_settings.files.schema + '",\n')
        f.write('    "results_csv":"' + job_settings.files.csv  + '",\n')
        f.write('    "logfile":"' + job_settings.files.log  + '"\n')
        f.write('  },\n')
        f.write('  "scrape":{\n')
        f.write('    "pubmed":"' + job_settings.scrape.scrape_pubmed  + '",\n')                
        f.write('    "arxiv":"' + job_settings.scrape.scrape_arxiv  + '",\n')                
        f.write('    "scienceopen":"' + job_settings.scrape.scrape_scienceopen  + '",\n')                
        f.write('    "unpaywall":"' + job_settings.scrape.scrape_unpaywall  + '",\n')                
        f.write('    "customdb":"' + job_settings.scrape.scrape_custom_db  + '",\n')
        f.write('    "base_url":"' + job_settings.scrape.base_url  + '",\n')                
        f.write('    "email":"' + job_settings.scrape.email  + '",\n')                
        f.write('    "retmax":' + str(job_settings.scrape.retmax)  + '\n')                
        f.write('  },\n')
        f.write('  "extract":{\n')
        f.write('    "max_retries":' + str(job_settings.extract.max_retries)  + ',\n')
        f.write('    "ollama_url":"' + job_settings.extract.ollama_url  + '",\n')
        f.write('    "user_instructions":"' + job_settings.extract.user_instructions  + '"\n')
        f.write('  }\n')
        f.write('}')

def interactive_main(job_settings: JobSettings):
    ### Interactive Mode:  Cannot run automatically.  Used to generate specific files necessary for batch mode.
    ### STRONGLY recommend that options to scrape/extract be removed to automatic mode only.  This will ensure that all necessary files are present.
    # Initialize task variable
    task = "0"
    while task not in ["1", "2", "3", "4"]:
        # Prompt the user to select a task
        print("Please select a task:")
        print("1. Define a CSV Structure (Schema File)")
        print("2. Set up automatic job (batch mode).")
        print("3. Exit interactive mode and submit automatic mode.")
        ### Removed scrape, extract, and concurrent from interactive mode, added "exit and submit automatic mode" option instead.  This will allow us to eliminate a lot of the interactive stuff deeper in the program.
        # print("3. Scrape Papers")
        # print("4. Extract Data from Papers into Defined CSV Structure")
        # print("5. Scrape and Extract Concurrently")
        print("4. Exit")

        # Get user input
        task = input("Enter the task number: ")
        task = str(task) ## force string representation of integer inputs.

        # Execute the selected task based on user input
        if task == "1":
            from src.meta_model import UI_schema_creator
            UI_schema_creator()

        elif task == "2":
            #### set up batch job information
            # 1. Set up main job settings
            # 1a. def_search_terms
            job_settings.def_search_terms = input("Enter 'definitely contains' search terms (comma separated) or type 'None' to only use maybe search terms: ").lower().split(',')
            # 1b. maybe_search_terms
            job_settings.maybe_search_terms = input("Enter 'maybe contains' search terms (comma separated) or type 'None' to only use definite search terms: ").lower().split(',')
            # 1c. model_name_version
            job_settings.model_name_version = select_data_model_file()
            # 1d. concurrent
            job_settings.concurrent = input("Do you wish to scrape and extract concurrently? (y/n)").lower()

            # 2. Set up filenames
            # 2a. schema file - if file exists, okay, if not, prompt to create.
            job_settings.files.schema = input("Please enter the path to the desired schema file: ")
            if not os.path.exists(job_settings.files.schema):
                create_schema = input("Provided schema file not found.  Would you like to create it now? (y/n)")
                if create_schema.lower() == "n":
                    print("It is strongly recommended that you create a schema file before attempting to run a job.")
                else:
                    from src.meta_model import UI_schema_creator
                    UI_schema_creator()
            # 2b. json file
            job_settings.files.json = input("Please enter the path to your desired JSON file: ")
            # 2c. results csv file
            job_settings.files.csv = input("Please enter the path to your desired results CSV file: ")
            # 2d. output log file
            job_settings.files.log = input("Please enter the path to your desired output log file: ")

            # 3. Set up Scrape settings
            ## PubMed
            job_settings.scrape.scrape_pubmed = get_yn_response("Would you like to search PubMed?(y/n): ")
            ## ArXiv
            job_settings.scrape.scrape_arxiv = get_yn_response("Would you like to search through the ArXivs?(y/n): ")
            ## ScienceOpen
            job_settings.scrape.scrape_scienceopen = get_yn_response("Would you like to scrape ScienceOpen?(y/n): ")
            ## Unpaywall
            job_settings.scrape.scrape_unpaywall = get_yn_response("Would you like to scrape Unpaywall?(y/n): ")
            if job_settings.scrape.scrape_unpaywall == "y":
                job_settings.scrape.email = input("Enter email for use with Unpaywall:").lower()
            ## Custom Database
            job_settings.scrape.scrape_custom_db = get_yn_response("Would you like to search and download from a custom database?(y/n): ")
            if job_settings.scrape.scrape_custom_db == "y":
                job_settings.scrape.base_url = input("Enter base url:")
            # Define maximum returned papers per search term
            job_settings.scrape.retmax = int(input("Set the maximum number of papers to fetch per search:"))
            attempt_count = 0
            while job_settings.scrape.retmax < 1:
                print("Please enter a positive integer.")
                job_settings.scrape.retmax = int(input("Set the maximum number of papers to fetch per search:"))
                attempt_count+=1
                if attempt_count == 5:
                    print("Sorry you're having difficulty. Setting max fetch to 10.")
                    job_settings.scrape.retmax = 10 ## Prevent user from being stuck in an eternal loop if they don't know what a positive integer is after five tries.

            # 4. Set up Extract settings
            # 4a. max_retries = 3
            job_settings.extract.max_retries = int(input("Please enter the number of retries to attempt for each paper extraction: "))
            # 4b. ollama_url  = "http://localhost:11434"
            job_settings.extract.ollama_url = input("Please enter the url for your ollama server (ENTER for default): ")
            if job_settings.extract.ollama_url.strip() == "":
                job_settings.extract.ollama_url = "http://localhost:11434"
            # 4c. user_instructions = "Explain the extraction task here"
            job_settings.extract.user_instructions = input("Please briefly tell the model what information it is trying to extract, in the format of a command orinstructions:\n\n")

            ### Write all settings to JSON file
            write_json_jobfile(job_settings)

        # elif task == "3":
        #     from src.scrape import scrape
        #     scrape({})

        # elif task == "4":
        #     print("Loading models, please wait...")
        #     from src.extract import batch_extract
        #     batch_extract({})

        # elif task == "5":
        #     from src.single_paper import scrape_and_extract_concurrent
        #     scrape_and_extract_concurrent({})

        elif task == "3":
            print("Exiting interactive mode and running in automatic mode.")
            import subprocess
            subprocess.Popen("python main.py -auto "+str(job_settings.files.json),shell=True)
            sys.exit()
        elif task == "4":
            print("Exiting the Library.")
            sys.exit()

        else:
            print("Invalid task number. Please enter 1 to 4.")

def print_all_settings(job_settings: JobSettings):
    print("######## Librarian of Alexandria Job Settings ########")
    print("# Search Information:")
    print("#   Required Terms:")
    for term in job_settings.def_search_terms:
        print("#     "+ str(term))
    print("#   Optional Terms:")
    for term in job_settings.maybe_search_terms:
        print("#     "+ str(term))
    print("#   Model Name:    "+str(job_settings.model_name))
    print("#   Model Version: "+str(job_settings.model_version))
    print("# ")
    
    print("# Filename Information:")
    print("#   JSON:    " + str(job_settings.files.json) )
    print("#   Schema:  " + str(job_settings.files.schema) )
    print("#   Results: " + str(job_settings.files.csv) )
    print("#   Logfile: " + str(job_settings.files.log) )
    print("# ")
    
    print("# Scraper Settings:")
    print("#   Scraping from: ")
    db_list = ""
    if job_settings.scrape.scrape_pubmed:
        db_list += "PubMed, "
    if job_settings.scrape.scrape_arxiv:
        db_list += "ArXiv, "
    if job_settings.scrape.scrape_scienceopen:
        db_list += "ScienceOpen, "
    if job_settings.scrape.scrape_unpaywall:
        db_list += "UnPaywall, "
    if job_settings.scrape.scrape_custom_db:
        db_list += f"Custom Database ({job_settings.scrape.base_url}), "
    db_list = db_list[:-2]
    print(f"#     {db_list}")
    print("#   Papers per search:  " + str(job_settings.scrape.retmax))
    print("# ")
    # Extract Settings - max retries, ollama server, user instructions.
    print("# Extraction Settings:")
    print("#   User Instructions: ")
    print("#     " + str(job_settings.extract.user_instructions))
    print("#   Ollama Server: " + str(job_settings.extract.ollama_url))
    print("#   Maximum retries: " + str(job_settings.extract.max_retries))
    print("######################################################")
    
#################################### BEGIN MAIN ######################################
def main():
    # Print ASCII art banner
    splashbanner() # moved to src.utils to make main easier to read

    ###### Should we include a complete initialization step here?  
    ###### get all default values set, parse the input files, 
    ###### check file locations, etc.?
    AUTO_EPICFAIL = False # crashout variable if job is in automatic mode (batch) and an unrecoverable error is identified during initialization.
    # 0. Initialize Runtime Job Settings (class to hold job settings, including scrape and extract parameters)
    job_settings = JobSettings()
    
    if "-auto" not in sys.argv:  # Not in automatic mode, therefore in interactive mode.
        interactive_main(job_settings)
        return ## ensures exit of program after completion of interactive_main (though sys.exit in that function should take care of this)

    # In automatic mode, a.k.a. batch mode.  Runs off json file provided
    # Parse command line arguments.
    job_settings.auto = True
    if len(sys.argv) > sys.argv.index("-auto") + 1:
        # Use the next argument as the path to the JSON file
        job_settings.files.json = sys.argv[sys.argv.index("-auto") + 1]
    else:
        print(f"JSON file not specified.  Defaulting to {job_settings.files.json}.\n")

    # Open the JSON file and parse its contents
    tasks = json.load(open(os.path.join(os.getcwd(), job_settings.files.json), "r"))
    for task_name, task_params in tasks.items():
        if task_name == "settings":
            job_settings._parse_from_json(task_params)
        elif task_name == "files":
            job_settings.files._parse_from_json(task_params)
        elif task_name == "scrape":
            job_settings.scrape._parse_from_json(task_params)
            job_settings.run_scrape = True
        elif task_name == "extract":
            job_settings.extract._parse_from_json(task_params)
            job_settings.run_extract = True
        else:
            print(f"Unrecognized JSON section: {task_name}. Job will not continue.\n")
            AUTO_EPICFAIL = True
    job_settings._finalize()

    # Print log of all identified settings.
    print_all_settings(job_settings)
    
    # Create all necessary directories.
    os.makedirs(os.path.join(os.getcwd(), 'scraped_docs'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'search_info'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'results'), exist_ok=True)

    if not job_settings.use_openai:
        ################## SAFETY CHECKS! ##################
        ## Check if the model is available, download if not.  If unable, crash out.
        if check_model_file(job_settings.model_name_version):
            print(f"Unable to find or obtain primary model file for {job_settings.model_name_version}.  Terminating.")
            AUTO_EPICFAIL = True
    
    ################## SAFETY CHECKS! ##################
    ## Check if the check model is available, download if not.  If unable, crash out.
    if check_model_file(job_settings.check_model_name_version):
        print(f"Unable to find or obtain check model file for {job_settings.model_name_version}.  Terminating.")
        AUTO_EPICFAIL = True

    # Verify filenames exist or can be created.
    if not os.path.exists(job_settings.files.json):
        print(f"JSON file {job_settings.files.json} not found.  Exiting program.\n")
        AUTO_EPICFAIL = True
    if not os.path.exists(job_settings.files.schema):
        print(f"SCHEMA file {job_settings.files.schema} not found.")
        AUTO_EPICFAIL = True
    if job_settings.auto and AUTO_EPICFAIL:
        print("Error(s) encountered.  Please review logfile. Terminating program (unable to switch to interactive mode).\n")
        sys.exit(1)


    ################## ACTUALLY PROCESS THE JOB! ##################
    # Rather than loop over the tasks and do them one at a time, we check to see if certain conditions exist in a given priority list.
    if all([job_settings.concurrent, job_settings.run_scrape, job_settings.run_extract]):
        # First, are both "scrape" and "extract" tasks active, and is the "concurrent" flag also set?  If so, we should run the "do both at once" function.
        from src.single_paper import scrape_and_extract_concurrent
        scrape_and_extract_concurrent(job_settings)

    else:
        if job_settings.run_scrape:
            # Then, if not "concurrent", is "scrape" active?  If so, do that before running "extract"
            from src.scrape import scrape
            scrape(job_settings)

        if job_settings.run_extract:
            # Finally, is "extract" active?  If so, run that after completing scrape.
            # Initialize ollama server if necessary?
            from src.extract import batch_extract
            batch_extract(job_settings)

if __name__ == '__main__':
    main()