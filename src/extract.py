import sys
import os
import csv
import requests
import json
from src.utils import (
    generate_prompt,
    parse_llm_response,
    validate_result,
    write_to_csv,
    list_files_in_directory,
    begin_ollama_server,
    extract_smiles_for_paper,
)
from src.classes import JobSettings,PromptData
from openai import OpenAI


def get_files_to_process(job_settings:JobSettings):
    ## Process "search_info_file" to get list of files to process
    if job_settings.files.search_info_file == 'All':
        files_to_process = list_files_in_directory(os.path.join(os.getcwd(), 'scraped_docs'))
    else:
        if os.path.exists(job_settings.files.search_info_file):
            with open(job_settings.files.search_info_file, 'r') as f:
                files_to_process = f.read().splitlines()
            files_to_process =  [file for file in files_to_process if os.path.isfile(os.path.join(os.getcwd(), 'scraped_docs', file))]
        else:
            files_to_process = []

    ## Check for already processed papers
    if os.path.exists(job_settings.files.csv):
        with open(job_settings.files.csv, 'r') as f:
            reader = csv.reader(f)
            processed_papers = [row[-1] for row in reader]
            processed_papers = processed_papers[1:]  # Exclude the header row
    else:
        processed_papers = []

    # Filter out already processed papers and return list.
    return [file for file in files_to_process if os.path.splitext(os.path.basename(file))[0] not in processed_papers]

def batch_extract(job_settings: JobSettings):
    '''
    Batch extraction function for processing multiple papers.

    This function is designed to work with the scraping system and file structure.
    It processes multiple papers in a batch, which is more efficient as it only
    needs to load the model once. However, it requires pre-downloaded papers and
    a specific file structure.

    Args:
    job_settings (JobSettings): A JobSettings object containing configuration parameters.

    Returns:
    None: Results are written to a CSV file.
    '''

    # Check for Ollama binary and start server when using local models
    if not job_settings.use_openai:
        begin_ollama_server()

    data = PromptData(
        model_name_version=job_settings.model_name_version,
        check_model_name_version=job_settings.check_model_name_version,
        use_openai=job_settings.use_openai,
        api_key=job_settings.api_key,
        use_hi_res=job_settings.use_hi_res,
        use_multimodal=job_settings.use_multimodal,
        use_thinking=job_settings.use_thinking,
    )

    # Determine which files to process
    files_to_process = get_files_to_process(job_settings)

    print(f"Found {len(files_to_process)} files to process, starting!")

    # Process each file
    for file in files_to_process:
        print(f"Now processing {file}")
        retry_count = 0
        success = False
        if job_settings.skip_check:
            if data._refresh_paper_content(
                file,
                job_settings.extract.prompt,
                job_settings.check_prompt,
                check_only=False,
            ):
                continue
            check_result = "yes"
            allow_verification = True
        else:
            if data._refresh_paper_content(
                file,
                job_settings.extract.prompt,
                job_settings.check_prompt,
                check_only=True,
            ):
                continue
            # Use a check prompt to lower cost
            check_response = requests.post(
                f"{job_settings.extract.ollama_url}/api/generate", json=data.__check__()
            )
            check_response.raise_for_status()
            check_result = check_response.json()["response"]
            print(f"Check result was '{check_result}'")
            allow_verification = str(check_result).strip().lower().startswith("yes")

        if allow_verification:
            if not job_settings.skip_check:
                data._refresh_paper_content(
                    file,
                    job_settings.extract.prompt,
                    job_settings.check_prompt,
                    check_only=False,
                )

            if job_settings.use_decimer:
                print("Attempting to pull SMILES from images in paper...")
                new_text, locations = extract_smiles_for_paper(file, data.paper_content)
                data.paper_content = new_text
                base_prompt = generate_prompt(
                    job_settings.extract.schema_data,
                    job_settings.extract.user_instructions,
                    job_settings.extract.key_columns,
                    job_settings.target_type,
                )
                data.prompt = (
                    f"Paper Contents:\n{data.paper_content}\n\n{base_prompt}\n\nAgain, please make sure to respond only in the specified format exactly as described, or you will cause errors.\nResponse:"
                )
                if locations:
                    parts = []
                    for smi, snip in locations:
                        snippet = snip.replace("\n", " ").strip()
                        parts.append(f"{smi} -> '{snippet}'")
                    loc_str = "; ".join(parts)
                    print(f"Inserted SMILES with context: {loc_str}")
            else:
                print("DECIMER extraction disabled")
        else:
            print("No SMILES found")
            data.images = []
            data.si_images = []

        # Attempt extraction with retries
        while retry_count < job_settings.extract.max_retries and not success:
            data._refresh_data(retry_count)
            try:
                if check_result == "yes" or check_result == "Yes":
                    if job_settings.use_openai:
                        client = OpenAI()
                        parts = [{"type": "input_text", "text": data.prompt}]
                        if job_settings.use_multimodal:
                            for img in data.images + data.si_images:
                                parts.append(
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/png;base64,{img}",
                                    }
                                )
                        try:
                            resp = client.responses.create(
                                model=job_settings.model_name,
                                input=[{"role": "user", "content": parts}],
                            )
                            result = resp.output_text
                        except Exception as err:
                            print(f"Responses endpoint failed: {err}. Falling back to chat completions.")
                            content_parts = [{"type": "text", "text": data.prompt}]
                            if job_settings.use_multimodal:
                                for img in data.images + data.si_images:
                                    content_parts.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/png;base64,{img}"},
                                        }
                                    )
                            completion = client.chat.completions.create(
                                model=job_settings.model_name,
                                messages=[{"role": "user", "content": content_parts}],
                            )
                            result = completion.choices[0].message.content
                    else:
                        # Use Ollama API
                        response = requests.post(f"{job_settings.extract.ollama_url}/api/generate", json=data.__dict__())
                        response.raise_for_status()
                        result = response.json()["response"]
                else:
                    result = ", ".join(["null" for _ in range(job_settings.extract.num_columns)])

                print("Unparsed result:")
                print(result)

                # Parse and validate the result
                parsed_result = parse_llm_response(result, job_settings.extract.num_columns)
                if not parsed_result:
                    print("Parsed result empty, trying again")
                    retry_count += 1
                    continue

                print("Parsed result:")
                print(parsed_result)

                # Clean up 'null' values
                for row in parsed_result:
                    for idx, item in enumerate(row):
                        try:
                            if item.lower().replace(" ", "") == 'null' or item == '' or item == '""' or item == "''" or item.strip().lower().replace('"', '').replace("'", "") == 'no information found':
                                row[idx] = 'null'
                        except Exception:
                            pass

                validated_result = validate_result(
                    parsed_result,
                    job_settings.extract.schema_data,
                    job_settings.extract.examples,
                    job_settings.extract.key_columns,
                    job_settings.target_type,
                    verify_target=allow_verification,
                    assume_water=job_settings.assume_water,
                )

                if validated_result:
                    print("Validated result:")
                    print(validated_result)

                    paper_filename = os.path.splitext(os.path.basename(file))[0]

                    # Filter results based on key columns
                    if job_settings.extract.key_columns:
                        key_values = set()
                        filtered_result = []
                        for row in validated_result:
                            key = tuple(row[i - 1] for i in job_settings.extract.key_columns)
                            if key not in key_values:
                                key_values.add(key)
                                filtered_result.append(row)
                        validated_result = filtered_result

                    # Add paper filename to each row
                    for row in validated_result:
                        row.append(paper_filename)

                    # Write results to CSV
                    write_to_csv(validated_result, job_settings.extract.headers, filename=job_settings.files.csv)

                    success = True
                else:
                    print("Result failed to validate, trying again.")
                    retry_count += 1

            except ValueError as e:
                print(f"Validation failed for {file}: {str(e)}")
                retry_count += 1
                print(f"Retrying ({retry_count}/{job_settings.extract.max_retries})...")

            except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
                if isinstance(e, requests.exceptions.RequestException) and hasattr(e, "response") and e.response is not None:
                    print(f"Ollama response:\n{e.response.text}")
                print(f"Error processing {file}: {type(e).__name__} - {str(e)}")
                retry_count += 1
                print(f"Retrying ({retry_count}/{job_settings.extract.max_retries})...")

        if not success:
            print(f"Failed to extract data from {file} after {job_settings.extract.max_retries} retries.")
            failed_result = ["failed" for _ in range(job_settings.extract.num_columns)]
            failed_result.append(os.path.splitext(os.path.basename(file))[0])
            write_to_csv([failed_result], job_settings.extract.headers, filename=job_settings.files.csv)

    # If not in auto mode, restart the script
    if not job_settings.auto:
        python = sys.executable
        os.execl(python, python, *sys.argv)
        

def extract(file_path, job_settings:JobSettings):
    '''
    Extract function for processing a single paper and writing results to a CSV file.

    Args:
    file_path (str): Path to the file to be processed.
    job_settings (JobSettings): Job settings object containing configuration.

    Returns:
    list or None: Validated results if successful, None if extraction fails.
    '''
    data = PromptData(
        model_name_version=job_settings.model_name_version,
        check_model_name_version=job_settings.check_model_name_version,
        use_openai=job_settings.use_openai,
        api_key=job_settings.api_key,
        use_hi_res=job_settings.use_hi_res,
        use_multimodal=job_settings.use_multimodal,
        use_thinking=job_settings.use_thinking,
    )

    # Prepare prompt data
    data._refresh_paper_content(
        file_path,
        generate_prompt(
            job_settings.extract.schema_data,
            job_settings.extract.user_instructions,
            job_settings.extract.key_columns,
            job_settings.target_type,
        ),
        check_prompt=job_settings.check_prompt,
        check_only=not job_settings.skip_check,
    )

    validated_result = single_file_extract(job_settings, data, file_path)
    if not validated_result:
        print(f"Failed to extract data from {file_path} after {job_settings.extract.max_retries} retries.")
        return None
    else:
        return validated_result


def single_file_extract(job_settings: JobSettings, data: PromptData, file_path):
    retry_count = 0

    if job_settings.skip_check:
        check_result = "yes"
        allow_verification = True
    else:
        # Use a check prompt to lower cost
        check_response = requests.post(
            f"{job_settings.extract.ollama_url}/api/generate", json=data.__check__()
        )
        check_response.raise_for_status()
        check_result = check_response.json()["response"]
        print(f"Check result was '{check_result}'")
        allow_verification = str(check_result).strip().lower().startswith("yes")

    if allow_verification:
        if not job_settings.skip_check:
            data._refresh_paper_content(
                file_path,
                generate_prompt(
                    job_settings.extract.schema_data,
                    job_settings.extract.user_instructions,
                    job_settings.extract.key_columns,
                    job_settings.target_type,
                ),
                check_prompt=job_settings.check_prompt,
                check_only=False,
            )
        if job_settings.use_decimer:
            print("Attempting to pull SMILES from images in paper...")
            new_text, locations = extract_smiles_for_paper(file_path, data.paper_content)
            data.paper_content = new_text
            base_prompt = generate_prompt(
                job_settings.extract.schema_data,
                job_settings.extract.user_instructions,
                job_settings.extract.key_columns,
                job_settings.target_type,
            )
            data.prompt = (
                f"Paper Contents:\n{data.paper_content}\n\n{base_prompt}\n\nAgain, please make sure to respond only in the specified format exactly as described, or you will cause errors.\nResponse:"
            )
            if locations:
                parts = []
                for smi, snip in locations:
                    snippet = snip.replace("\n", " ").strip()
                    parts.append(f"{smi} -> '{snippet}'")
                loc_str = "; ".join(parts)
                print(f"Inserted SMILES with context: {loc_str}")
        else:
            print("DECIMER extraction disabled")
    else:
        print("No SMILES found")
        data.images = []
        data.si_images = []
    
    while retry_count < job_settings.extract.max_retries:
        data._refresh_data(retry_count)
        try:
            if check_result == "yes" or check_result == "Yes":
                if job_settings.use_openai:
                    client = OpenAI()
                    parts = [{"type": "input_text", "text": data.prompt}]
                    if job_settings.use_multimodal:
                        for img in data.images + data.si_images:
                            parts.append(
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{img}",
                                }
                            )
                    try:
                        resp = client.responses.create(
                            model=job_settings.model_name,
                            input=[{"role": "user", "content": parts}],
                        )
                        result = resp.output_text
                    except Exception as err:
                        print(f"Responses endpoint failed: {err}. Falling back to chat completions.")
                        content_parts = [{"type": "text", "text": data.prompt}]
                        if job_settings.use_multimodal:
                            for img in data.images + data.si_images:
                                content_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{img}"},
                                    }
                                )
                        completion = client.chat.completions.create(
                            model=job_settings.model_name,
                            messages=[{"role": "user", "content": content_parts}],
                        )
                        result = completion.choices[0].message.content
                else:
                    # Use Ollama API
                    response = requests.post(f"{job_settings.extract.ollama_url}/api/generate", json=data.__dict__())
                    response.raise_for_status()
                    result = response.json()["response"]
            else:
                result = ", ".join(["null" for _ in range(job_settings.extract.num_columns)])

            print(f"Unparsed Result:\n{result}")

            # Parse and validate the result
            parsed_result = parse_llm_response(result, job_settings.extract.num_columns)
            if not parsed_result:
                print("Parsed Result empty, trying again")
                retry_count += 1
                continue
            print(f"Parsed Result:\n{parsed_result}")

            # Clean up 'null' values
            for row in parsed_result:
                for idx, item in enumerate(row):
                    try:
                        if any([
                            item.lower().replace(" ", "") == 'null',
                            item in ['', '""', "''"],
                            item.strip().lower().replace('"', '').replace("'", "") == 'no information found'
                        ]):
                            row[idx] = 'null'
                    except Exception:
                        pass

            validated_result = validate_result(
                parsed_result,
                job_settings.extract.schema_data,
                job_settings.extract.examples,
                job_settings.extract.key_columns,
                job_settings.target_type,
                verify_target=allow_verification,
                assume_water=job_settings.assume_water,
            )
            print(f"Validated Result:\n{validated_result}")
            if not validated_result:
                print("Result failed to validate, trying again.")
                retry_count += 1
                continue

            if validated_result:
                # Filter results based on key columns
                if job_settings.extract.key_columns:
                    key_values = set()
                    filtered_result = []
                    for row in validated_result:
                        key = tuple(row[i - 1] for i in job_settings.extract.key_columns)
                        if key not in key_values:
                            key_values.add(key)
                            filtered_result.append(row)
                    validated_result = filtered_result

                # Add paper filename to each row
                for row in validated_result:
                    row.append(os.path.splitext(os.path.basename(file_path))[0])

                # Write results to CSV
                write_to_csv(validated_result, job_settings.extract.headers, filename=job_settings.files.csv)
                return validated_result

        except Exception as e:
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, "response") and e.response is not None:
                print(f"Ollama response:\n{e.response.text}")
            print(f"Error processing {file_path}: {type(e).__name__} - {str(e)}")
            retry_count += 1
            print(f"Retrying ({retry_count}/{job_settings.extract.max_retries})...")
    return None
