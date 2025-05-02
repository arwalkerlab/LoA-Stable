import os
import pandas as pd
import random

def process_csv(csv_file):
    # Read the CSV file, ensuring that 'null' and 'failed' strings remain as is
    df = pd.read_csv(
        csv_file,
        header=None,
        dtype=str,
        engine='python',
        on_bad_lines='skip',
        na_values=[],           # Do not interpret any strings as NaN
        keep_default_na=False   # Prevent default NaN values like 'null', 'NA' from being treated as NaN
    )

    # Extract the original headers (first row of the CSV)
    original_headers = df.iloc[0].tolist()

    # Determine the index of the 'paper' column
    paper_col_index = df.columns[-1]  # Last column index (assuming it's the 'paper' column)

    # Initialize lists to hold different types of rows
    valid_rows = []
    null_rows = []
    failed_rows = []

    # Shuffle the DataFrame to ensure randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    for index, row in df.iterrows():
        # Get the actual number of columns in this row
        row_non_null = row.dropna()
        actual_columns = len(row_non_null)

        # If there are extra columns beyond 'paper', discard the row
        if actual_columns > paper_col_index + 1:
            continue  # Skip the row entirely

        # Get data columns (excluding 'paper')
        data_columns = row.iloc[:paper_col_index]
        # Ensure data_columns is a Series of strings
        data_columns = data_columns.fillna('').astype(str)

        # Get the 'paper' reference
        paper_reference = row.iloc[paper_col_index]

        # If the paper reference is missing, skip the row
        if pd.isnull(paper_reference) or paper_reference == '':
            continue  # Skip the row

        # Check for null row: all data columns are exactly 'null' strings
        if (data_columns == 'null').all():
            null_rows.append([paper_reference])
        # Check for failed row: any data column is exactly 'failed'
        elif (data_columns == 'failed').any():
            failed_rows.append([paper_reference])
        else:
            # Valid row (include up to 'paper' column)
            valid_row = row.iloc[:paper_col_index + 1].tolist()
            valid_rows.append(valid_row)

    # Create DataFrames from the lists
    valid_df = pd.DataFrame(valid_rows)
    null_df = pd.DataFrame(null_rows, columns=['Paper Reference'])
    failed_df = pd.DataFrame(failed_rows, columns=['Paper Reference'])

    # Remove rows from valid_df where all data columns (excluding 'paper' column) are empty strings
    if not valid_df.empty:
        data_columns_valid = valid_df.iloc[:, :-1]  # All columns except the last one (paper column)
        data_columns_valid = data_columns_valid.fillna('').astype(str)
        valid_df = valid_df[~(data_columns_valid == '').all(axis=1)]

    # Randomly select up to 100 rows for each category
    if not valid_df.empty:
        valid_df = valid_df.sample(n=min(100, len(valid_df)), random_state=42)
    if not null_df.empty:
        null_df = null_df.sample(n=min(100, len(null_df)), random_state=42)
    if not failed_df.empty:
        failed_df = failed_df.sample(n=min(100, len(failed_df)), random_state=42)

    # Ensure the output folder exists
    output_folder = os.path.splitext(csv_file)[0]
    os.makedirs(output_folder, exist_ok=True)

    # Save the valid results, including the original headers
    with open(os.path.join(output_folder, 'valid.csv'), 'w', newline='') as f:
        # Write the original headers first
        f.write(','.join(original_headers) + '\n')
        # Write the valid rows
        valid_df.to_csv(f, index=False, header=False)

    # Save null and failed rows
    if not null_df.empty:
        null_df.to_csv(os.path.join(output_folder, 'null.csv'), index=False)
    else:
        print("No null rows to save.")

    if not failed_df.empty:
        failed_df.to_csv(os.path.join(output_folder, 'failed.csv'), index=False)
    else:
        print("No failed rows to save.")

def select_and_process_csv():
    # List all CSV files in the current directory
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    # Print out CSV files to choose from
    print("Available CSV files:")
    for i, csv_file in enumerate(csv_files):
        print(f"{i + 1}. {csv_file}")

    # Get user input to select a file
    choice = int(input("Select the CSV file number to process: ")) - 1

    if 0 <= choice < len(csv_files):
        selected_csv = csv_files[choice]
        print(f"Processing {selected_csv}...")
        process_csv(selected_csv)
        print(f"Processing complete. Output stored in folder: {os.path.splitext(selected_csv)[0]}")
    else:
        print("Invalid selection.")

if __name__ == "__main__":
    select_and_process_csv()
