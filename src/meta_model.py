import os
import sys
from src.utils import print  # Custom print function for logging

def create_schema_file(fields, key_columns, filename='./dataModels/default_schema.txt'):
    """
    Generates a schema file with the specified fields and types, then writes it to a text file.

    Args:
    fields (list): A list of dictionaries representing the fields in the schema.
    key_columns (list): A list of column numbers to be used as keys for checking duplicates.
    filename (str): The name of the file to write the schema to.
    """
    try:
        # Create the schema definition as a string
        schema_definition = ""
        for field in fields:
            schema_definition += f"{field['column_number']} - Type: \'{field['type']}\'\n"
            schema_definition += f"{field['column_number']} - Name: \'{field['name']}\'\n"
            schema_definition += f"{field['column_number']} - Description: \'{field['description']}\'\n\n"

        # Write the schema definition to a file
        with open(f"{filename}.pkl", 'w') as file:
            file.write(f"Key Columns: {','.join(str(column) for column in key_columns)}\n\n")
            file.write(schema_definition)

        print(f"Schema file has been written to {filename}")
        return True

    except:
        return False

def UI_schema_creator():
    """
    Interactive function to create a schema file based on user input.
    """
    # Get the number of columns from the user
    numColumns = 0
    while numColumns < 1:
        try:
            numColumns = int(input("Please enter the number of columns you would like in the CSV: "))
        except ValueError:
            print("Must be an integer!")
        if numColumns < 1:
            print("Integer must be positive!")

    fields = []
    # Collect information for each column
    for i in range(numColumns):
        columnName = input(f"Please enter the column name for column {i + 1}: ")
        columnType = input(f"""
        Please choose from the following options for the data type of column {i + 1}:
        1. String
        2. Integer
        3. Float (number with decimals)
        4. Complex number
        5. Range
        6. Boolean

        Choice: """)

        # Map user input to Python data types ## Never gets used in this function/scope
        # type_mapping = {
        #     "1": "str",
        #     "2": "int",
        #     "3": "float",
        #     "4": "complex",
        #     "5": "range",
        #     "6": "bool"
        # }

        columnDescription = input(f"Please enter a brief description of the data to be stored in this column: ")

        # Collect additional information based on the column type
        if columnType == "1":  # String
            # Get allowed values, length constraints, and substring restrictions
            allowed_values_input = input(
                f"Enter allowed values for column {i + 1}, separated by commas (leave blank for no restriction): ")
            allowed_values = [value.strip() for value in allowed_values_input.split(',') if value.strip()]
            min_length = input(f"Enter the minimum length for column {i + 1} (leave blank for no minimum): ")
            max_length = input(f"Enter the maximum length for column {i + 1} (leave blank for no maximum): ")
            whitelist_substrings = input(
                f"Enter required substrings for column {i + 1}, separated by commas (leave blank for none): ").split(
                ',')
            blacklist_substrings = input(
                f"Enter blacklisted substrings for column {i + 1}, separated by commas (leave blank for none): ").split(
                ',')
            fields.append({
                'column_number': i + 1,
                'type': columnType,
                'name': columnName,
                'description': columnDescription,
                'min_length': int(min_length) if min_length else None,
                'max_length': int(max_length) if max_length else None,
                'whitelist_substrings': [substring.strip() for substring in whitelist_substrings if substring.strip()],
                'blacklist_substrings': [substring.strip() for substring in blacklist_substrings if substring.strip()],
                'allowed_values': allowed_values if allowed_values else None
            })
        elif columnType == "2":  # Integer
            # Get allowed values and range constraints
            allowed_values_input = input(
                f"Enter allowed values for column {i + 1}, separated by commas (leave blank for no restriction): ")
            allowed_values = [value.strip() for value in allowed_values_input.split(',') if value.strip()]
            min_value = input(f"Enter the minimum value for column {i + 1} (leave blank for no minimum): ")
            max_value = input(f"Enter the maximum value for column {i + 1} (leave blank for no maximum): ")
            fields.append({
                'column_number': i + 1,
                'type': columnType,
                'name': columnName,
                'description': columnDescription,
                'min_value': int(min_value) if min_value else None,
                'max_value': int(max_value) if max_value else None,
                'allowed_values': allowed_values if allowed_values else None
            })
        elif columnType == "3":  # Float
            # Get allowed values and range constraints
            allowed_values_input = input(
                f"Enter allowed values for column {i + 1}, separated by commas (leave blank for no restriction): ")
            allowed_values = [value.strip() for value in allowed_values_input.split(',') if value.strip()]
            min_value = input(f"Enter the minimum value for column {i + 1} (leave blank for no minimum): ")
            max_value = input(f"Enter the maximum value for column {i + 1} (leave blank for no maximum): ")
            fields.append({
                'column_number': i + 1,
                'type': columnType,
                'name': columnName,
                'description': columnDescription,
                'min_value': float(min_value) if min_value else None,
                'max_value': float(max_value) if max_value else None,
                'allowed_values': allowed_values if allowed_values else None
            })
        elif columnType == "5":  # Range
            # Get range constraints
            min_value = input(f"Enter the lowest minimum value for column {i + 1} (leave blank for no minimum): ")
            max_value = input(f"Enter the highest maximum value for column {i + 1} (leave blank for no maximum): ")
            fields.append({
                'column_number': i + 1,
                'type': columnType,
                'name': columnName,
                'description': columnDescription,
                'min_value': float(min_value) if min_value else None,
                'max_value': float(max_value) if max_value else None
            })
        else:  # Other types (Complex, Boolean)
            fields.append({
                'column_number': i + 1,
                'type': columnType,
                'name': columnName,
                'description': columnDescription
            })

    # Get key columns for duplicate checking
    key_columns = []
    while True:
        key_column_input = input(
            "Enter the column numbers to be used as keys for checking duplicates, separated by commas (leave blank for none): ")
        if key_column_input == "":
            break
        try:
            key_columns = [int(column.strip()) for column in key_column_input.split(',')]
            if any(column < 1 or column > len(fields) for column in key_columns):
                print("Invalid column number. Please try again.")
                key_columns = []
            else:
                break
        except ValueError:
            print("Invalid input. Please enter valid column numbers separated by commas.")

    # Allow user to specify a file name
    filename = input("Please enter a name for the schema file (default 'default_schema.txt'): ") or 'default_schema.txt'
    filename = "./dataModels/" + filename

    # Create the schema file
    create_schema_file(fields, key_columns, filename)

    # Restart the script
    python = sys.executable
    os.execl(python, python, *sys.argv)