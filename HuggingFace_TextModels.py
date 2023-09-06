import os
import glob

# Define the folder path where your Python code files are located
folder_path = '/home/saranya-19160/Saranya/Huggingface/Text_Models'

# Use glob to get a list of all Python files in the folder
python_files = glob.glob(os.path.join(folder_path, '*.py'))

# Iterate through the Python files and execute them
for python_file in python_files:
    try:
        with open(python_file, 'r') as file:
            code = file.read()
        # Execute the code using exec
        exec(code)
        print(f"Executed: {python_file}")
    except Exception as e:
        print(f"Error executing {python_file}: {e}")

