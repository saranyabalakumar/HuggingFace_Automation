import pandas as pd

# Read input data from the text file
with open("HuggingFace_TextModels.txt", "r") as file:
    lines = file.readlines()

# Initialize variables to store information
text_model = ""
framework = ""
use_case = ""
model = ""
latency = ""
cpu_utilization = ""
memory_utilization = ""
gops = ""

# Initialize an empty list to store data
data = []

# Process the lines and extract information
for line in lines:
    line = line.strip()
    if line.startswith("Latency in ms"):
        latency = round(float(line.split(":")[1].strip()), 2)
    elif line.startswith("CPU Utilization"):
        cpu_utilization = float(line.split(":")[1].strip()[:-1])  # Remove the '%' symbol
    elif line.startswith("Memory Utilization"):
        memory_utilization = float(line.split(":")[1].strip()[:-1])  # Remove the '%' symbol
    elif line.startswith("Number of GOPs"):
        gops_str = line.split(":")[1].strip().replace(" GOPs", "")
        gops = round(float(gops_str), 2)
        data.append([text_model, framework, use_case, model, latency, cpu_utilization, memory_utilization, gops])
    elif "_" in line and "[" not in line:
        parts = line.split("_")
        if len(parts) >= 4:
            text_model, framework, use_case, model = parts[:4]

# Create a DataFrame from the extracted data
df = pd.DataFrame(data, columns=["TextModels", "Framework", "UseCase", "Models", "Latency in ms", "CPU Utilization in %", "Memory Utilization in %", "GOPs"])

# Write the DataFrame to a CSV file
df.to_csv("HuggingFace_TextModels.csv", index=False)
print("Output saved to HuggingFace_TextModels.csv")
