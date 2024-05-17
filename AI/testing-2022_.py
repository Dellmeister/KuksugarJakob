import pandas as pd
import json

# Load the JSON file into a DataFrame
with open('2022.json', 'r') as file:
    data = json.load(file)

# If the JSON file has nested structures, pandas will automatically flatten them
jobtech_dataset = pd.json_normalize(data)

# Exclude columns containing only "None" values
jobtech_dataset = jobtech_dataset.dropna(axis=1, how='all')

# Optionally, limit the DataFrame to a smaller size for demonstration
jobtech_dataset = jobtech_dataset.head(5)

# Iterate over each row and print it
for index, row in jobtech_dataset.iterrows():
    print(f"Row {index}:")
    for col in jobtech_dataset.columns:
        print(f"    {col}: {row[col]}")
    print("\n---\n")  # Separator between rows
