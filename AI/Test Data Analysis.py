# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Code Snippet 1: Using jsonlines and tqdm
# Purpose: This code is designed to quickly scan through every line of the .jsonl file, 
# collecting all unique keys (which act as column names in the context of a DataFrame) across the entire dataset.
import jsonlines
from tqdm import tqdm

file_path = 'c:/Systemvetenskap/Job Listings Dataset/2023.enriched.jsonl'

all_columns = set()  # Set to hold all unique column names

with jsonlines.open(file_path) as reader:
    for obj in tqdm(reader, desc="Scanning columns"):
        all_columns.update(obj.keys())  # Update the set with keys from the current JSON object

# Print all unique column names found in the dataset
print("All unique column names in the dataset:")
print(all_columns)
##-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Code Snippet 2: Using pandas.read_json with chunksize
# # Purpose: This code aims to load the dataset into a Pandas DataFrame in chunks, 
# # allowing for a quick peek at the structure of the data in terms of column names and the first few rows of the first chunk.

# import pandas as pd

# file_path = 'c:/Systemvetenskap/Python/Dataset Job Listings/2023.enriched.jsonl'

# # Read the first few lines to get a glimpse of the data
# df_sample = pd.read_json(file_path, lines=True, chunksize=100)

# # Get the first chunk
# sample_chunk = next(df_sample)

# # Print column names and the first few rows to inspect the structure
# print(sample_chunk.columns)
# print(sample_chunk.head())
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

# import pandas as pd

# file_path = 'c:/Systemvetenskap/Job Listings Dataset/2023.enriched.jsonl'
# df = pd.read_json(file_path, lines=True)

# # Check the number of columns
# num_columns = len(df.columns)
# if num_columns >= 100:
#     print(f"The dataset has {num_columns} columns, which is >= 100.")
# else:
#     print(f"The dataset has only {num_columns} columns.")

# chunk_iter = pd.read_json(file_path, lines=True, chunksize=50000)
# first_chunk = next(chunk_iter)  # Load the first chunk

# Check the number of columns in the first chunk
# num_columns = len(first_chunk.columns)
# if num_columns >= 100:
#     print(f"The first chunk has {num_columns} columns, assuming the dataset is consistently structured, it should have >= 100 columns.")
# else:
#     print(f"The first chunk has only {num_columns} columns.")


# Check the number of columns in the first chunk
# num_columns = len(first_chunk.columns)
# print(f"The first chunk has {num_columns} columns.")

# # Print the column names
# print("Column names:")
# print(first_chunk.columns.tolist())



## This code prints unique columns for the subset_data.json file
# SUBSET 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# import json
# from tqdm import tqdm

# file_path = 'C:\Systemvetenskap\subset_data.json'
# all_columns = set()  # Set to hold all unique column names

# # Load the entire JSON file
# with open(file_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)  # Assumes the file contains a JSON array of objects

# # Iterate over each object in the array
# for obj in tqdm(data, desc="Scanning columns"):
#     all_columns.update(obj.keys())  # Update the set with keys from the current JSON object

# # Print all unique column names found in the dataset
# print("All unique column names in the dataset:")
# print(all_columns)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# This code prints 20 randow values from description column

# To modify your script to print 20 random values from the Description column instead of scanning for all unique column names, you'll need to adjust the logic. Since your dataset is in .jsonl format, you can iterate through the file, collect Description values, and then select 20 random values from this collection.

# Because you might be dealing with a large dataset and only need a sample, it's efficient to collect descriptions while iterating without loading the entire dataset into memory. To select random values as you go, you could use a reservoir sampling algorithm for efficiency, or if the dataset size is manageable, simply collect all descriptions and then sample from them.

# Here's a simplified approach using the latter strategy, suitable for moderately sized datasets:

# import jsonlines
# from tqdm import tqdm
# import random

# file_path = 'c:/Systemvetenskap/Job Listings Dataset/2023.enriched.jsonl'

# # Initialize an empty list to collect descriptions
# descriptions = []

# # Open the .jsonl file and iterate through each object
# with jsonlines.open(file_path) as reader:
#     for obj in tqdm(reader, desc="Collecting descriptions"):
#         # Check if 'description' key exists and the value is not None
#         if obj.get('description'):
#             descriptions.append(obj['description'])

# # Once all descriptions are collected, sample 20 random values from the list
# # Ensure there are at least 20 descriptions to sample from
# if len(descriptions) >= 20:
#     sampled_descriptions = random.sample(descriptions, 20)
#     print("20 Random Descriptions:")
#     for desc in sampled_descriptions:
#         print(desc)
# else:
#     print("Less than 20 descriptions available.")


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# This code picks out listings related to IT in a list and shows 2 randomly selected listings from this list
# import jsonlines
# from tqdm import tqdm
# import random

# file_path = 'c:/Systemvetenskap/Job Listings Dataset/2023.enriched.jsonl'

# # Keywords that might indicate a listing is related to IT
# it_keywords = ["driven"]

# # Initialize an empty list to collect IT-related descriptions
# it_descriptions = []

# # Open the .jsonl file and iterate through each object
# with jsonlines.open(file_path) as reader:
#     for obj in tqdm(reader, desc="Collecting descriptions"):
#         # Attempt to extract the description text from a nested structure within the description dictionary
#         description_text = obj.get('description', {}).get('text', '')  # Adjust 'text' to the correct key if different
#         description_text = description_text.lower()  # Convert to lowercase for case-insensitive matching
        
#         # Check if any of the IT-related keywords are in the description text
#         if any(keyword in description_text for keyword in it_keywords):
#             it_descriptions.append(description_text)

# # Once IT-related descriptions are collected, sample 2 random values from the list
# # Ensure there are at least 2 descriptions to sample from
# if len(it_descriptions) >= 2:
#     sampled_descriptions = random.sample(it_descriptions, 2)
#     print("2 Random IT-Related Descriptions:")
#     for desc in sampled_descriptions:
#         print(desc)
# else:
#     print("Less than 2 descriptions available.")

