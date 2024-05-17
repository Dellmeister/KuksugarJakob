## Cleans the csv file, removing common formatting errors

# import csv

# def clean_csv(input_filename, output_filename):
#     with open(input_filename, 'r', newline='', encoding='utf-8') as infile, \
#          open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
#         reader = csv.reader(infile)
#         writer = csv.writer(outfile)

#         for row in reader:
#             # Clean up rows by removing unexpected line breaks and fixing quotes
#             cleaned_row = [field.replace('\n', ' ').replace('\r', '').replace('"', "'") for field in row]
#             writer.writerow(cleaned_row)

# if __name__ == "__main__":
#     input_path = '2023_vasili.csv'
#     output_path = '2023_vasili_clean.csv'
#     clean_csv(input_path, output_path)


#---------------------------------------------------------------------------------------------------------------

# import csv
# from tqdm import tqdm

# def clean_csv(input_filename, output_filename):
#     with open(input_filename, 'r', newline='', encoding='utf-8') as infile, \
#          open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
#         reader = csv.reader(infile)
#         writer = csv.writer(outfile)
        
#         # Determine the total number of rows for the progress bar
#         total_rows = sum(1 for row in infile)
#         infile.seek(0)  # Reset file pointer to the start after counting

#         # Initialize tqdm progress bar
#         progress_bar = tqdm(reader, total=total_rows, desc="Cleaning CSV")

#         for row in progress_bar:
#             # Clean up rows by removing unexpected line breaks and fixing quotes
#             cleaned_row = [field.replace('\n', ' ').replace('\r', '').replace('"', "'") for field in row]
#             writer.writerow(cleaned_row)

# if __name__ == "__main__":
#     input_path = '2023_vasili.csv'
#     output_path = '2023_vasili_clean.csv'
#     clean_csv(input_path, output_path)


#----------------------------------------------------------------------------------------------------------------

# import csv
# from tqdm import tqdm

# def clean_csv(input_filename, output_filename):
#     try:
#         with open(input_filename, 'r', newline='', encoding='utf-8') as infile, \
#             open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
#             reader = csv.reader(infile)
#             writer = csv.writer(outfile)
            
#             # Determine the total number of rows for the progress bar
#             total_rows = sum(1 for row in infile)
#             infile.seek(0)  # Reset file pointer to the start after counting

#             # Initialize tqdm progress bar
#             progress_bar = tqdm(reader, total=total_rows, desc="Cleaning CSV")

#             for row in progress_bar:
#                 # Clean up rows by removing unexpected line breaks and fixing quotes
#                 cleaned_row = [field.replace('\n', ' ').replace('\r', '').replace('"', "'") for field in row]
#                 writer.writerow(cleaned_row)
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     input_path = '2023_vasili.csv'
#     output_path = '2023_vasili_clean.csv'
#     clean_csv(input_path, output_path)

#-----------------------------------------------------------------------------------------------------------

# import csv
# from tqdm import tqdm
# import logging

# logging.basicConfig(filename='cleaning_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

# def clean_csv(input_filename, output_filename):
#     try:
#         with open(input_filename, 'r', newline='', encoding='utf-8') as infile, \
#             open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
#             reader = csv.reader(infile)
#             writer = csv.writer(outfile)
            
#             # Log the start
#             logging.info("Starting to count total rows...")
#             total_rows = sum(1 for row in infile)
#             logging.info(f"Total rows counted: {total_rows}")
            
#             infile.seek(0)  # Reset file pointer to the start after counting
#             logging.info("File pointer reset to start.")

#             # Initialize tqdm progress bar
#             progress_bar = tqdm(reader, total=total_rows, desc="Cleaning CSV")

#             for i, row in enumerate(progress_bar):
#                 # Log progress
#                 if i % 1000 == 0:
#                     logging.info(f"Processing row {i}")
                
#                 cleaned_row = [field.replace('\n', ' ').replace('\r', '').replace('"', "'") for field in row]
#                 writer.writerow(cleaned_row)

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     input_path = '2023_vasili.csv'
#     output_path = '2023_vasili_clean.csv'
#     clean_csv(input_path, output_path)


#----------------------------------------------------------------------------------------------------------------
#1
# import pandas as pd
# from tqdm import tqdm

# def clean_and_save_chunk(df, outfile, header=True):
#     # Clean data
#     df = df.applymap(lambda x: x.replace('\n', ' ').replace('\r', '').replace('"', "'") if isinstance(x, str) else x)
#     # Save to CSV
#     df.to_csv(outfile, mode='a', index=False, header=header)

# # Define file paths
# input_path = '2023_vasili.csv'  # Ensure this path is correct
# output_path = '2023_vasili_clean.csv'

# # Read in chunks
# chunk_size = 10000  # Adjust as needed based on your system's capability
# reader = pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)

# # Initialize progress bar with the total number of chunks approximated
# total_chunks = sum(1 for row in open(input_path, 'r', encoding='utf-8')) // chunk_size
# progress_bar = tqdm(total=total_chunks, desc="Processing chunks")

# first_chunk = True
# for chunk in reader:
#     clean_and_save_chunk(chunk, output_path, header=first_chunk)
#     first_chunk = False
#     progress_bar.update(1)  # Update progress for each chunk processed

# progress_bar.close()


#2 

import pandas as pd
from tqdm import tqdm

def clean_and_save_chunk(df, outfile, header=True):
    try:
        # Clean data
        df = df.applymap(lambda x: x.replace('\n', ' ').replace('\r', '').replace('"', "'") if isinstance(x, str) else x)
        # Save to CSV
        df.to_csv(outfile, mode='a', index=False, header=header)
    except Exception as e:
        print("Error processing a chunk:", e)
        raise

# Define file paths
input_path = '2023_vasili.csv'
output_path = '2023_vasili_clean.csv'

try:
    # Read in chunks
    chunk_size = 10000
    reader = pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)

    # Initialize progress bar with the total number of chunks approximated
    total_chunks = sum(1 for row in open(input_path, 'r', encoding='utf-8')) // chunk_size
    progress_bar = tqdm(total=total_chunks, desc="Processing chunks")

    first_chunk = True
    for chunk in reader:
        clean_and_save_chunk(chunk, output_path, header=first_chunk)
        first_chunk = False
        progress_bar.update(1)  # Update progress for each chunk processed

    progress_bar.close()
except Exception as e:
    print("Failed to read or process the file:", e)
