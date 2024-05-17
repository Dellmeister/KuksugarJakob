# import pandas as pd

# # Load the data from CSV
# df = pd.read_csv('2023.enriched_preprocessed.csv')


# # Filter rows where the description mentions 'NLP'
# nlp_jobs = df[df['Description.text'].str.contains('NLP', case=False, na=False)]

# # Extract job titles: this example assumes job titles end with a colon or specific phrase
# # Adjust the regular expression as necessary based on your data's format
# nlp_jobs['Extracted Title'] = nlp_jobs['Description'].str.extract(r'^(.*?):')

# # Count how many times each extracted job title appears
# title_counts = nlp_jobs['Extracted Title'].value_counts()

# # Display the most common job titles
# print(title_counts)

#************************************************************************************************************

# import pandas as pd

# # Load the data from CSV with low_memory set to False
# df = pd.read_csv('2023.enriched_preprocessed.csv', low_memory=False)

# # Print the column names to verify the correct column name
# print(df.columns)

# # Make sure the column name you use matches exactly with what's printed
# # Adjust the following code to use the correct column name

# # Assuming the correct column name is 'Description' and it does exist
# try:
#     nlp_jobs = df[df['Description.text'].str.contains('NLP', case=False, na=False)]
#     nlp_jobs['Extracted Title'] = nlp_jobs['Description.text'].str.extract(r'^(.*?):')
#     title_counts = nlp_jobs['Extracted Title'].value_counts()
#     print(title_counts)
# except KeyError:
#     print("Column 'Description' does not exist. Please check the column names listed above.")

#*************************************************************************************************************

# import pandas as pd

# # Load the data from CSV with low_memory set to False
# df = pd.read_csv('2023.enriched_preprocessed.csv', low_memory=False)

# # Print the column names to verify the correct column name
# print(df.columns)

# # Since we know the column 'description.text' exists as per your output, let's work directly with it.
# try:
#     # Filter rows where the description mentions 'NLP'
#     nlp_jobs = df[df['description.text'].str.contains('NLP', case=False, na=False)]

#     # Attempt to extract job titles if they are mentioned before a colon in the description
#     # This regular expression assumes that the job title ends with a colon
#     nlp_jobs['Extracted Title'] = nlp_jobs['description.text'].str.extract(r'^(.*?):')

#     # Count how many times each extracted job title appears
#     title_counts = nlp_jobs['Extracted Title'].value_counts()

#     # Display the most common job titles
#     print(title_counts)
# except KeyError as e:
#     print(f"Error: Column not found - {e}")
# except Exception as e:
#     print(f"An error occurred: {e}")

#*************************************************************************************************************

# import pandas as pd

# # Load the data from CSV with low_memory set to False
# df = pd.read_csv('2023.enriched_preprocessed.csv', low_memory=False)

# # Print the column names to verify the correct column name
# print(df.columns)

# # Filter rows where the description mentions 'NLP'
# nlp_jobs = df[df['description.text'].str.contains('NLP', case=False, na=False)].copy()

# # Safely extract job titles if they are mentioned before a colon in the description
# # Use .loc to avoid SettingWithCopyWarning
# nlp_jobs.loc[:, 'Extracted Title'] = nlp_jobs['description.text'].str.extract(r'^(.*?):')

# # Count how many times each extracted job title appears
# title_counts = nlp_jobs['Extracted Title'].value_counts()

# # Display the most common job titles
# print(title_counts)


#***************************************************************************************************************

# import pandas as pd

# # Load the data from CSV with low_memory set to False
# df = pd.read_csv('2023.enriched_preprocessed.csv', low_memory=False)

# # Filter rows where the description mentions 'NLP'
# nlp_jobs = df[df['description.text'].str.contains('NLP', case=False, na=False)].copy()

# # Safely extract job titles if they are mentioned before a colon in the description
# # Adjust the regex if necessary
# nlp_jobs['Extracted Title'] = nlp_jobs['description.text'].str.extract(r'^(.*?):')

# # Debugging: print out some of the extracted titles to inspect them
# print(nlp_jobs['Extracted Title'].head(10))

# # Count how many times each extracted job title appears
# title_counts = nlp_jobs['Extracted Title'].value_counts()

# # Display the most common job titles
# print(title_counts)

#****************************************************************************************************************

### VERKAR TA UT EN LISTA PÅ MEST FÖREKOMMANDE ORDEN

# import pandas as pd

# # Load the data from CSV with low_memory set to False
# df = pd.read_csv('2023.enriched_preprocessed.csv', low_memory=False)

# # Filter rows where the description mentions 'NLP'
# nlp_jobs = df[df['description.text'].str.contains('NLP', case=False, na=False)].copy()

# # Attempt to extract more precise job titles using a refined regex
# # Adjust this regex based on actual content observations
# nlp_jobs['Extracted Title'] = nlp_jobs['description.text'].str.extract(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')

# # Debugging: print out some of the extracted titles to inspect them
# print(nlp_jobs['Extracted Title'].head(20))

# # Count how many times each extracted job title appears
# title_counts = nlp_jobs['Extracted Title'].value_counts()

# # Display the most common job titles
# print(title_counts)

#*****************************************************************************************************************

# import pandas as pd

# # Load the data
# df = pd.read_csv('2023.enriched_preprocessed.csv', low_memory=False)

# # Define a list of keywords that are common in job titles
# keywords = ['Engineer', 'Manager', 'Developer', 'Consultant', 'Specialist', 'Director']

# # Function to search and extract job titles based on keywords
# def extract_title(text):
#     words = text.split()
#     for idx, word in enumerate(words):
#         if any(key in word for key in keywords):
#             # This tries to capture the word with one word before and after as context
#             return ' '.join(words[max(0, idx-1):idx+2])
#     return 'No title found'  # Return a placeholder if no keyword is found

# # Apply the function to the description column
# df['Extracted Title'] = df['description.text'].apply(extract_title)

# # Count occurrences of extracted titles
# title_counts = df['Extracted Title'].value_counts()

# # Display results
# print(title_counts.head(20))

#**********************************************************************************************************

##VISAR EN LISTA PÅ HUR MÅNGA AV OLIKA JOBBTITLAR SOM FINNS

# import pandas as pd

# # Load the data
# df = pd.read_csv('2023.enriched_preprocessed.csv', low_memory=False)

# # Define a list of keywords that are common in job titles
# keywords = ['Engineer', 'Manager', 'Developer', 'Consultant', 'Specialist', 'Director']

# # Function to search and extract job titles based on keywords
# def extract_title(text):
#     # Check if the text is a string (not NaN)
#     if isinstance(text, str):
#         words = text.split()
#         for idx, word in enumerate(words):
#             if any(key in word for key in keywords):
#                 # This tries to capture the word with one word before and after as context
#                 return ' '.join(words[max(0, idx-1):idx+2])
#     return 'No title found'  # Return a placeholder if no keyword is found or text is NaN

# # Apply the function to the description column
# df['Extracted Title'] = df['description.text'].apply(extract_title)

# # Count occurrences of extracted titles
# title_counts = df['Extracted Title'].value_counts()

# # Display results
# print(title_counts.head(20))

#**************************************************************************************************************

import pandas as pd

# Load the data
#df = pd.read_csv('2023.enriched_preprocessed.csv', low_memory=False)
df = pd.read_csv('2023_vasili.csv', low_memory=False)

# Define a list of keywords that are common in job titles
keywords = ['Engineer', 'Manager', 'Developer', 'Consultant', 'Specialist', 'Director']

# Function to search and extract job titles based on keywords
def extract_title(text):
    # Check if the text is a string (not NaN)
    if isinstance(text, str):
        words = text.split()
        # Ensure case insensitivity by comparing lowercased words and keywords
        for idx, word in enumerate(words):
            if any(key.lower() in word.lower() for key in keywords):
                # Capture two words before and after the keyword for context
                start = max(0, idx-2)
                end = min(len(words), idx+3)
                return ' '.join(words[start:end])
    return 'No title found'  # Return a placeholder if no keyword is found or text is NaN

# Apply the function to the description column
df['Extracted Title'] = df['description.text'].apply(extract_title)

# Count occurrences of extracted titles
title_counts = df['Extracted Title'].value_counts()

# Display results
print(title_counts.head(20))
