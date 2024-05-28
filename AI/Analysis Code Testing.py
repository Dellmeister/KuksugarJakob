# Here Goes Analysis Code


# import os
# # Set the directory and load the CSV
# directory = 'c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis'
# os.chdir(directory)
# print("Current Working Directory:", os.getcwd())




## make subset 

# import pandas as pd
# import random

# # Read the CSV file
# df = pd.read_csv('description-column.csv')

# # Check the number of rows in the dataframe
# num_rows = len(df)

# # Set the number of rows you want in the subset
# subset_size = 1000

# # Check if the dataset has more than 1000 rows
# if num_rows > subset_size:
#     # Select a random subset of 1000 rows
#     random_indices = random.sample(range(num_rows), subset_size)
#     subset_df = df.iloc[random_indices]
# else:
#     # If the dataset has 1000 or fewer rows, use the entire dataset
#     subset_df = df

# # Write the subset to a new CSV file
# subset_df.to_csv('description_subset_1000rows.csv', index=False)




######################################## JSONL SPLIT   ############################################################
# import os

# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())
# import json
# from sklearn.model_selection import train_test_split

# def split_data(input_file, train_file, val_file, test_size=0.2):
#     # Read all data from the input file
#     with open(input_file, 'r', encoding='utf-8') as file:
#         lines = file.readlines()

#     # Split data into training and validation sets
#     train_lines, val_lines = train_test_split(lines, test_size=test_size, random_state=42)

#     # Write training data to the train file
#     with open(train_file, 'w', encoding='utf-8') as file:
#         for line in train_lines:
#             file.write(line)

#     # Write validation data to the validation file
#     with open(val_file, 'w', encoding='utf-8') as file:
#         for line in val_lines:
#             file.write(line)

# # Usage
# input_jsonl = '2023.enriched_cleaned.jsonl'
# train_jsonl = '2023.enriched_cleaned_train.jsonl'
# val_jsonl = '2023.enriched_cleaned_validation.jsonl'
# split_data(input_jsonl, train_jsonl, val_jsonl)






######################################### JSONL FILE COLUMNS_CLEANED_v1 ############################################################

# import os

# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())

# import json
# import logging

# # Setup basic configuration for logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # List of columns to keep from the dataset
# columns_to_keep = [
#     "headline", "employment_type", "salary_description", "experience_required", "access_to_own_car", 
#     "driving_license_required", "occupation", "occupation_group", "occupation_field", "remote_work", 
#     "detected_language", "description.text", "description.conditions", "salary_type.label",
#     "duration.label", "working_hours_type.label", "employer.name", "workplace_address.municipality",
#     "workplace_address.region"
# ]

# def filter_jsonl(input_file, output_file):
#     try:
#         with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#             for line in infile:
#                 data = json.loads(line)
#                 filtered_data = {}

#                 for key in columns_to_keep:
#                     keys = key.split('.')
#                     temp_data = data
                    
#                     try:
#                         for sub_key in keys:
#                             temp_data = temp_data[sub_key]
#                         filtered_data[key] = temp_data
#                     except KeyError:
#                         logging.debug(f"Key {key} not found in the data.")
                
#                 json.dump(filtered_data, outfile)
#                 outfile.write('\n')
        
#         logging.info(f"Data successfully filtered and saved to {output_file}")
#     except FileNotFoundError:
#         logging.error(f"The file {input_file} does not exist.")
#     except Exception as e:
#         logging.error(f"An error occurred: {e}")

# # Usage
# input_jsonl = '2023.enriched.jsonl'
# output_jsonl = '2023.enriched_cleaned.jsonl'
# filter_jsonl(input_jsonl, output_jsonl)


############################################################################################################################################


# import os
# import streamlit as st
# from openai import OpenAI

# client = OpenAI()
# from openai import OpenAI

# client = OpenAI(api_key=["SECRET"])

# import pdfplumber
# from io import BytesIO

# # Function to load CSS
# def load_css(file_name):
#     # Get the directory of the current script
#     script_dir = os.path.dirname(__file__)

#     # Construct the absolute path to the CSS file
#     file_path = os.path.join(script_dir, file_name)

#     with open(file_path) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# def get_recommendations(text, gender, experience, age, language):
#     if language == 'Swedish':
#         prompt = f"{text}\n\nGivet att den ideala kandidaten är {gender}, {experience}, och {age}, hur kan denna jobbannons förbättras?"
#         system_message = "Du är en hjälpsam assistent."
#     else:  # Default to English
#         prompt = f"{text}\n\nGiven that the ideal candidate is {gender}, {experience}, and {age}, how could this job posting be improved?"
#         system_message = "You are a helpful assistant."

#     response = client.completions.create(model="gpt-3.5-turbo-instruct",
#     prompt=prompt,
#     max_tokens=500,
#     temperature=0.7)

#     return response.choices[0].text.strip()

# # Function to read file
# def read_file(file):
#     if file.type == 'application/pdf':
#         with pdfplumber.open(BytesIO(file.getvalue())) as pdf:
#             return ' '.join(page.extract_text() for page in pdf.pages)
#     else:
#         return file.getvalue().decode()

# # Load CSS
# load_css('styles.css')

# # Sidebar
# st.sidebar.title('Options')

# # Add a language selection option
# language = st.sidebar.radio('Language', ['English', 'Swedish'])

# gender = st.sidebar.radio('Gender Preference', ['N/A', 'Male', 'Female', 'Non-binary'])
# experience = st.sidebar.radio('Experience Preference', ['N/A', 'Entry Level', 'Mid Level', 'Experienced'])
# age = st.sidebar.radio('Age', ['N/A', 'Young', 'Middle aged', 'Old'])

# # Main Area
# st.title('CoRecruit AI')

# uploaded_file = st.file_uploader("Upload a job posting", type=['txt', 'pdf'])

# if uploaded_file is not None:
#     # Process the text from the job posting
#     text = read_file(uploaded_file)

#     # Use the GPT API to recommend changes
#     recommendations = get_recommendations(text, gender, experience, age, language)
#     st.write(recommendations)



######################################################  DATA SPLIT TEST  ##########################################################################################################



# import os

# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())

# import pandas as pd
# from sklearn.model_selection import train_test_split

# def split_large_dataset(file_path, chunk_size=100000):
#     # Initialize empty lists to store chunks of training and validation sets
#     train_frames = []
#     val_frames = []

#     # Read the CSV file in chunks
#     for chunk in pd.read_csv(file_path, chunksize=chunk_size):
#         # Split each chunk into training and validation
#         train_chunk, val_chunk = train_test_split(chunk, test_size=0.3, random_state=42)
#         train_frames.append(train_chunk)
#         val_frames.append(val_chunk)

#     # Concatenate all chunks into one DataFrame for each set
#     train_set = pd.concat(train_frames)
#     validation_set = pd.concat(val_frames)

#     # Save the training and validation sets to new CSV files
#     train_set.to_csv('train_set_2023.csv', index=False)
#     validation_set.to_csv('validation_set_2023.csv', index=False)

#     print("Dataset split into training and validation sets and saved as CSV files.")

# # Replace 'path_to_your_large_file.csv' with the path to your dataset file
# file_path = '2023.csv'
# split_large_dataset(file_path)



########################################   CLASSIFIER TEST   ########################################################################################################################


# import os

# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())


# # file_path = '2023_preprocessed_subset_notnull_removed columns_feature_extraction_250000rows.csv'
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.utils.class_weight import compute_class_weight

# # Load the dataset
# file_path = '2023_preprocessed_subset_notnull_removed columns_feature_extraction_250000rows.csv'
# data = pd.read_csv(file_path, low_memory=False)

# # Combine text columns into a single text feature
# data['headline'] = data['headline'].fillna('').astype(str)
# data['occupation'] = data['occupation'].fillna('').astype(str)
# data['occupation_group'] = data['occupation_group'].fillna('').astype(str)
# data['occupation_field'] = data['occupation_field'].fillna('').astype(str)
# data['combined_text'] = data['headline'] + ' ' + data['occupation'] + ' ' + data['occupation_group'] + ' ' + data['occupation_field']

# # Encoding the job categories (labels)
# label_encoder = LabelEncoder()
# data['job_category'] = label_encoder.fit_transform(data['occupation_group'])  # Use the actual category label column

# # Calculate class weights for all unique categories in the entire dataset
# all_classes = np.unique(data['job_category'])
# class_weights = compute_class_weight(class_weight='balanced', classes=all_classes, y=data['job_category'])
# class_weight_dict = dict(zip(all_classes, class_weights))

# # Splitting the dataset
# X_train, X_test, y_train, y_test = train_test_split(data['combined_text'], data['job_category'], test_size=0.2, random_state=42)

# # Text feature extraction
# tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# # Classifier - Using SVM with class weights
# svm_classifier = SVC(kernel='linear', class_weight=class_weight_dict, random_state=42)

# # Creating a pipeline
# pipeline = Pipeline([
#     ('tfidf', tfidf_vectorizer),
#     ('classifier', svm_classifier)
# ])

# # Training the model
# pipeline.fit(X_train, y_train)

# # Predicting the test set results
# y_pred = pipeline.predict(X_test)

# # Evaluating the model
# print("Adjusted Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))



########################################     ########################################################################################################################


# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline

# # Load the dataset
# file_path = '2023_preprocessed_subset_notnull_removed columns_feature_extraction_250000rows.csv'
# data = pd.read_csv(file_path)


# # Assuming 'employment_type_label' is a preprocessed column from 'employment_type' JSON
# X = data['headline']  # or combine with other text columns
# y = data['employment_type_label']  # This needs to be extracted and encoded from JSON

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(stop_words='swedish')),
#     ('clf', RandomForestClassifier(n_estimators=100))
# ])

# pipeline.fit(X_train, y_train)
# print("Accuracy on test set:", pipeline.score(X_test, y_test))


# # Predicting employment type for new job listings
# new_job_listings = ["Developer needed for a startup", "Part-time sales associate", "Project manager for short-term project"]
# predicted_types = pipeline.predict(new_job_listings)

# # Optionally, getting the probability of each category
# predicted_probabilities = pipeline.predict_proba(new_job_listings)

# for job, prediction, probabilities in zip(new_job_listings, predicted_types, predicted_probabilities):
#     print(f"Job: {job}")
#     print(f"Predicted Employment Type: {prediction}")
#     print(f"Probability Distribution: {probabilities}\n")




############################################################ Display Info ################################################################################

# import pandas as pd

# # Load the dataset
# file_path = '2023_preprocessed_subset_notnull_removed columns_feature_extraction_250000rows.csv'
# data = pd.read_csv(file_path)

# # Display the first few rows of the dataset
# print(data.head())

# # Display summary information about the dataset
# print(data.info())


############################################################       ####################################################################################################


# import pandas as pd
# import os

# try:
#     # Set the path of your preprocessed file
#     input_file_path = '2023_preprocessed_subset_notnull_removed columns_feature_extraction_250000rows.csv'
    
#     # Check if the file exists
#     if not os.path.exists(input_file_path):
#         print(f"Error: No file found at {input_file_path}")
#         exit()
    
#     print("Loading data from file...")
#     df = pd.read_csv(input_file_path, low_memory=False)
#     print(f"Total rows loaded: {len(df)}")

#     # # Optionally, you can drop more columns if there are any that are not useful for the model
#     # columns_to_drop = ['some_additional_column_to_drop']
#     # df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
#     # Sampling a subset of the dataset to reduce size
#     df_sample = df.sample(n=2000)  # Adjust n to your needs, depending on the size limit
#     print(f"Sampled {len(df_sample)} rows.")

#     # Saving to a new CSV file
#     output_file_path = input_file_path.replace('.csv', '_smaller.csv')
#     df_sample.to_csv(output_file_path, index=False)
#     print(f"Smaller dataset saved successfully at {output_file_path}")

# except FileNotFoundError as e:
#     print(f"File not found error: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")


################################################### CLEANING DATA #############################################################################################################

# import pandas as pd
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')

# try:
#     print("Reading data from CSV file...")
#     file_path = os.path.join(os.path.dirname(__file__), "2023.csv")
#     if not os.path.exists(file_path):
#         print(f"Error: No file found at {file_path}")
#         exit()
#     print(f"CSV file path read successfully: {file_path}")


#     print("Loading a subset of data from CSV file...")
#     # Estimate the number of rows to load, example: load about 500,000 rows
#     df = pd.read_csv(file_path, low_memory=False, nrows=100000)
#     print(f"Data subset loaded successfully. Total rows loaded: {len(df)}")

#     # Initial removal of specific columns (safe to drop)
#     columns_to_drop = [

#     'id', 'logo_url', 'webpage_url', 'external_id', 'original_id', 'keywords.extracted.occupation', 
#     'keywords.extracted.skill', 'keywords.extracted.location', 'keywords.extracted.employer', 'application_deadline', 
#     'application_contacts', 'timestamp', 'salary_type.concept_id', 'salary_type.legacy_ams_taxonomy_id', 'workplace_address.municipality_concept_id', 
#     'working_hours_type.legacy_ams_taxonomy_id', 'workplace_address.region_concept_id', 'application_details.via_af', 'duration.legacy_ams_taxonomy_id',
#     'salary_description', 'access'
    
#     ]
#     columns_to_drop = [col for col in columns_to_drop if col in df.columns]
#     df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
#     print(f"Additional specified columns dropped: {columns_to_drop}")

#     # Analyzing null values
#     print("Analyzing null values in columns...")
#     null_percentage = df.isnull().mean() * 100
#     print("Percentage of null values per column:")
#     print(null_percentage)

#     # Dropping columns based on null percentage threshold
#     threshold = 30
#     columns_to_drop = null_percentage[null_percentage > threshold].index
#     columns_to_drop = [col for col in columns_to_drop if col in df.columns]
#     df.drop(columns=columns_to_drop, inplace=True)
#     print(f"Columns with more than {threshold}% null values dropped.")

#     # Drop columns with 100% null values
#     columns_to_drop = null_percentage[null_percentage == 100].index
#     columns_to_drop = [col for col in columns_to_drop if col in df.columns]
#     df.drop(columns=columns_to_drop, inplace=True)
#     print(f"Columns with 100% null values dropped.")

#     print("Remaining columns:")
#     print(df.columns)

#     print("Cleaning data...")
#     df.drop_duplicates(inplace=True)
#     print("Data cleaned.")

#     print("Extracting features...")
#     swedish_stopwords = stopwords.words('swedish')
#     text_columns = ['description.text']  # Adjust this if your text column has a different name
#     existing_text_columns = [col for col in text_columns if col in df.columns]


#     if not existing_text_columns:
#         print("No text columns found for feature extraction.")
#     else:
#         # Combine text data into a single column if there are multiple text fields
#         df['combined_text'] = df[existing_text_columns].fillna('').agg(' '.join, axis=1)

#         # Check if there is any non-empty content to vectorize
#         if df['combined_text'].str.strip().eq('').all():
#             print("All text data is empty after stopwords removal.")
#         else:
#             try:
#                 tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words=swedish_stopwords, min_df=2)
#                 tfidf_features = tfidf_vectorizer.fit_transform(df['combined_text'])
#                 if tfidf_features.shape[1] == 0:
#                     print("No features were extracted. Adjusting parameters might be necessary.")
#                 else:
#                     tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
#                     df_tfidf = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_feature_names)
#                     df = pd.concat([df.drop(existing_text_columns + ['combined_text'], axis=1), df_tfidf], axis=1)
#                     print("TF-IDF features extracted and added.")
#             except ValueError as e:
#                 print(f"Error during TF-IDF feature extraction: {e}")


#     output_file_path = file_path.replace('.csv', '_preprocessed_subset_notnull_removed columns_feature_extraction_250000rows.csv')
#     df.to_csv(output_file_path, index=False)
#     print(f"Preprocessed dataset saved successfully at {output_file_path}")

#     print("Preprocessing complete. The script is ready for machine learning modeling.")
# except FileNotFoundError as e:
#     print(f"File not found error: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")


############################################################  TD-IDF ENCODING   ############################################################################################################

# import os

# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer

# def preprocess_text(series):
#     """Preprocess a Pandas series by filling NaNs and converting lists to strings."""
#     return series.fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

# def process_text_features_in_chunks(series, chunk_size=1000):
#     """Process text features in chunks to manage memory usage."""
#     tfidf_vectorizer = TfidfVectorizer()
#     chunks = (series.iloc[i:i + chunk_size] for i in range(0, len(series), chunk_size))
#     vectorized_data = []
#     for chunk in chunks:
#         vectorized_chunk = tfidf_vectorizer.fit_transform(chunk)
#         df_chunk = pd.DataFrame.sparse.from_spmatrix(vectorized_chunk, columns=tfidf_vectorizer.get_feature_names_out())
#         vectorized_data.append(df_chunk)
#     return pd.concat(vectorized_data, axis=0)

# def print_memory_usage():
#     """Print the current memory usage."""
#     import os, psutil
#     process = psutil.Process(os.getpid())
#     print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# # Load the CSV file with proper handling for mixed data types
# df = pd.read_csv('2023.csv', low_memory=False)

# # List of text columns to process
# text_columns = [
#     'detected_language', 'description.text', 'must_have.skills', 
#     'must_have.languages', 'must_have.work_experiences', 'must_have.education', 
#     'must_have.education_level', 'nice_to_have.skills', 'nice_to_have.languages',
#     'nice_to_have.work_experiences', 'nice_to_have.education', 
#     'nice_to_have.education_level', 'workplace_address.city', 'working_hours_type.label',
# ]

# print("Extracting features...")
# # Concatenate text from multiple columns into a single text column
# df['combined_text'] = df[text_columns].apply(preprocess_text).agg(' '.join, axis=1)

# # Apply TF-IDF to the combined text
# tfidf_df = process_text_features_in_chunks(df['combined_text'])
# df = pd.concat([df, tfidf_df], axis=1)

# print("TF-IDF features extracted and added to DataFrame.")
# print_memory_usage()

# # Save the DataFrame to a new CSV file
# df.to_csv('file_for_text_analysis.csv', index=False)
# print("DataFrame saved to CSV.")








# import os
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# import numpy as np
# import psutil

# def optimize_data_types(df):
#     float_cols = df.select_dtypes(include=['float']).columns
#     df[float_cols] = df[float_cols].astype('float32')
#     int_cols = df.select_dtypes(include=['int', 'integer']).columns
#     df[int_cols] = df[int_cols].astype('int32')
#     return df

# def process_text_features_in_chunks(text_series, chunk_size=5000):
#     vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', dtype=np.float32)
#     chunks = [text_series.iloc[i:i + chunk_size] for i in range(0, len(text_series), chunk_size)]
#     features_list = []
#     for chunk in chunks:
#         chunk = chunk.fillna('')  # Ensure no NaN values
#         features = vectorizer.fit_transform(chunk)
#         features_df = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())
#         features_list.append(features_df)
#     return pd.concat(features_list, ignore_index=True)

# def print_memory_usage():
#     print('Current memory usage:', psutil.virtual_memory())

# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())

# try:
#     csv_file_name = '2023.csv'
#     print("Loading data from .csv file...")
#     df = pd.read_csv(csv_file_name, low_memory=False)
#     df = optimize_data_types(df)
#     print("Data loaded and optimized.")
#     print_memory_usage()

#     print("Cleaning data...")
#     df.dropna(axis=1, how='all', inplace=True)
#     print("Removed columns with only null values.")

#     print("Extracting features...")
#     if 'description.text' in df.columns:
#         df['description.text'] = df['description.text'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
#         tfidf_df = process_text_features_in_chunks(df['description.text'])
#         df = pd.concat([df, tfidf_df], axis=1)
#         print("TF-IDF features extracted and added to DataFrame.")
#     print_memory_usage()


    # for column in ['occupation_field', 'industry', 'detected_language', 'driving_license_required', 'remote_work']:
    #     if column in df.columns:
    #         df = pd.get_dummies(df, columns=[column], dummy_na=True)
    #         print(f"Categorical variable '{column}' one-hot encoded.")

    # if 'must_have.skills' in df.columns:
    #     df['must_have.skills'] = df['must_have.skills'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    #     skills_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    #     skills_features = skills_vectorizer.fit_transform(df['must_have.skills'])
    #     skills_df = pd.DataFrame(skills_features.toarray(), columns=skills_vectorizer.get_feature_names_out())
    #     df = pd.concat([df, skills_df], axis=1)
    #     print("Skills features vectorized and added to DataFrame.")

    # if 'must_have.education_level' in df.columns:
    #     edu_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    #     education_features = edu_vectorizer.fit_transform(df['must_have.education_level'])
    #     education_df = pd.DataFrame(education_features.toarray(), columns=edu_vectorizer.get_feature_names_out())
    #     df = pd.concat([df, education_df], axis=1)
    #     print("Education level features vectorized and added to DataFrame.")

    # print("Splitting data into training and test sets...")
    # if 'is_tech' not in df.columns:
    #     tech_keywords = ['tech', 'software', 'developer', 'IT', 'programming', 'engineer', 'programmering', 'IT-arkitekt', 'mjukvaruutveckling', 'ingenjör']
    #     df['is_tech'] = df['description.text'].apply(lambda x: 1 if any(keyword in x.lower() for keyword in tech_keywords) else 0)
    #     print("'is_tech' column created based on 'description.text'.")

    # if 'is_tech' in df.columns:
    #     X = df.drop('is_tech', axis=1)
    #     y = df['is_tech']
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #     print("Data split complete.")

#     output_file_path = csv_file_name.replace('.csv', '_cleaned_description_dataframe.csv')
#     print(f"Saving the preprocessed dataset to {output_file_path}...")
#     df.to_csv(output_file_path, index=False)
#     print("Preprocessed dataset saved successfully.")

# except Exception as e:
#     print(f"An error occurred: {e}")



























######################################## DATA CLEANING AND FEATURE EXTTRACTION ################################################################################


# import os

# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Dataset Job Listings_Test')
# print("New Working Directory:", os.getcwd())

# try:
#     import pandas as pd
#     import json
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import StandardScaler
#     from scipy.sparse import hstack

#     # Specify the JSONL file name directly
#     jsonl_file_name = '2023.enriched.jsonl'  # Change 'data.jsonl' to your actual JSONL file name

#     # Load and process the .jsonl file line by line to handle large files efficiently
#     print("Loading data from .jsonl file...")
#     data = []
#     counter = 0
#     with open(jsonl_file_name, 'r', encoding='utf-8') as file:
#         for line in file:
#             data.append(json.loads(line))
#             counter += 1
#             if counter % 10000 == 0:
#                 print(f"Processed {counter} lines...")
#     print(f"Data loaded successfully. Total lines processed: {counter}")

#     # Normalizing JSON data into a flat DataFrame
#     df = pd.json_normalize(data)
#     print("Data normalized into DataFrame.")

#     # Clean the Data
#     print("Cleaning data...")
#     # Remove columns where all values are null
#     df.dropna(axis=1, how='all', inplace=True)
#     print("Removed columns with only null values.")
#     print("Data cleaned.")

#     # Feature Extraction
#     print("Extracting features...")
#     # Convert 'description.text' to TF-IDF features; replace 'description.text' with your column of interest
#     if 'description.text' in df.columns:
#         df['description.text'] = df['description.text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
#         tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
#         tfidf_features = tfidf_vectorizer.fit_transform(df['description.text'])
#         tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
#         df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
#         print("TF-IDF features extracted and added to DataFrame.")
#     else:
#         print("'description.text' column not found, skipping TF-IDF feature extraction.")

#     # One-hot encode selected categorical variables; adjust column names as needed
#     for column in ['occupation_field', 'industry', 'detected_language', 'driving_license_required', 'remote_work']:
#         if column in df.columns:
#             df = pd.get_dummies(df, columns=[column], dummy_na=True)  # Consider NA as a category
#             print(f"Categorical variable '{column}' one-hot encoded.")
#         else:
#             print(f"'{column}' column not found. Skipping one-hot encoding for '{column}'.")

#     # Text vectorization for skills and education levels
#     if 'must_have.skills' in df.columns:
#         df['must_have.skills'] = df['must_have.skills'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
#         skills_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
#         skills_features = skills_vectorizer.fit_transform(df['must_have.skills'])
#         skills_df = pd.DataFrame(skills_features.toarray(), columns=skills_vectorizer.get_feature_names_out())
#         df = pd.concat([df, skills_df], axis=1)
#         print("Skills features vectorized and added to DataFrame.")
#     if 'must_have.education_level' in df.columns:
#         edu_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
#         education_features = edu_vectorizer.fit_transform(df['must_have.education_level'])
#         education_df = pd.DataFrame(education_features.toarray(), columns=edu_vectorizer.get_feature_names_out())
#         df = pd.concat([df, education_df], axis=1)
#         print("Education level features vectorized and added to DataFrame.")

#     # Preparing data for machine learning
#     print("Splitting data into training and test sets...")
#     if 'is_tech' not in df.columns:
#         # Creating 'is_tech' column based on keywords in 'description.text'
#         tech_keywords = ['tech', 'software', 'developer', 'IT', 'programming', 'engineer', 'programmering', 'IT-arkitekt', 'mjukvaruutveckling', 'ingenjör']
#         df['is_tech'] = df['description.text'].apply(lambda x: 1 if any(keyword in x.lower() for keyword in tech_keywords) else 0)
#         print("'is_tech' column created based on 'description.text'.")

#     if 'is_tech' in df.columns:
#         X = df.drop('is_tech', axis=1)  # Drop the target variable to isolate features; adjust as necessary
#         y = df['is_tech']  # Assume 'is_tech' is the target variable
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         print("Data split complete.")
#     else:
#         print("Data split skipped due to missing 'is_tech' column.")

#     # Save the preprocessed dataset for future use
#     output_file_path = jsonl_file_name.replace('.jsonl', '_preprocessed1.csv')
#     print(f"Saving the preprocessed dataset to {output_file_path}...")
#     df.to_csv(output_file_path, index=False)
#     print("Preprocessed dataset saved successfully.")

#     print("Preprocessing complete. The script is ready for machine learning modeling.")
# except Exception as e:
#     print(f"An error occurred: {e}")






############################################  GET UNIQUE COLUMNS NAMES   #########################################################################################################################


# import os
# import pandas as pd

# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Dataset Job Listings_Test')
# print("New Working Directory:", os.getcwd())

# def unique_columns_from_jsonl(file_path):
#     chunk_size = 1000  # Adjust this size based on your memory capacity
#     unique_columns = set()

#     for chunk in pd.read_json(file_path, lines=True, chunksize=chunk_size):
#         unique_columns.update(chunk.columns)

#     return list(unique_columns)

# file_path = '2023.enriched.jsonl'
# unique_columns = unique_columns_from_jsonl(file_path)
# print("Unique columns in the dataset:", unique_columns)



## ['workplace_address', 'occupation', 'removed', 'access_to_own_car', 'number_of_vacancies', 'experience_required', 'employer', 'headline', 'open_for_all', 
# 'external_id', 'description', 'keywords', 'application_deadline', 'employment_type', 'id', 'must_have', 'trainee', 'application_details', 'occupation_group', 
# 'webpage_url', 'remote_work', 'application_contacts', 'source_type', 'franchise', 'access', 'larling', 'occupation_field', 'driving_license_required', 'timestamp', 
# 'removed_date', 'publication_date', 'logo_url', 'salary_type', 'last_publication_date', 'working_hours_type', 'driving_license', 'detected_language', 
# 'hire_work_place', 'nice_to_have', 'scope_of_work', 'salary_description', 'original_id', 'duration']




# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# This code picks out 10 most common headlines related to 'sustainability' or 'hållbarhet'


# import pandas as pd
# import re  # For regular expression operations


# import os

# correct_path = 'C:\\Systemvetenskap\\Python\\Molkor First Fork\\MPTJB\\Data Analysis'

# # Change the current working directory to the script's folder
# os.chdir(correct_path)

# def load_and_filter_data(csv_path):
#     print("Loading data from CSV file...")
#     data = pd.read_csv(csv_path, low_memory=False)
#     print("Data loaded successfully.")
    
#     # Filter data to find mentions of 'sustainability' or 'hållbarhet' in the 'description.text' column
#     filtered_data = data[data['description.text'].str.contains('sustainability|hållbarhet', case=False, na=False)]
#     return filtered_data

# def find_common_headlines(filtered_data):
#     if not filtered_data.empty:
#         # Filter out headlines that are likely to be dates (pattern: YYYY-MM-DDTHH:MM:SS)
#         filtered_data = filtered_data[~filtered_data['headline'].str.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$')]
        
#         # Count occurrences of each headline
#         headline_counts = filtered_data['headline'].value_counts()
#         return headline_counts.head(10)  # Return top 10 most common headlines
#     else:
#         print("No entries mention 'sustainability' or 'hållbarhet'.")
#         return pd.Series()  # Return empty series if no data

# def main():
#     csv_path = '2023.csv'
#     filtered_data = load_and_filter_data(csv_path)
#     common_headlines = find_common_headlines(filtered_data)
#     print("Most common headlines for entries related to sustainability or hållbarhet (excluding dates):")
#     print(common_headlines)

# if __name__ == "__main__":
#     main()




## OUTPUT:

# Loading data from CSV file...
# Data loaded successfully.
# Most common headlines for entries related to sustainability or hållbarhet (excluding dates):
# Servicerådgivare sökes till Mobility Motors - Sätra         81
# Redovisningsekonom till Johanson Design i Markaryd          78
# 24500:- plus provision                                      69
# Kreativ och driven säljare till Bisevo                      66
# 🌟 Få Ditt Karriärlyft hos ABC SALES för Göta Energi! 🌟      60
# Systemutvecklare till innovativt IT-företag                 49
# Senior cyber security engineer till innovativt techbolag    49
# Elektronikkonstruktör till tekniskt konsultföretag          49
# E-mobility lead till kreativt techbolag                     49
# Embedded utvecklare till kreativt techbolag                 49
# Name: headline, dtype: int64









# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------



## How top 5 keywords for Testers different from UX designers?

# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# import pandas as pd
# from nltk.corpus import stopwords as nltk_stopwords
# from nltk.tokenize import word_tokenize
# from nltk.probability import FreqDist
# from stop_words import get_stop_words

# import os

# # The correct path based on the screenshot you've shared
# correct_path = 'C:\\Systemvetenskap\\Python\\Molkor First Fork\\MPTJB\\Data Analysis'

# # Change the current working directory to the script's folder
# os.chdir(correct_path)

# # Your code to load the CSV should follow here


# def load_and_filter_data(csv_path, keywords):
#     df = pd.read_csv(csv_path)
#     pattern = '|'.join([f"\\b{k}\\b" for k in keywords])  # Regex for whole word match
#     return df[df['headline'].str.contains(pattern, case=False, na=False)]


# def extract_top_keywords(texts, top_n=5):
#     english_stopwords = set(nltk_stopwords.words('english'))
#     swedish_stopwords = set(get_stop_words('swedish'))
#     all_stopwords = english_stopwords.union(swedish_stopwords)

#     words = word_tokenize(texts.str.cat(sep=' ').lower())
#     filtered_words = [word for word in words if word.isalnum() and word not in all_stopwords]
#     freq_dist = FreqDist(filtered_words)
#     return [word for word, _ in freq_dist.most_common(top_n)]


# def main(csv_file):
#     tester_keywords = ['tester', 'QA', 'test engineer']
#     ux_keywords = ['UX designer', 'user experience', 'UI/UX']

#     tester_df = load_and_filter_data(csv_file, tester_keywords)
#     ux_df = load_and_filter_data(csv_file, ux_keywords)

#     tester_top_keywords = extract_top_keywords(tester_df['description.text'])
#     ux_top_keywords = extract_top_keywords(ux_df['description.text'])

#     print("Top 5 Keywords for Testers:", tester_top_keywords)
#     print("Top 5 Keywords for UX Designers:", ux_top_keywords)


# main('2023.csv')



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## In which context do the keywords “hållbarhet” vs. “sustainability” appear? 

# import csv
# import re
# import os

# # Print the current working directory
# print("Current Working Directory:", os.getcwd())

# # Change the working directory
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')

# # Verify the change
# print("New Working Directory:", os.getcwd())

# # Path to the CSV file
# file_path = '2023.csv'

# # Compiled regular expressions for each phrase for efficiency
# pattern_hallbarhet = re.compile(r'(\b\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s)?hållbarhet(\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\b)?', re.IGNORECASE)
# pattern_sustainability = re.compile(r'(\b\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s)?sustainability(\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\b)?', re.IGNORECASE)

# # Function to process rows and capture context
# def process_rows(reader):
#     results_hallbarhet = []
#     results_sustainability = []
#     for row in reader:
#         # Process each regex and store results up to 10 instances
#         for pattern, result_list, label in [
#             (pattern_hallbarhet, results_hallbarhet, "Hållbarhet"),
#             (pattern_sustainability, results_sustainability, "Sustainability")
#         ]:
#             if len(result_list) < 10:
#                 for match in pattern.finditer(row['description.text']):
#                     before = match.group(1) or ""
#                     after = match.group(2) or ""
#                     result_list.append(f"Context ({label}): {before.strip()} {label.lower()} {after.strip()}")
#                     if len(result_list) == 10:
#                         break

#             if all(len(lst) == 10 for lst in [results_hallbarhet, results_sustainability]):
#                 break

#     return results_hallbarhet, results_sustainability

# # Main function to open the file and process the CSV
# def main():
#     with open(file_path, newline='', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         results_hallbarhet, results_sustainability = process_rows(reader)

#     # Print results for each term
#     print("\nHållbarhet Contexts:")
#     for result in results_hallbarhet:
#         print(result)
#     print("\nSustainability Contexts:")
#     for result in results_sustainability:
#         print(result)

# if __name__ == "__main__":
#     main()




# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # How homogeneous are descriptions in a small town (Marstrand) and a larger town (Borås)?



# import os
# import pandas as pd
# import numpy as np
# import csv
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity




# # Print the current working directory
# print("Current Working Directory:", os.getcwd())

# # Change the working directory
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')

# # Verify the change
# print("New Working Directory:", os.getcwd())

# # Path to the CSV file
# file_path = '2023.csv'

# # Load the data
# df = pd.read_csv('2023.csv')

# # Filter rows based on municipalities
# filtered_df = df[df['workplace_address.municipality'].isin(['Borås', 'Kungälv'])]

# # Process each municipality separately
# municipalities = ['Borås', 'Kungälv']
# results = {}

# for city in municipalities:
#     # Subset data for the city
#     subset = filtered_df[filtered_df['workplace_address.municipality'] == city]
    
#     # Vectorize the text descriptions
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(subset['description.text'])
    
#     # Calculate cosine similarity matrix
#     cos_sim_matrix = cosine_similarity(tfidf_matrix)
    
#     # Calculate average cosine similarity for the city
#     # We avoid self-comparison by subtracting 1 from the count and setting the diagonal to 0
#     n = cos_sim_matrix.shape[0]
#     if n > 1:
#         np.fill_diagonal(cos_sim_matrix, 0)
#         avg_cos_sim = cos_sim_matrix.sum() / (n * (n - 1))
#     else:
#         avg_cos_sim = None  # Not enough data to calculate similarity

#     results[city] = avg_cos_sim

# # Output the results
# print("Average Cosine Similarity by Municipality:")
# for city, sim in results.items():
#     print(f"{city}: {sim if sim is not None else 'Insufficient data'}")




# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


##########################   DEPRECATED ANALYSIS CODE ####################################################################################################


# import os
# import pandas as pd
# import numpy as np
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from textblob import TextBlob
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Download necessary NLTK data for TextBlob and stopwords
# def download_nltk_data():
#     nltk.download('punkt')
#     nltk.download('averaged_perceptron_tagger')
#     nltk.download('brown')
#     nltk.download('stopwords')

# download_nltk_data()

# # Define custom stopwords
# custom_stopwords = [
#     'specific', 'general', 'terms', 'common', 'examples', 'including', 'etc', 'och', 'att', 'du', 'är', 'vi', 'med','som', 'en', 'för', 'av', 'ett', 'https', 'har'
#     'på', 'kommer'
#     # Add any additional domain-specific stopwords
# ]

# # Get the list of default stopwords for English and combine with custom stopwords
# default_stopwords = stopwords.words('english')  # Adjust language as necessary
# all_stopwords = default_stopwords + custom_stopwords

# # Print the current working directory and change it if necessary
# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())

# # Load data
# file_path = '2023.csv'
# df = pd.read_csv(file_path, low_memory=False)

# # Define sustainability-related terms and filter for mentions of sustainability
# sustainability_terms = ['sustainability', 'sustainable', 'green', 'environmental', 'renewable', 'hållbarhet', 'miljövänlig']
# filter_pattern = '|'.join(sustainability_terms)
# df['mentions_sustainability'] = df['description.text'].str.contains(filter_pattern, case=False, na=False)

# # Filter DataFrame for sustainability mentions and create a definitive copy for modifications
# sustainable_df = df[df['mentions_sustainability']].copy()

# # Process text for analysis, integrating all stopwords
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=all_stopwords)
# tfidf_matrix = tfidf_vectorizer.fit_transform(sustainable_df['description.text'])

# # Apply LDA for topic modeling
# lda = LatentDirichletAllocation(n_components=5, random_state=42)
# topics = lda.fit_transform(tfidf_matrix)

# # Output topics
# feature_names = tfidf_vectorizer.get_feature_names_out()
# for topic_idx, topic in enumerate(lda.components_):
#     print(f"Topic {topic_idx + 1}:")
#     print(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))

# # Sentiment Analysis: use .loc to ensure modifications are reflected in the DataFrame copy
# sustainable_df['sentiment'] = sustainable_df['description.text'].apply(lambda text: TextBlob(text).sentiment.polarity)

# # Displaying results or further analysis here...

# # Analyze frequency of mentions and visualize
# sustainability_count_by_city = df.groupby('workplace_address.municipality')['mentions_sustainability'].sum()
# sns.barplot(x=sustainability_count_by_city.index, y=sustainability_count_by_city.values)
# plt.title('Frequency of Sustainability Mentions by Municipality')
# plt.ylabel('Count')
# plt.xlabel('Municipality')
# plt.xticks(rotation=45)
# plt.show()



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------



########################## DEPRECATED ANALYSIS CODE ####################################################################################################


# import os
# import re
# import pandas as pd
# import numpy as np
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from textblob import TextBlob
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Download necessary NLTK data for TextBlob and stopwords
# def download_nltk_data():
#     nltk.download('punkt')
#     nltk.download('averaged_perceptron_tagger')
#     nltk.download('brown')
#     nltk.download('stopwords')

# download_nltk_data()

# # Define custom stopwords
# custom_stopwords = [
#     'specific', 'general', 'terms', 'common', 'examples', 'including', 'etc',
#     'och', 'att', 'du', 'är', 'vi', 'med', 'som', 'en', 'för', 'av', 'ett',
#     'https', 'har', 'på', 'kommer', 'dig', 'oss', 'till', 'vara', 'samt', 
#     'eftersom', 'vår', 'jobbet', 'vill', 'egen', 'jag', 'work', 'den', 'jobba', 
#     'jobbet', 'within', 'våra','inom', 'det', 'din', 'söker', 'kan', 'om', 'alla', 
#     'kunna', 'då', 'ser', 'din', 'får', 'kan', 'working', 'experience', 'us', 'utan', 
#     'role', 'både', 'arbetar', 'hos', 'eller', 'arbeta', 'tillsammans', 'ditt', 'efter',
# ]

# # Get the list of default stopwords for English and combine with custom stopwords
# default_stopwords = stopwords.words('english')  # Adjust language as necessary
# all_stopwords = default_stopwords + custom_stopwords

# # Print the current working directory and change it if necessary
# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())

# # Load data
# file_path = '2023.csv'
# df = pd.read_csv(file_path, low_memory=False)

# # Define sustainability-related terms and filter for mentions of sustainability
# sustainability_terms = ['sustainability', 'sustainable', 'green', 'environmental', 'renewable', 'hållbarhet', 'miljövänlig']
# filter_pattern = '|'.join(sustainability_terms)
# df['mentions_sustainability'] = df['description.text'].str.contains(filter_pattern, case=False, na=False)

# # Filter DataFrame for sustainability mentions and create a definitive copy for modifications
# sustainable_df = df[df['mentions_sustainability']].copy()

# # Process text for analysis, integrating all stopwords
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=all_stopwords)
# tfidf_matrix = tfidf_vectorizer.fit_transform(sustainable_df['description.text'])

# # Apply LDA for topic modeling
# lda = LatentDirichletAllocation(n_components=5, random_state=42)
# topics = lda.fit_transform(tfidf_matrix)

# # Output topics
# feature_names = tfidf_vectorizer.get_feature_names_out()
# for topic_idx, topic in enumerate(lda.components_):
#     print(f"Topic {topic_idx + 1}:")
#     print(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))

# # Sentiment Analysis: use .loc to ensure modifications are reflected in the DataFrame copy
# sustainable_df['sentiment'] = sustainable_df['description.text'].apply(lambda text: TextBlob(text).sentiment.polarity)

# # Analyze frequency of mentions and visualize
# sustainability_count_by_city = df.groupby('workplace_address.municipality')['mentions_sustainability'].sum()
# sns.barplot(x=sustainability_count_by_city.index, y=sustainability_count_by_city.values)
# plt.title('Frequency of Sustainability Mentions by Municipality')
# plt.ylabel('Count')
# plt.xlabel('Municipality')
# plt.xticks(rotation=45)
# plt.show()

# # Compiled regular expressions for keyword contexts
# pattern_hallbarhet = re.compile(r'(\b\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s)?hållbarhet(\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\b)?', re.IGNORECASE)
# pattern_sustainability = re.compile(r'(\b\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s)?sustainability(\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\b)?', re.IGNORECASE)

# # Extract context for keywords and limit to 10 instances for each
# results_hallbarhet, results_sustainability = [], []
# for description in df['description.text'].dropna():
#     if len(results_hallbarhet) < 10:
#         matches = pattern_hallbarhet.findall(description)
#         results_hallbarhet.extend(matches[:10-len(results_hallbarhet)])
#     if len(results_sustainability) < 10:
#         matches = pattern_sustainability.findall(description)
#         results_sustainability.extend(matches[:10-len(results_sustainability)])
#     if len(results_hallbarhet) >= 10 and len(results_sustainability) >= 10:
#         break

# print("\nHållbarhet Contexts:")
# for result in results_hallbarhet:
#     print(result)
# print("\nSustainability Contexts:")
# for result in results_sustainability:
#     print(result)



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------



########################## DEPRECATED ANALYSIS ########################################################################################################################



# import os
# import pandas as pd
# import numpy as np
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from textblob import TextBlob
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm


# # Download necessary NLTK data for TextBlob and stopwords
# def download_nltk_data():
#     nltk.download('punkt')
#     nltk.download('averaged_perceptron_tagger')
#     nltk.download('brown')
#     nltk.download('stopwords')

# download_nltk_data()

# # Define custom stopwords
# custom_stopwords = [
#     'specific', 'general', 'terms', 'common', 'examples', 'including', 'etc',
#     'och', 'att', 'du', 'är', 'vi', 'med', 'som', 'en', 'för', 'av', 'ett',
#     'https', 'har', 'på', 'kommer', 'dig', 'oss', 'till', 'vara', 'samt', 
#     'eftersom', 'vår', 'jobbet', 'vill', 'egen', 'jag', 'work', 'den', 'jobba', 
#     'jobbet', 'within', 'våra', 'inom', 'det', 'din', 'söker', 'kan', 'om', 'alla', 
#     'kunna', 'då', 'ser', 'din', 'får', 'kan', 'working', 'experience', 'us', 'utan', 
#     'role', 'både', 'arbetar', 'hos', 'eller', 'arbeta', 'tillsammans', 'ditt', 'efter',
#     'arbete', 'tiden', 'fått', 'air', 'schema', 'god', 'åker', 'inget', 'finns', 'mycket', 
#     'något', 'där', 'genom', 'så', 'från', 'dina', 'uppdrag', 'annat', 'ska', 'vid', 'stad',
#     'även', 'rollen', 'mer', 'erbjuder', 'de', 'arbetet', 'medarbetare', 'fast', 'del', 'vårt',
#     'inte', 'lön', 'plus', 'solid', 'ta', 'andra', 'olika', 'se', 'sky', 'västra', 'ha', 'rätt',
#     'clear', 'new', 'kr', 'företaget', 'företag', 'part', 'stockholm', 'skåne', 'ansökan', 'person', 
#     'tjänster', 'nya', 'tjänsten', 'löpande', 'arbetsuppgifter', 'anställning', 'bemannia'
# ]

# # Get the list of default stopwords for English and combine with custom stopwords
# default_stopwords = stopwords.words('english')
# all_stopwords = default_stopwords + custom_stopwords

# # Print the current working directory and change it if necessary
# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())

# # Load data
# file_path = 'description-column.csv'
# df = pd.read_csv(file_path, low_memory=False)

# # Define sustainability-related terms and filter for mentions of sustainability
# sustainability_terms = [
#     'sustainability', 'sustainable', 'green', 'environmental', 'renewable', 'hållbarhet', 'miljövänlig', 'clean energy', 'green economy',
#     'sustainable tech', 'technology', 'biodegradable', 'sustainable production', 'green technology', 'carbon neutral', 'circular economy', 'hållbar',
#     'hållbar teknologi', 'hållbar produktion', 'rekrytering', 'recruiter', 'recruiting'
#     ]
# filter_pattern = '|'.join(sustainability_terms)
# df['mentions_sustainability'] = df['description.text'].str.contains(filter_pattern, case=False, na=False)

# # Filter DataFrame for sustainability mentions and create a definitive copy for modifications
# sustainable_df = df[df['mentions_sustainability']].copy()

# # Process text for analysis, integrating all stopwords
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=all_stopwords)
# tfidf_matrix = tfidf_vectorizer.fit_transform(sustainable_df['description.text'])

# # Apply LDA for topic modeling
# lda = LatentDirichletAllocation(n_components=5, random_state=42)
# topics = lda.fit_transform(tfidf_matrix)

# # Output topics
# feature_names = tfidf_vectorizer.get_feature_names_out()
# for topic_idx, topic in enumerate(lda.components_):
#     print(f"Topic {topic_idx + 1}:")
#     print(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))


# # Sentiment Analysis
# print("Analyzing sentiments...")
# sustainable_df['sentiment'] = sustainable_df['description.text'].apply(lambda text: TextBlob(text).sentiment.polarity)

# # Print out the summary statistics of the sentiment scores
# print("Sentiment Analysis Summary:")
# print(sustainable_df['sentiment'].describe())

# # Visualizing sentiment distribution
# plt.figure(figsize=(10, 6))
# plt.hist(sustainable_df['sentiment'], bins=30, color='blue', alpha=0.7)
# plt.title('Distribution of Sentiment Scores')
# plt.xlabel('Sentiment Polarity')
# plt.ylabel('Frequency')
# plt.show()




# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# import os
# import pandas as pd


# # Print the current working directory and change it if necessary
# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())

# # # Load data
# # file_path = '2023.csv'
# # df = pd.read_csv(file_path, low_memory=False)


# # Load the dataset
# df = pd.read_csv('2023.csv', low_memory=False)

# # Inspect data types
# print(df.dtypes)

# # Inspect unique values for potential categorical columns
# for col in df.columns:
#     print(f"{col}: {df[col].unique()}")  # Adjust this to df[col].nunique() if there are too many unique values


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
######################################## DESCRIPTION COLUMN ONLY ################################################################################

# import os
# import pandas as pd


# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())

# # Load the CSV file while addressing dtype warnings
# df = pd.read_csv('2023.csv', low_memory=False)

# # Check columns to ensure the correct column names are used
# print(df.columns)

# # Assuming 'Description' is the correct column name
# try:
#     subset = df[['description.text']]
# except KeyError:
#     print("Error: 'description.text' column not found in the DataFrame.")

# # Proceed with further analysis or save the subset as needed

# # Save subset to a new CSV
# subset.to_csv('description-column.csv', index=False)



######################################## CHECK DATA TYPES ################################################################################

# import pandas as pd
# import os


# # Print the current working directory and change it if necessary
# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())

# # Load the dataset
# df = pd.read_csv('2023.csv')

# # Display the first few rows of the dataframe
# print(df.head())

# # Display data types of each column
# print(df.dtypes)

# # Further inspection to understand if numerical data might be categorical
# for column in df.columns:
#     # Check if the column is of a numeric type
#     if pd.api.types.is_numeric_dtype(df[column]):
#         # Print unique values if they are very few
#         if df[column].nunique() < 10:  # you can adjust this threshold
#             print(f"{column} unique values: {df[column].unique()}")

