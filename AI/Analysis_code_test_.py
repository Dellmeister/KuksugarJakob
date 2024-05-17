###FÅR SLUT PÅ MINNE

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


#------------------------------------------------------------------------------------------------


import os
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import psutil
import dask.dataframe as dd
from scipy.sparse import vstack

# Helper Functions
def optimize_data_types(df):
    # Optimize float and int types for memory usage
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].astype('float32')
    int_cols = df.select_dtypes(include=['int', 'integer']).columns
    df[int_cols] = df[int_cols].astype('int32')
    return df

def process_text_features_in_chunks(text_series, chunk_size=5000):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', dtype=np.float32)
    chunks = [text_series.iloc[i:i + chunk_size] for i in range(0, len(text_series), chunk_size)]
    sparse_features = [vectorizer.fit_transform(chunk) for chunk in chunks]
    return vstack(sparse_features)

def convert_to_dask(df):
    dask_df = dd.from_pandas(df, npartitions=10)
    return dask_df

def print_memory_usage():
    print('Current memory usage:', psutil.virtual_memory())

# # Script Starts Here
# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Dataset Job Listings_Test')
# print("New Working Directory:", os.getcwd())

try:
    jsonl_file_name = '2023.enriched.jsonl'
    data = []
    with open(jsonl_file_name, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
            if len(data) % 10000 == 0:
                print(f"Processed {len(data)} lines...")

    df = pd.json_normalize(data)
    df = optimize_data_types(df)
    print("Data normalized into DataFrame.")
    print_memory_usage()

    print("Cleaning data...")
    df.dropna(axis=1, how='all', inplace=True)
    print("Removed columns with only null values.")
    print("Data cleaned.")

    print("Extracting features...")
    if 'description.text' in df.columns:
        df['description.text'] = df['description.text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        sparse_matrix = process_text_features_in_chunks(df['description.text'])
        tfidf_df = pd.DataFrame(sparse_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
        print("TF-IDF features extracted and added to DataFrame.")
    print_memory_usage()

    for column in ['occupation_field', 'industry', 'detected_language', 'driving_license_required', 'remote_work']:
        if column in df.columns:
            df = pd.get_dummies(df, columns=[column], dummy_na=True)
            print(f"Categorical variable '{column}' one-hot encoded.")

    print("Splitting data into training and test sets...")
    if 'is_tech' not in df.columns:
        tech_keywords = ['tech', 'software', 'developer', 'IT', 'programming', 'engineer', 'programmering', 'IT-arkitekt', 'mjukvaruutveckling', 'ingenjör']
        df['is_tech'] = df['description.text'].apply(lambda x: 1 if any(keyword in x.lower() for keyword in tech_keywords) else 0)
        print("'is_tech' column created based on 'description.text'.")

    if 'is_tech' in df.columns:
        X = df.drop('is_tech', axis=1)
        y = df['is_tech']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split complete.")
    else:
        print("Data split skipped due to missing 'is_tech' column.")

    output_file_path = jsonl_file_name.replace('.jsonl', '_preprocessed1.csv')
    print(f"Saving the preprocessed dataset to {output_file_path}...")
    df.to_csv(output_file_path, index=False)
    print("Preprocessed dataset saved successfully.")
    print("Preprocessing complete. The script is ready for machine learning modeling.")

except Exception as e:
    print(f"An error occurred: {e}")
