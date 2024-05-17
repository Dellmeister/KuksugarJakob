import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Sample code for clustering words in job descriptions

# Load the data
jobtech_dataset = pd.read_csv('2023_vasili.csv', encoding='utf-8', low_memory=False)

# Text preprocessing
jobtech_dataset['description_processed'] = jobtech_dataset['description.text'].str.replace('[^\w\s]', '', regex=True).str.lower()

# Fill NaN values with a placeholder string (e.g., 'missing')
jobtech_dataset['description_processed'] = jobtech_dataset['description_processed'].fillna('missing')

# Feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(jobtech_dataset['description_processed'])

# Clustering words using K-Means
kmeans = KMeans(n_clusters=20, random_state=42)
kmeans.fit(X)

# Get the top terms per cluster
def get_top_terms_per_cluster(centroids, feature_names, n_terms=10):
    top_terms = {}
    for i, c in enumerate(centroids):
        # Get the indices of the sorted array and reverse it
        sorted_indices = np.argsort(c)[::-1]
        top_terms[i] = [feature_names[idx] for idx in sorted_indices[:n_terms]]
    return top_terms

feature_names = vectorizer.get_feature_names_out()
centroids = kmeans.cluster_centers_
top_terms_per_cluster = get_top_terms_per_cluster(centroids, feature_names)

# Output the top terms for each cluster
for cluster, terms in top_terms_per_cluster.items():
    print(f"Cluster {cluster}: {', '.join(terms)}")
