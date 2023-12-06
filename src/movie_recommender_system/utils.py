import os
import pickle
import numpy as np
import math
from collections import Counter

def movie_ids_to_movie_titles(movie_ids, movies_df):
    titles = []
    for movie_id in movie_ids:
        t = movies_df[movies_df['movieId'] == movie_id]['title']
        titles.append(t.values[0] if len(t.values) > 0 else "Not found")
        
    return titles

def search_and_read_pickle(directory, file_name):
    # Get a list of all files in the directory
    if os.path.exists(directory):
            print(f'Directory exists: {directory}')
    else:
        print(f'Directory does not exist: {directory}')
        return None

    files = os.listdir(directory)

    if file_name in files:
        file_path = os.path.join(directory, file_name)

        # Read the dictionary from the pickle file
        with open(file_path, 'rb') as file:
            loaded_dict = pickle.load(file)

        print(f'Successfully loaded dictionary from {file_path}')
        return loaded_dict
    else:
        print(f'File {file_name} not found in {directory}')
        return None

class CustomTfidfVectorizer:
    def __init__(self):
        self.documents_count = 0
        self.word_document_count = {}
        self.tfidf_matrix = None

    def fit_transform(self, documents):
        self.documents_count = len(documents)

        # Count document frequency for each term
        for document in documents:
            terms = set(document.split())
            for term in terms:
                self.word_document_count[term] = self.word_document_count.get(term, 0) + 1

        # Create a vocabulary based on the unique terms in all documents
        vocabulary = list(self.word_document_count.keys())
        vocabulary.sort()

        # Create a matrix to store the TF-IDF values
        self.tfidf_matrix = np.zeros((self.documents_count, len(vocabulary)))

        # Calculate TF-IDF matrix
        for i, document in enumerate(documents):
            tfidf_vector = self.calculate_tfidf_vector(document, vocabulary)
            self.tfidf_matrix[i, :] = tfidf_vector

        return self.tfidf_matrix

    def calculate_tfidf_vector(self, document, vocabulary):
        terms = document.split()
        tf_vector = dict(Counter(terms))
        tfidf_vector = np.zeros(len(vocabulary))

        for j, term in enumerate(vocabulary):
            tf = tf_vector.get(term, 0)
            idf = np.log(self.documents_count / (1 + self.word_document_count.get(term, 0)))
            tfidf_vector[j] = tf * idf

        return tfidf_vector