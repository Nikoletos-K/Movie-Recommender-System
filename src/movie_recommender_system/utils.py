import os
import pickle
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

class TFIDF:
    def __init__(self, documents):
        self.documents = documents
        self.tf = []
        self.idf = {}
        self.tfidf_matrix = []

        self._calculate_tf()
        self._calculate_idf()
        self._calculate_tfidf_matrix()

    def _calculate_tf(self):
        for doc in self.documents:
            term_frequency = Counter(doc)
            total_terms = len(doc)
            tf_doc = {term: count / total_terms for term, count in term_frequency.items()}
            self.tf.append(tf_doc)

    def _calculate_idf(self):
        total_documents = len(self.documents)
        for doc in self.documents:
            for term in set(doc):
                self.idf[term] = self.idf.get(term, 0) + 1

        for term, doc_count in self.idf.items():
            self.idf[term] = math.log(total_documents / (1 + doc_count))

    def _calculate_tfidf_matrix(self):
        for tf_doc in self.tf:
            tfidf_doc = {term: tf * self.idf[term] for term, tf in tf_doc.items()}
            self.tfidf_matrix.append(tfidf_doc)

    def get_tfidf_matrix(self):
        return self.tfidf_matrix