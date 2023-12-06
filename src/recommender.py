import pandas as pd
import os
import pandas as pd
import time

import argparse

from movie_recommender_system.algorithms import (
    UserUserALGO,
    ItemItemALGO,
    TagBasedALGO,
    ContentBasedALGO,
    HybridALGO
)
from movie_recommender_system.utils import movie_ids_to_movie_titles, search_and_read_pickle

# READ CLI INPUTS
parser = argparse.ArgumentParser(description='Movie Recommender System')
parser.add_argument('-d', type=str, required=True)
parser.add_argument('-n', type=int, required=True)
parser.add_argument('-s', type=str, required=True)
parser.add_argument('-a', type=str, required=True)
parser.add_argument('-i', type=str, required=True)
parser.add_argument('-preprocess', type=str, required=False)
parser.add_argument('-nrows', type=str, required=False)
args = parser.parse_args()
data_dir = args.d
num_recommendations = args.n
similarity_metric = args.s
algorithm = args.a
input_i = args.i
preprocess = args.preprocess if args.preprocess else False
nrows = int(args.nrows) if args.nrows else None

print("\n\nMovie Recommender System")
print("-----------------------")
print("Data directory: ", data_dir)
print("Number of recommendations: ", num_recommendations)
print("Similarity metric: ", similarity_metric)
print("Algorithm: ", algorithm)
print("Input: ", input_i)

# DATA READING
dfs = {}

dataset = 'mllatest'
if 'ml-latest' in data_dir:
    data_folder = data_dir
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    print("\nReading CSV files from: ", data_folder)
    print("CSV files found: ", csv_files)
    dfs = {}
    csv_files = ['genome-scores.csv', 'genome-tags.csv', 'links.csv', 'movies.csv', 'ratings.csv', 'tags.csv']
    for file in csv_files:
        if file == 'genome-scores.csv' or file == 'genome-tags.csv' or file == 'links.csv':
            continue
        print("\nReading: ", file)
        if nrows:
            df = pd.read_csv(os.path.join(data_folder, file), nrows=nrows)
        else:
            df = pd.read_csv(os.path.join(data_folder, file))
        file_name = file.split('.')[0]
        print("Saving as: ", file_name)
        print("Size: ", df.shape)
        print("Columns: ", df.columns)
        dfs[file_name] = df
elif 'ml-100k' in data_dir:
    dataset = 'ml100k'
    data_folder = data_dir
    print("\nReading: u.data")
    dfs['ratings'] = pd.read_csv(os.path.join(data_folder, 'u.data'), sep='\t', header=None, names=['userId','movieId','rating','timestamp'])
    print(dfs['ratings'].head())

    print("Reading: u.item")
    dfs['movies'] = pd.read_csv(os.path.join(data_folder, 'u.item'), sep='|', header=None, names=['movieId','title'], usecols=[0,1], encoding='latin-1')
    print(dfs['movies'].head())

# Calculate execution time
start_time = time.time()

if algorithm == "user":

    if preprocess:
        directory_path = './preprocessed_data_100k' if 'ml-100k' in data_dir else './preprocessed_data_latest'
        file_to_search = dataset+'_user_' + str(input_i) +'_'+ similarity_metric  + '.pkl'

        loaded_dictionary = search_and_read_pickle(directory_path, file_to_search)

        if loaded_dictionary:
            print('Loaded Dictionary!')
        mapping = loaded_dictionary[int(input_i)]
    else:
        mapping = None

    algo = UserUserALGO()
    movie_ids = algo.fit(dfs['ratings'], input_i, similarity_metric, num_recommendations, preprocess=False, n_most_similar_mapping=mapping)
elif algorithm == "item":
    if preprocess:
        directory_path = './preprocessed_data_100k' if 'ml-100k' in data_dir else './preprocessed_data_latest'
        file_to_search = dataset+'_item_' + str(input_i) +'_'+ similarity_metric  + '.pkl'

        loaded_dictionary = search_and_read_pickle(directory_path, file_to_search)

        if loaded_dictionary:
            print('Loaded Dictionary!')
        mapping = loaded_dictionary[int(input_i)]
    else:
        mapping = None

    algo = ItemItemALGO()
    movie_ids = algo.fit(dfs['ratings'], input_i, similarity_metric, num_recommendations, preprocess=False, n_most_similar_mapping=mapping)
elif algorithm == "tag":
    algo = TagBasedALGO()
    movie_ids = algo.fit(dfs['tags'], input_i, similarity_metric, num_recommendations)
elif algorithm == "title":
    content_based_algo = ContentBasedALGO()
    movie_ids = content_based_algo.fit(dfs['movies'], input_i, similarity_metric, num_recommendations)
    
elif algorithm == "hybrid":

    num_of_reccomendations_returned = -1
    if preprocess:
        directory_path = './preprocessed_data_100k' if 'ml-100k' in data_dir else './preprocessed_data_latest'
        file_to_search = dataset+'_user_' + str(input_i) +'_'+ similarity_metric  + '.pkl'

        loaded_dictionary = search_and_read_pickle(directory_path, file_to_search)

        if loaded_dictionary:
            print('Loaded Dictionary!')
        mapping = loaded_dictionary[int(input_i)]
    else:
        mapping = None

    algo = UserUserALGO()
    user_movie_ids = algo.fit(dfs['ratings'], input_i, similarity_metric, num_of_reccomendations_returned, preprocess=False, n_most_similar_mapping=mapping)
    
    if preprocess:
        directory_path = './preprocessed_data_100k' if 'ml-100k' in data_dir else './preprocessed_data_latest'
        file_to_search = dataset+'_item_' + str(input_i) +'_'+ similarity_metric  + '.pkl'

        loaded_dictionary = search_and_read_pickle(directory_path, file_to_search)

        if loaded_dictionary:
            print('Loaded Dictionary!')
        mapping = loaded_dictionary[int(input_i)]
    else:
        mapping = None

    algo = ItemItemALGO()
    item_movie_ids = algo.fit(dfs['ratings'], input_i, similarity_metric, num_of_reccomendations_returned, preprocess=False, n_most_similar_mapping=mapping)

    hybrid_algo = HybridALGO()
    weights = [0.5, 0.5]
    movie_ids = hybrid_algo.fit(user_movie_ids, item_movie_ids, weights, num_recommendations)

else:
    print("Algorithm not recognized: ", algorithm)
    exit(1)

print("\n\nRecommendations: ")
print("-----------------")

titles = movie_ids_to_movie_titles(movie_ids, dfs['movies'])

for i, title in enumerate(titles):
    print(i+1, " -> ", title)
print("\n\nExecution time: ", time.time() - start_time, " seconds")
print("\n\n")
