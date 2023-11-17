import pandas as pd
import os
import pandas as pd

import argparse

from movie_recommender_system.metrics import jaccard_similarity, cosine_similarity, pearson_similarity
from movie_recommender_system.algorithms import UserUserALGO, ItemItemALGO, TagBasedALGO, ContentBasedALGO, HybridALGO


# READ CLI INPUTS
parser = argparse.ArgumentParser(description='Movie Recommender System')
parser.add_argument('-d', type=str, required=True)
parser.add_argument('-n', type=int, required=True)
parser.add_argument('-s', type=str, required=True)
parser.add_argument('-a', type=str, required=True)
parser.add_argument('-i', type=str, required=True)
args = parser.parse_args()
data_dir = args.d
num_recommendations = args.n
similarity_metric = args.s
algorithm = args.a
input_i = args.i

print("\n\nMovie Recommender System")
print("-----------------------")
print("Data directory: ", data_dir)
print("Number of recommendations: ", num_recommendations)
print("Similarity metric: ", similarity_metric)
print("Algorithm: ", algorithm)
print("Input: ", input_i)

# DATA READING
# data_folder = '../../data/ml-latest/ml-latest'
data_folder = data_dir
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
print("\nReading CSV files from: ", data_folder)
print("CSV files found: ", csv_files)
dfs = {}
for file in csv_files:
    print("\nReading: ", file)
    df = pd.read_csv(os.path.join(data_folder, file), nrows=3383216)
    file_name = file.split('.')[0]
    print("Saving as: ", file_name)
    print("Size: ", df.shape)
    print("Columns: ", df.columns)
    dfs[file_name] = df
# print(dfs)

if algorithm == "user":
    # find ratings for user with id input_i
    # user_ratings = dfs[2][dfs[2]['userId'] == input_i]
    user_ratings = dfs['ratings']
    algo = UserUserALGO()
    algo.fit(user_ratings, input_i, similarity_metric, num_recommendations)
    # print(user_ratings)
elif algorithm == "item":
    pass
elif algorithm == "tag":
    pass
elif algorithm == "title":
    pass
elif algorithm == "hybrid":
    pass
else:
    print("Algorithm not recognized: ", algorithm)
    exit(1)
