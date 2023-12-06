import pandas as pd
import os
import pandas as pd
import time
from tqdm import tqdm
import argparse
import pickle

from movie_recommender_system.algorithms import (
    UserUserALGO,
    ItemItemALGO,
    TagBasedALGO,
    ContentBasedALGO,
    HybridALGO
)
from movie_recommender_system.utils import movie_ids_to_movie_titles

# READ CLI INPUTS
parser = argparse.ArgumentParser(description='Movie Recommender System')
parser.add_argument('-d', type=str, required=True)
parser.add_argument('-u', type=str, required=False)

args = parser.parse_args()
data_dir = args.d
num_of_users = int(args.u) if args.u else -1


print("\n\nMovie Recommender System - PREPROCESSING")
print("-----------------------")
print("Data directory: ", data_dir)


# DATA READING
dfs = {}
dataset = 'mllatest'
if 'ml-latest' in data_dir:
    data_folder = data_dir
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    print("\nReading CSV files from: ", data_folder)
    print("CSV files found: ", csv_files)
    dfs = {}
    for file in csv_files:
        if file == 'genome-scores.csv' or file == 'genome-tags.csv' or file == 'links.csv':
            continue
        print("\nReading: ", file)
        df = pd.read_csv(os.path.join(data_folder, file), nrows=1000)
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

similarity_metrics = ['cosine', 'dice', 'jaccard', 'pearson']

for similarity_metric in tqdm(similarity_metrics):
    user_preprocess_data = {}
    users = sorted(dfs['ratings']['userId'].unique())
    for user in tqdm(users, desc="UserUserALGO"):
        # print("\nUser: ", user)
        algo = UserUserALGO()
        N_most_similar = algo.fit(dfs['ratings'], user, similarity_metric, 0, preprocess=True)
        user_preprocess_data[user] = N_most_similar

        # Save dict to file
        dir_path = './preprocessed_data_100k' if 'ml-100k' in data_dir else './preprocessed_data_latest'

        if not os.path.exists(dir_path):
            # Create the directory if it does not exist
            os.makedirs(dir_path)
            print(f'Directory created: {dir_path}')
        else:
            print(f'Directory already exists: {dir_path}')



        file_name = dataset + '_user_' + str(user) +'_'+ similarity_metric  + '.pkl'
        
        # Save the dictionary to the pickle file
        full_path = './'+dir_path+'/'+file_name
        with open(full_path, 'wb') as file:
            pickle.dump(user_preprocess_data, file)

        print(f'Dictionary saved to {str(full_path)}')
        
        if user == num_of_users:
            break
    
item_preprocess_data = {}

for similarity_metric in tqdm(similarity_metrics):
    item_preprocess_data = {}
    users = sorted(dfs['ratings']['userId'].unique())
    for user in tqdm(users, desc="ItemItemALGO"):
        algo = ItemItemALGO()
        N_most_similar = algo.fit(dfs['ratings'], user, similarity_metric, 0, preprocess=True)
        user_preprocess_data[user] = N_most_similar

        # Save dict to file
        dir_path = './preprocessed_data_100k' if 'ml-100k' in data_dir else './preprocessed_data_latest'

        if not os.path.exists(dir_path):
            # Create the directory if it does not exist
            os.makedirs(dir_path)
            print(f'Directory created: {dir_path}')
        else:
            print(f'Directory already exists: {dir_path}')

        file_name = dataset + '_item_' + str(user) +'_'+ similarity_metric  + '.pkl'

        # Save the dictionary to the pickle file
        full_path = './'+dir_path+'/'+file_name
        with open(full_path, 'wb') as file:
            pickle.dump(user_preprocess_data, file)

        print(f'Dictionary saved to {str(full_path)}')

        if user == 10:
            break
