import pandas as pd
import os
import pandas as pd

movies_df = pd.read_csv('./data/ml-latest/movies.csv')

data_folder = './data/ml-latest'
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

dfs = []
for file in csv_files:
    df = pd.read_csv(os.path.join(data_folder, file))
    dfs.append(df)

# Now you can access each dataframe by its index in the dfs list
