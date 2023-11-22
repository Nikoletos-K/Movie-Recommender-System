from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
from .metrics import cosine_similarity, dice_similarity, jaccard_similarity, pearson_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class RecommenderALGO:

    def __init__(self):
        self.metric = None
    
class UserUserALGO(RecommenderALGO):
    
    def __init__(self):
        super().__init__()
        
    def get_recommendation_score(self, i, movie_user_matrix, target_user_index, N):
        # print("Calculating recommendation score for movie: ", i)
        # print("Target user index: ", target_user_index)
        # print("Most similar users: ", N)
        # print("Movie ratings: ", x[target_user_index][i])
        # print("Movie ratings for similar users: ", [x[y][i] for y in N])
        # print("Similarity scores: ", [self.metric(np.array(x[y]), np.array(x[target_user_index])) for y in N])
        
        numerator = np.sum([self.metric(np.array(movie_user_matrix[y]), np.array(movie_user_matrix[target_user_index])) * movie_user_matrix[y][i] for y in N])
        denominator = np.sum([self.metric(np.array(movie_user_matrix[y]), np.array(movie_user_matrix[target_user_index])) for y in N])

        rec_score = numerator / denominator if denominator != 0 else 0
        return rec_score
    
    def most_similar_users(self, movie_index, movie_user_matrix, target_user_index, k):
        # Get users who have rated the item
        users_with_ratings = [user_id for user_id in range(len(movie_user_matrix)) if movie_user_matrix[user_id][movie_index] is not None]

        # Calculate similarities only for users who have rated the item
        similarities = np.array([self.metric(np.array(movie_user_matrix[user_id]), \
                                             np.array(movie_user_matrix[target_user_index])) for user_id in users_with_ratings])
        similar_users = [users_with_ratings[i] for i in np.argsort(similarities)[::-1][:k]]

        return set(similar_users)
            
    def fit(self, ratings_df, target_user_id, similarity_metric, top_n):
        print("\n\nUserUserALGO\n",20*'-')
        
        k = 128
        x = target_user_id
        
        if similarity_metric == "cosine":
            self.metric = cosine_similarity
        elif similarity_metric == "jaccard":
            self.metric = jaccard_similarity
        elif similarity_metric == "pearson":
            self.metric = pearson_similarity
        elif similarity_metric == 'dice':
            self.metric = dice_similarity
        else:
            raise ValueError("Unknown similarity metric: ", similarity_metric)
        
        # Pivot the DataFrame
        r_df = ratings_df.pivot(index='userId', columns='movieId', values='rating')
        r_df.replace({np.nan: None}, inplace=True)

        # Convert the pivot DataFrame to a numpy array
        r = r_df.to_numpy()
        self.user_mapping = r_df.index.to_list()
        print("user_mapping: ", self.user_mapping)

        if int(x) not in self.user_mapping:
            raise ValueError("Unknown user id: "+str(x))
        print("Given user id (x): ", x)
        x = self.user_mapping.index(int(x))
        print("Mapped: ", x)
        r_x = r[x].tolist()
        recommendations = []
        print("r_x: ", r_x)
        # For every movie that the user has not rated
        for i in range(0, len(r_x)):
            if r_x[i] is None:
                # Find the top k most similar users
                N = self.most_similar_users(i, r, x, k)
                rec_score = self.get_recommendation_score(i, r, x, N)
                recommendations.append((i,rec_score))

        # sort by similarity and keep the top n movies
        recommendations.sort(key=lambda x: x[1], reverse=True)
        recommendations = recommendations[:top_n]
        # map movies to movieIds
        recommendations = [(r_df.columns[x[0]], x[1]) for x in recommendations]
        # keep only the movieIds
        recommendations = [x[0] for x in recommendations]
        print("Recommendations for user with id: ", target_user_id, "\n", recommendations)

class ItemItemALGO(RecommenderALGO):
    
    def __init__(self):
        super().__init__()
        self.metric = None
    
    def get_recommendation_score(self, u, x, target_movie_index, N):
        # print("Calculating recommendation score for user: ", u)
        # print("Target movie index: ", target_movie_index)
        # print("Most similar items: ", N)
        # print("User ratings: ", x[u][target_movie_index])
        # print("User ratings for similar items: ", [x[u][target_movie] for target_movie in N])
        # print("Similarity scores: ", [self.metric(np.array(x[:, target_movie_index]), np.array(x[:, target_movie])) for target_movie in N])
        
            
        numerator = np.sum([self.metric(np.array(x[:, target_movie_index]), np.array(x[:,target_movie])) * x[u][target_movie] for target_movie in N])
        denominator = np.sum([self.metric(np.array(x[:,target_movie_index]), np.array(x[:,target_movie])) for target_movie in N])

        rec_score = numerator / denominator if denominator != 0 else 0
        return rec_score
    
    def most_similar_items(self, user_index, x, target_movie_index, k):
        # Get items that have been rated by the user
        rated_items = [movie_index for movie_index in range(len(x[user_index])) if x[user_index][movie_index] is not None]

        # Calculate similarities only for items that have been rated by the user
        similarities = np.array([self.metric(np.array(x[:, target_movie_index]), np.array(x[:, target_movie])) for target_movie in rated_items])
        similar_items = [rated_items[i] for i in np.argsort(similarities)[::-1][:k]]

        return set(similar_items)
            
    def fit(self, ratings_df, target_user_id, similarity_metric, top_n):
        print("\n\nItemItemALGO\n", 20 * '-')
        
        k = 128
        x = target_user_id

        if similarity_metric == "cosine":
            self.metric = cosine_similarity
        elif similarity_metric == "jaccard":
            self.metric = jaccard_similarity
        elif similarity_metric == "pearson":
            self.metric = pearson_similarity
        elif similarity_metric == 'dice':
            self.metric = dice_similarity
        else:
            raise ValueError("Unknown similarity metric: ", similarity_metric)
        
        # Pivot the DataFrame
        r_df = ratings_df.pivot(index='userId', columns='movieId', values='rating')
        r_df.replace({np.nan: None}, inplace=True)

        # Convert the pivot DataFrame to a numpy array
        r = r_df.to_numpy()
        
        self.movie_mapping = r_df.columns.to_list()
        # print("movie_mapping: ", self.movie_mapping)

        self.user_mapping = r_df.index.to_list()
        # print("user_mapping: ", self.user_mapping)
 
 
        if int(x) not in self.user_mapping:
            raise ValueError("Unknown user id: "+str(x))
        x = self.user_mapping.index(int(x))
        
        self.movie_index_mapping = {index: movie_id for index, movie_id in enumerate(self.movie_mapping)}
        # r_x = r[:, x]
        r_x = r[x].tolist()        
        # print("r_x: ", r_x)
        recommendations = []
        # print(r_x)
        for u in range(0, len(r_x)):
            if r_x[u] is None:
                N = self.most_similar_items(x, r, u, k)
                # print("Most similar items:", N)
                rec_score = self.get_recommendation_score(x, r, u, N)
                recommendations.append((u, rec_score))

        # Sort by similarity and keep the top n users
        recommendations.sort(key=lambda x: x[1], reverse=True)
        recommendations = recommendations[:top_n]

        print(recommendations)
        # print(self.movie_index_mapping)
        # Map user indices to user IDs
        recommendations = [self.movie_index_mapping[x[0]] for x in recommendations]
       
        print("Recommendations for user with id: ", target_user_id, "\n", recommendations)

class TagBasedALGO(RecommenderALGO):
    
    def __init__(self):
        super().__init__()
    
    def create_tag_vectors(self, tags_df):
        # Create a mapping between tag names and indices
        tag_mapping = {tag: i for i, tag in enumerate(tags_df['tag'].unique())}
        
        # Create a DataFrame to store the counts of tags for each movie
        tag_counts_df = pd.pivot_table(tags_df, values='userId', index='movieId', columns='tag', aggfunc='count', fill_value=0)
        
        # Convert the DataFrame to a 2D array
        unique_movies = tag_counts_df.index.to_numpy()
        tag_vectors = tag_counts_df.to_numpy()
        
        return unique_movies, tag_vectors, tag_mapping

    
    def fit(self, tags_df, movie_id, similarity_metric, num_recommendations):
        
        print("\n\nItemItemALGO\n", 20 * '-')

        if similarity_metric == "cosine":
            self.metric = cosine_similarity
        elif similarity_metric == "jaccard":
            self.metric = jaccard_similarity
        elif similarity_metric == "pearson":
            self.metric = pearson_similarity
        elif similarity_metric == 'dice':
            self.metric = dice_similarity
        else:
            raise ValueError("Unknown similarity metric: ", similarity_metric)
        
        
        # Assuming tags_df is your DataFrame with tags information
        unique_movies, tag_vectors, tag_mapping = self.create_tag_vectors(tags_df)

        
        # Access tag vectors and mapping as needed
        # print("Unique movies: \n", unique_movies)
        # print("Tag vectors: \n", tag_vectors[8])
        # print("Tag mapping: \n", tag_mapping)
    
        # Get the index of the movie
        movie_index = unique_movies[int(movie_id)]

        # Get the tag vector for the movie
        movie_tag_vector = tag_vectors[movie_index]
        
        # Calculate the similarity score between the movie and all other movies
        similarity_scores = [self.metric(movie_tag_vector, tag_vector) for tag_vector in tag_vectors]
        
        # Get the indices of the top n most similar movies
        similar_movie_indices = np.argsort(similarity_scores)[::-1][:num_recommendations]
        
        # Get the movie IDs of the top n most similar movies
        similar_movies = unique_movies[similar_movie_indices]
        
        print("Recommendations for movie with id: ", movie_id, "\n", similar_movies)

class ContentBasedALGO(RecommenderALGO):
    
    def __init__(self):
        super().__init__()
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.movie_mapping = None
        self.metric = None
    
    def create_tfidf_matrix(self, movies_df):
        # Use the TfidfVectorizer to create a TF-IDF matrix based on movie titles
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(movies_df['title'])
    
    def get_recommendation_score(self, target_movie_index, N):
        target_vector = self.tfidf_matrix[target_movie_index]
        similarity_scores = [self.metric(target_vector, self.tfidf_matrix[target_movie]) for target_movie in N]
        rec_score = np.sum(similarity_scores)
        return rec_score
    
    def most_similar_movies(self, target_movie_index, k):
        # Calculate similarities with all movies
        similarities = [self.metric(self.tfidf_matrix[target_movie_index], self.tfidf_matrix[target_movie]) for target_movie in range(len(self.tfidf_matrix))]
        
        # Find the top k similar movies
        similar_movies = [i for i in np.argsort(similarities)[::-1][:k]]

        return set(similar_movies)
    
    def fit(self, movies_df, target_movie_id, similarity_metric, top_n):
        print("\n\nContentBasedALGO\n", 20 * '-')
        
        x = target_movie_id
        
        if similarity_metric == "cosine":
            self.metric = cosine_similarity
        elif similarity_metric == "jaccard":
            self.metric = jaccard_similarity
        elif similarity_metric == "pearson":
            self.metric = pearson_similarity
        elif similarity_metric == 'dice':
            self.metric = dice_similarity
        else:
            raise ValueError("Unknown similarity metric: ", similarity_metric)
        

        #  Lowercase and remove punctuation
        movies_df['title'] = movies_df['title'].str.lower().str.replace('[^\w\s]', '')
        titles = movies_df['title'].tolist()
        # print("Titles ", titles)
        
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(titles)
        
        # Store movie mapping
        self.movie_mapping = movies_df['movieId'].to_list()
        
        if int(x) not in self.movie_mapping:
            raise ValueError("Unknown movie id: " + str(x))
        x = self.movie_mapping.index(int(x))
        
        recommendations = []
        num_movies = len(titles)

        for i in range(num_movies):
            recommendations.append((i,self.metric(self.tfidf_matrix[i].toarray().flatten(), self.tfidf_matrix[x].toarray().flatten())))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        recommendations = recommendations[:top_n]
        
        # map movies to movieIds and remove the target movie
        recommendations = [(self.movie_mapping[t[0]], t[1]) for t in recommendations if t[0] != x]
        
        print("Recommendations for movie with id: ", target_movie_id, "\n", recommendations)

class HybridALGO(RecommenderALGO):
    
    def __init__(self, data, user, item, rating):
        super().__init__(data, user, item, rating)
        
        