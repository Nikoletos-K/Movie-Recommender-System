from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
from .metrics import cosine_similarity, dice_similarity, jaccard_similarity, pearson_similarity
from .utils import CustomTfidfVectorizer
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_matrix

class RecommenderALGO:

    def __init__(self):
        self.metric = None

class UserUserALGO(RecommenderALGO):
    
    def __init__(self):
        super().__init__()

    def get_recommendation_score(self, i, movie_user_matrix, target_user_index, N):
        target_vector = movie_user_matrix[target_user_index].toarray().flatten()
        numerator = np.sum([self.metric(movie_user_matrix[y].toarray().flatten(), target_vector) * movie_user_matrix[y, i] for y in N])
        denominator = np.sum([self.metric(movie_user_matrix[y].toarray().flatten(), target_vector) for y in N])

        rec_score = numerator / denominator if denominator != 0 else 0
        return rec_score
    
    def most_similar_users(self, movie_index, movie_user_matrix, target_user_index, k):        
        nonzero_indices = movie_user_matrix[:, movie_index].nonzero()[0]
        similarities = np.array([self.metric(movie_user_matrix[user_id].toarray().flatten(), \
                                             movie_user_matrix[target_user_index].toarray().flatten()) for user_id in nonzero_indices])
        similar_users = [nonzero_indices[i] for i in np.argsort(similarities)[::-1][:k]]
        return set(similar_users)
            
    def fit(self, ratings_df, target_user_id, similarity_metric, top_n, preprocess=False, n_most_similar_mapping=None, verbose=True):
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
        r_df.fillna(0, inplace=True)
        r_sparse = csr_matrix(r_df.values)

        self.user_mapping = r_df.index.to_list()
        if int(x) not in self.user_mapping:
            raise ValueError("Unknown user id: "+str(x))
        x = self.user_mapping.index(int(x))
        
        r_x = r_sparse[x].toarray().flatten()
        recommendations = []
        
        preprocess_calculations = {}
        for i in tqdm(range(0, len(r_x)), desc=str("UserUserALGO (user_id:" +str(target_user_id)+ ")")):
            if r_x[i] == 0:
                # Find the top k most similar users
                if n_most_similar_mapping is not None and preprocess==False:
                    N = n_most_similar_mapping[i]
                else:
                    N = self.most_similar_users(i, r_sparse, x, k)
                if preprocess:
                    preprocess_calculations[i] = N
                    continue
                rec_score = self.get_recommendation_score(i, r_sparse, x, N)
                recommendations.append((i,rec_score))

        if preprocess:
            return preprocess_calculations

        recommendations.sort(key=lambda x: x[1], reverse=True)

        if top_n == -1:
            return recommendations
        
        recommendations = recommendations[:top_n]
        recommendations = [(r_df.columns[x[0]], x[1]) for x in recommendations]
        recommendations = [x[0] for x in recommendations]
            
        return recommendations

class ItemItemALGO(RecommenderALGO):

    def __init__(self):
        super().__init__()
        self.metric = None

    def get_recommendation_score(self, u, x, target_movie_index, N):
        target_vector = x[:, target_movie_index].toarray().flatten()
        numerator = np.sum([self.metric(target_vector, x[:,target_movie].toarray().flatten()) * x[u,target_movie] for target_movie in N])
        denominator = np.sum([self.metric(target_vector, x[:,target_movie].toarray().flatten()) for target_movie in N])

        rec_score = numerator / denominator if denominator != 0 else 0
        return rec_score

    def most_similar_items(self, user_index, x, target_movie_index, k):
        nonzero_rated_items = x[user_index, :].nonzero()[1]
        similarities = np.array([self.metric(x[:, target_movie_index].toarray().flatten(), 
                                             x[:, target_movie].toarray().flatten()) for target_movie in nonzero_rated_items])
        similar_items = [nonzero_rated_items[i] for i in np.argsort(similarities)[::-1][:k]]

        return set(similar_items)
            
    def fit(self, ratings_df, target_user_id, similarity_metric, top_n, preprocess=False, n_most_similar_mapping=None, verbose=True):
        
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
        r_df.fillna(0, inplace=True)
        r_sparse = csr_matrix(r_df.values)
        
        self.movie_mapping = r_df.columns.to_list()
        self.user_mapping = r_df.index.to_list()
        if int(x) not in self.user_mapping:
            raise ValueError("Unknown user id: "+str(x))
        x = self.user_mapping.index(int(x))
        
        self.movie_index_mapping = {index: movie_id for index, movie_id in enumerate(self.movie_mapping)}

        r_x = r_sparse[x].toarray().flatten()
        recommendations = []
        preprocess_calculations = {}
        for u in tqdm(range(0, r_x.shape[0]), desc=str("ItemItemALGO (user_id:" +target_user_id+ ")")):
            if r_x[u]!=0:
                if n_most_similar_mapping is not None and preprocess==False:
                    N = n_most_similar_mapping[u]
                else:
                    N = self.most_similar_items(x, r_sparse, u, k)
                    
                if preprocess:
                    preprocess_calculations[u] = N
                    continue
                rec_score = self.get_recommendation_score(x, r_sparse, u, N)
                recommendations.append((u, rec_score))

        if preprocess:
            return preprocess_calculations

        recommendations.sort(key=lambda x: x[1], reverse=True)

        if top_n == -1:
            return recommendations

        recommendations = recommendations[:top_n]
        recommendations = [self.movie_index_mapping[x[0]] for x in recommendations]
        
        return recommendations

class TagBasedALGO(RecommenderALGO):
    
    def __init__(self):
        super().__init__()
    
    def create_tag_vectors(self, tags_df):        
        tag_counts_df = pd.pivot_table(tags_df, values='userId', index='movieId', columns='tag', aggfunc='count', fill_value=0)
        unique_movies = tag_counts_df.index.to_numpy()
        tag_vectors = tag_counts_df.to_numpy()

        return unique_movies, tag_vectors

    
    def fit(self, tags_df, movie_id, similarity_metric, num_recommendations):
        
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

        unique_movies, tag_vectors = self.create_tag_vectors(tags_df)
        movie_index = unique_movies[int(movie_id)]
        movie_tag_vector = tag_vectors[movie_index]
        similarity_scores = [self.metric(movie_tag_vector, tag_vector) for tag_vector in tag_vectors]
        similar_movie_indices = np.argsort(similarity_scores)[::-1][:num_recommendations]
        similar_movies = unique_movies[similar_movie_indices]
        
        return similar_movies

class ContentBasedALGO(RecommenderALGO):
    
    def __init__(self):
        super().__init__()
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.movie_mapping = None
        self.metric = None

    def most_similar_movies(self, target_movie_index, k):
        similarities = [self.metric(self.tfidf_matrix[target_movie_index], self.tfidf_matrix[target_movie]) for target_movie in range(len(self.tfidf_matrix))]
        similar_movies = [i for i in np.argsort(similarities)[::-1][:k]]

        return set(similar_movies)

    def fit(self, initial_movies_df, target_movie_id, similarity_metric, top_n):

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

        movies_df = initial_movies_df.copy()
        movies_df['title'] = movies_df['title'].str.lower().str.replace('[^\w\s]', '')
        titles = movies_df['title'].tolist()
        
        self.tfidf_vectorizer = CustomTfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(titles)
        
        self.movie_mapping = movies_df['movieId'].to_list()
        
        if int(x) not in self.movie_mapping:
            raise ValueError("Unknown movie id: " + str(x))
        x = self.movie_mapping.index(int(x))        
        
        recommendations = []
        num_movies = len(titles)
        for i in tqdm(range(num_movies), desc=str("ContentBasedALGO (movie_id:" +str(target_movie_id)+ ")")):
            recommendations.append((i,self.metric(self.tfidf_matrix[i],
                                                  self.tfidf_matrix[x])))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        recommendations = [(self.movie_mapping[t[0]], t[1]) for t in recommendations if t[0] != x]
        
        if top_n == -1:
            return [r[0] for r in recommendations]

        recommendations = recommendations[:top_n]

        return [r[0] for r in recommendations]

class HybridALGO(RecommenderALGO):
    
    def __init__(self):
        super().__init__()

    def fit(self, user_movie_ids, item_movie_ids, weights, titles, similarity_metric, top_n):
        
        all_movie_ids = []
        all_user_movie_ids = {}
        for movie in user_movie_ids:
            all_user_movie_ids[movie[0]]=movie[1]
            all_movie_ids.append(movie[0])

        all_item_movie_ids = {}
        for movie in item_movie_ids:
            all_item_movie_ids[movie[0]]=movie[1]
            all_movie_ids.append(movie[0])

        recommendations = []
        for movie in all_movie_ids:
            recommendations.append((movie,
                                    weights[0]*all_user_movie_ids[movie] if movie in all_user_movie_ids else 0 +
                                    weights[1]*all_item_movie_ids[movie] if movie in all_item_movie_ids else 0))
            
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        if top_n%2 == 0:
            num_movies_from_linear_combination = int(top_n/2)
        else:
            num_movies_from_linear_combination = int(top_n/2)+1

        recommendations = recommendations[:num_movies_from_linear_combination]
        reccomendations_ids = [r[0] for r in recommendations]
        reccomendations_ids_set = set(reccomendations_ids)
        new_reccomendations_ids = [x for x in reccomendations_ids]
        if top_n != 1:
            for movie_id in reccomendations_ids:
                content_based_algo = ContentBasedALGO()
                new_movie_ids = content_based_algo.fit(titles, movie_id, similarity_metric, -1)
                for new_movie_id in new_movie_ids:
                    if new_movie_id not in reccomendations_ids_set:
                        new_reccomendations_ids.append(new_movie_id)
                        reccomendations_ids_set.add(new_movie_id)
                        break
                if len(new_reccomendations_ids) == top_n:
                    break

        return new_reccomendations_ids
            
            

