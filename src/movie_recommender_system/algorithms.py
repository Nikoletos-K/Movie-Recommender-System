from scipy.sparse import coo_matrix
import numpy as np

from .metrics import cosine_similarity, dice_similarity, jaccard_similarity, pearson_similarity

class RecommenderALGO:
    
    def __init__(self):
        pass
    
class UserUserALGO(RecommenderALGO):
    
    def __init__(self):
        super().__init__()
        self.metric = None
    
    def get_recommendation_score(self, i, x, target_user_id, N):
        # print("get_recommendation_score")
        
        numerator = 0
        denominator = 0
        for y in N:
            numerator += self.metric(np.array(x[y]), np.array(x[target_user_id])) * x[y][i]
            denominator += self.metric(np.array(x[y]), np.array(x[target_user_id]))
        
        rec_score = numerator / denominator if denominator != 0 else 0

        # print("rec_score: ", rec_score)
        
        return rec_score
        
    
    def most_similar_users(self, movie_id, x, target_user_id, k):
        
        most_similar_users = []
        # print("len: ",len(x))
        for user_id in range(0, len(x)):
            if user_id != target_user_id and x[user_id][movie_id] is not None:
                similarity = self.metric(np.array(x[user_id]), np.array(x[target_user_id]))
                most_similar_users.append((user_id, similarity))

                    # print("User ", ,": ", x[movie_id], user_r[movie_id])        

        # sort by similarity and keep the top k users
        most_similar_users.sort(key=lambda x: x[1], reverse=True)
        most_similar_users = set(x[0] for x in most_similar_users[:k])
        # print(most_similar_users)
        return most_similar_users
    
    def fit(self, ratings_df, target_user_id, similarity_metric, top_n):
        print("\n\nUserUserALGO\n",20*'-')
        
        k = 128
        x = target_user_id
        print(ratings_df)
        
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

        # print(r_df)
        # Convert the pivot DataFrame to a numpy array
        r = r_df.to_numpy()
        
        # print('r:', r_df.index.to_list())
        self.user_mapping = r_df.index.to_list()
        print("user_mapping: ", self.user_mapping)
        if int(x) not in self.user_mapping:
            raise ValueError("Unknown user id: "+str(x))
        # print("recs of user with id: ", x, "\n", r_df.loc[int(x)].to_numpy())
        x = self.user_mapping.index(int(x))
        r_x = r[x].tolist()
        # print(r_x)

        recommendations = []
        
        for i in range(0, len(r_x)):
            if r_x[i] is None:
                # print("Movie ", i," has not been reccomended")
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
    
    def __init__(self, data, user, item, rating):
        super().__init__(data, user, item, rating)
    
class TagBasedALGO(RecommenderALGO):
    
    def __init__(self, data, user, item, rating):
        super().__init__(data, user, item, rating)
    
class ContentBasedALGO(RecommenderALGO):
    
    def __init__(self, data, user, item, rating):
        super().__init__(data, user, item, rating)
        
class HybridALGO(RecommenderALGO):
    
    def __init__(self, data, user, item, rating):
        super().__init__(data, user, item, rating)
        
        