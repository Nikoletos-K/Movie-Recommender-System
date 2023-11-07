

class RecommenderALGO:
    
    def __init__(self, data, user, item, rating):
        self.data = data
        self.user = user
        self.item = item
        self.rating = rating

class UserUserALGO(RecommenderALGO):
    
    def __init__(self, data, user, item, rating):
        super().__init__(data, user, item, rating)

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
        
        