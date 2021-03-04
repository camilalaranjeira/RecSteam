#based on https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101
import sys
import time
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:
    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(interactions_full_df['ItemId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, person_id):
        #Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['ItemId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['ItemId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['ItemId'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, 
                                               items_to_ignore=get_items_interacted(person_id, 
                                               interactions_train_indexed_df), 
                                               topn=100)
                                               #topn=10000000000)
        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, 
                                            sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                            seed=int(item_id)%(2**32))

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['ItemId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['ItemId'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            if idx % 100 == 0 and idx > 0:
                print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df 


class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df):
        self.popularity_df = popularity_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['ItemId'].isin(items_to_ignore)] \
                               .sort_values('Playtime', ascending = False) \
                               .head(topn)

        return recommendations_df



def process_json(content, filename):
    if filename == 'steam_reviews.json':
        content = content.replace('u\'', '\'')
        content = eval(content)
        user_id = content['username']   
            
        if 'hours' not in content:
            content['hours'] = 0
            
        play_times = [(user_id, content['product_id'], math.log10(content['hours']))]
    else:
        content = eval(content)
        user_id = content['user_id']
        play_times = []
        for item in content['items']:
            item_id = item['item_id']
            playtime = item['playtime_forever']
            if playtime > 0:
                playtime = math.log10(playtime)
            play_times.append((user_id, item_id, playtime))
        
    return play_times


def normalize_playtime(df):
    # Normalizing playtime
    print('Normalizing playtime')
    tic = time.time()
    for user in df.UserId.unique():
        selected = df.loc[df['UserId'] == user][['Playtime', 'ItemId']]
        rows = df[df['UserId'] == user].index.values
        items = selected.ItemId.values
        playtime = selected.Playtime.values
        mean = playtime.mean()
        std = playtime.std()
        n_items = len(items)
        for n in range(n_items):
            new_playtime = (playtime[n]-mean)/(std)
            if new_playtime < -3:
                new_playtime = -3
            elif new_playtime > 3:
                new_playtime = 3
            new_playtime += 3
            df.iat[rows[n],2] = new_playtime
    toc = time.time()
    print('Finished ormalizing in {}s'.format(toc-tic))


def read_datafile(data_file):
    list_reviews = []
    cont = 0
    with open(data_file, 'r') as f:
        print("Reading data file")
        tic = time.time()
        for review in f:        
            review = process_json(review, data_file)
            list_reviews.extend(review)
            cont += 1
            if cont == 500:
                break
        toc = time.time()
        print("Finished reading data file in {}s".format(int(toc-tic)))
    return list_reviews


def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the game information.
    interacted_items = interactions_df.loc[person_id]['ItemId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

data_file = sys.argv[1]

list_reviews = read_datafile(data_file)

interactions_df = pd.DataFrame(list_reviews, columns=['UserId','ItemId','Playtime'])

# Filtering dataframe
users_interactions_count_df = interactions_df.groupby(['UserId', 'ItemId']).size().groupby('UserId').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['UserId']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))
print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
    how = 'right',
    left_on = 'UserId',
    right_on = 'UserId')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))
interactions_full_df = interactions_from_selected_users_df

# Train/test split
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
    stratify=interactions_full_df['UserId'], 
    test_size=0.20,
    random_state=42)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))

#Indexing by personId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('UserId')
interactions_train_indexed_df = interactions_train_df.set_index('UserId')
interactions_test_indexed_df = interactions_test_df.set_index('UserId')

model_evaluator = ModelEvaluator()  

#Computes the most popular items
item_popularity_df = interactions_full_df.groupby('ItemId')['Playtime'].sum().sort_values(ascending=False).reset_index()
popularity_model = PopularityRecommender(item_popularity_df)
print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
print(pop_detailed_results_df.head(10))

