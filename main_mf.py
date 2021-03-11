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
#from sklearn.model_selection import train_test_split
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from surprise import SVD, KNNWithMeans
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import GridSearchCV

#RATING_TYPE = 'binary'
#RATING_TYPE = 'log'
RATING_TYPE = 'bins'

GRID_SEARCH = False

#ALGO = 'SVD'
ALGO = 'item-based'

def process_json(content, filename):
    if filename == 'steam_reviews.json':
        content = content.replace('u\'', '\'')
        content = eval(content)
        user_id = content['username']   
            
        if 'hours' not in content:
            content['hours'] = 0
        playtime = content['hours']
        if RATING_TYPE == 'log':
            if playtime > 0:
                playtime = math.log10(playtime)
        elif RATING_TYPE == 'bins':
            if playtime > 100:
                playtime = 100
            playtime = playtime//10
        else:
            playtime = 1
            
        play_times = [(user_id, content['product_id'], playtime)]
    else:
        content = eval(content)
        user_id = content['user_id']
        play_times = []
        for item in content['items']:
            item_id = item['item_id']
            playtime = item['playtime_forever']
            if playtime >= 0:
                if RATING_TYPE == 'log':
                    if playtime > 0:
                        playtime = math.log10(playtime)
                elif RATING_TYPE == 'bins':
                    if playtime > 100:
                        playtime = 100
                    playtime = playtime//10
                else:
                    playtime = 1
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
            if len(review) > 0:            
                list_reviews.extend(review)
                cont += 1
                #if cont == 100000:
                #    break
        toc = time.time()
        print("Finished reading data file in {}s".format(int(toc-tic)))
    return list_reviews


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
"""
if RATING_TYPE == 'log':
    reader = Reader(rating_scale=(0, max(interactions_full_df['Playtime'])))
elif RATING_TYPE == 'bins':
    reader = Reader(rating_scale=(0, 10))
else:
    reader = Reader(rating_scale=(0, 1))

data = Dataset.load_from_df(interactions_full_df[["UserId", "ItemId", "Playtime"]], reader)

if GRID_SEARCH:
    param_grid = {
        "n_epochs": [50, 75, 100],
        "lr_all": [0.002, 0.005],
        "reg_all": [0.4, 0.6]
    }
    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=4, n_jobs=4)

    gs.fit(data)

    print(gs.best_score["rmse"])
    print(gs.best_params["rmse"])
else:
    # Train/test split
    trainset, testset = train_test_split(data, test_size=.2)
    
    if ALGO == 'SVD':
        algo = SVD(n_epochs=1, lr_all=0.002, reg_all=0.4, verbose=True)
    else:
        sim_options = {
        "name": "cosine",
        "user_based": False,  # Compute  similarities between items
        }
        algo = KNNWithMeans(sim_options=sim_options)

    algo.fit(trainset)
    predictions = algo.test(testset)
    for pred in predictions:
        print('{}:{} - {} / {}'.format(pred[0],pred[1],pred[3],pred[2]))
    print(accuracy.rmse(predictions))
"""
