import sys
import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
import math
from sklearn.model_selection import train_test_split
from surprise import KNNWithMeans
import time


def process_json(content, filename):
    if filename == 'steam_reviews.json':
        content = content.replace('u\'', '\'')
        content = eval(content)
        #if 'user_id' in content:
        #    user_id = content['user_id']
        #else:
        user_id = content['username']   
            
        if 'hours' not in content:
            content['hours'] = 0
            
        play_times = [(user_id, content['product_id'], float(content['hours']))]
    else:
        content = eval(content)
        user_id = content['user_id']
        play_times = []
        for item in content['items']:
            item_id = item['item_id']
            playtime = item['playtime_forever']
            play_times.append((user_id, item_id, float(playtime)))
        
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


def filter_dataset(df):
    print("Filtering Dataset")
    tic = time.time()
    df = df.groupby('UserId').filter(lambda x : len(x)>=5)
    df = df.groupby('ItemId').filter(lambda x : len(x)>=5)
    toc = time.time()
    print("Dataset Filtered in {}s".format(int(toc-tic)))


def read_datafile(data_file):
    list_reviews = []
    cont = 0
    with open(data_file, 'r') as f:
        print("Reading data file")
        tic = time.time()
        for review in f:        
            review = process_json(review, data_file)
            list_reviews.extend(review)
            #cont += 1
            #if cont == 100:
            #    break
        toc = time.time()
        print("Finished reading data file in {}s".format(int(toc-tic)))
    return list_reviews


data_file = sys.argv[1]

cross_validate = False

list_reviews = read_datafile(data_file)

df = pd.DataFrame(list_reviews, columns=['UserId','ItemId','Playtime'])
#filter_dataset(df)
#normalize_playtime(df)

reader = Reader(rating_scale=(0, max(df.Playtime)))


sim_options = {
        "name": "cosine",
        "user_based": False,  # Compute  similarities between items
}
algo = KNNWithMeans(sim_options=sim_options)

if cross_validate:
    data = Dataset.load_from_df(df, reader)

    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
else:
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_data = Dataset.load_from_df(train_df, reader)
    training_set = train_data.build_full_trainset()
    algo.fit(training_set)
    
    for index, row in test_df.iterrows():
        user = row['UserId']
        item = row['ItemId']
        playtime = row['Playtime']
        prediction = algo.predict(user, item)
        print('{}:{} - {} / {}'.format(user,item,prediction,playtime))
