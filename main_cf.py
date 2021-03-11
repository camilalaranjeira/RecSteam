import argparse
import os
import sys
import time
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from scipy.sparse import csr_matrix
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

parser = argparse.ArgumentParser(description='Collaborative Filtering')
parser.add_argument('--folds_path', type=str, default="../folds", help='Path to the folder containing the train/test folds')
parser.add_argument('--fold', type=str, default='0', help='Fold number')
parser.add_argument('--grid_search', type=bool, default=False, help='Wether to use grid search or not')
parser.add_argument('--method', type=str, default='item-based', help='CF method to use (item-based|user-based|SVD)')

args = parser.parse_args()


train_path = os.path.join(args.folds_path, args.fold, 'train.csv')
test_path = os.path.join(args.folds_path, args.fold, 'test.csv')

train_df = pd.read_csv(train_path) 
test_df = pd.read_csv(test_path)

reader = Reader(rating_scale=(0, 10))

train_data = Dataset.load_from_df(train_df, reader).build_full_trainset()
test_data = Dataset.load_from_df(test_df, reader).build_full_trainset().build_testset()

if args.grid_search:
    param_grid = {
        "n_epochs": [50, 75, 100],
        "lr_all": [0.002, 0.005],
        "reg_all": [0.4, 0.6]
    }
    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=4, n_jobs=4)

    gs.fit(train_data)

    print(gs.best_score["rmse"])
    print(gs.best_params["rmse"])
else:    
    if args.method == 'SVD':
        algo = SVD(n_epochs=1, lr_all=0.002, reg_all=0.4, verbose=True)
    elif args.method == 'item-based':
        sim_options = {
        "name": "cosine",
        "user_based": False,  # Compute  similarities between items
        }
        algo = KNNWithMeans(sim_options=sim_options)
    elif args.method == 'user-based':
        sim_options = {
        "name": "cosine",
        "user_based": True,  # Compute  similarities between users
        }
        algo = KNNWithMeans(sim_options=sim_options)
    else:
        sys.exit('Invalid Method')

    algo.fit(train_data)
    predictions = algo.test(test_data)
    #for pred in predictions:
    #    print('{}:{} - {} / {}'.format(pred[0],pred[1],pred[3],pred[2]))
    print(accuracy.rmse(predictions))
