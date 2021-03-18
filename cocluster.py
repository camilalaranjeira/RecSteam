from surprise import CoClustering
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import GridSearchCV
import pandas as pd
import time

root = "./folds_score/"
gs = False

if gs:
    fold = ' ' + str(0)
    train_df, test_df = pd.read_csv(root + fold +'/train.csv'), pd.read_csv(root + fold + '/test.csv')

    reader = Reader(rating_scale=(0, 10))

#     train_data = Dataset.load_from_df(train_df, reader).build_full_trainset()
#     data = Dataset.load_from_folds([(root + fold +'/train.csv', root + fold + '/test.csv')], reader)
#     test_data = Dataset.load_from_df(test_df, reader).build_full_trainset().build_testset()
    data = Dataset.load_from_df(train_df, reader)
          
    param_grid = {
        "n_cltr_u": [20, 30, 40],
        "n_cltr_i": [20, 30, 40],
        "n_epochs": [50, 75, 100],
        "random_state": [42]
    }
    
    gs = GridSearchCV(CoClustering, param_grid, measures=["rmse", "mae"], cv=5, n_jobs=4)

    gs.fit(data)

    print(gs.best_score["rmse"])
    print(gs.best_params["rmse"])
        
else:
    for i in range(5):
        fold = ' ' + str(i)
        train_df, test_df = pd.read_csv(root + fold +'/train_score.csv').drop(["Unnamed: 0","playtime"], axis=1), pd.read_csv(root + fold + '/test_score.csv').drop(["Unnamed: 0","playtime"], axis=1)

        reader = Reader(rating_scale=(0, 10))

        train_data = Dataset.load_from_df(train_df, reader).build_full_trainset()
        test_data = Dataset.load_from_df(test_df, reader).build_full_trainset().build_testset()



        print("\nCo-CLustering - Testing fold{}:".format(fold))

        algo = CoClustering(n_cltr_u=40, n_cltr_i=40, n_epochs=100, random_state=42, verbose=True)

        tic = time.time()
        algo.fit(train_data)
        toc = time.time()

        print('Finished fit model in {}s'.format(toc-tic))

        predictions = algo.test(test_data)

        accuracy.rmse(predictions)
        #     print("RMSE: ", rmse)