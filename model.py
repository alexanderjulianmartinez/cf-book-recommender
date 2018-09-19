import pandas as pd
import numpy as np
import re
import os, sys
from sklearn.neighbors import NearestNeighbors
import ipywidgets as widgets
from contextlib import contextmanager
from IPython.display import display, clear_output

books_df = pd.read_csv("/Users/alexandermartinez/Downloads/book-recommend-data/Books.csv", encoding = "ISO-8859-1")
events_df = pd.read_csv("/Users/alexandermartinez/Downloads/book-recommend-data/UserEvents.csv", encoding = "ISO-8859-1")
users_df = pd.read_csv("/Users/alexandermartinez/Downloads/book-recommend-data/Users.csv", encoding = "ISO-8859-1")

global k, metric
k = 10
metric = 'cosine'

def find_k_users(user, impressions, metric = metric, k = k):
    """
    Finds k similar users
    :param: user: userId
    impressions: impressions matrix (rows = users, columns = bookIds)
    k: number of users to look for

    :return similarities and and indices of k similar users
    """
    similarities = []
    indices = []
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(impressions)

    distances, indices = model_knn.kneighbors(impressions.iloc[user-1,:].values.reshape(1, -1), n_neighbors=k+1)
    similarities = 1-distances.flatten()
    print('{0} most similar users for User {1}:\n'.format(k, user))
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user:
            continue
        else:
            print('{0}: User {1}, with similarity of {2}'.format(i, indices.flatten()[i] + 1, similarities.flatten()[i]))

    return similarities, indices


def find_k_items(bookId, impressions, metric=metric, k=k):
    """
    Finds k similar users
    :param: bookId: bookId
    impressions: impressions matrix (rows = users, columns = bookIds)
    k: number of users to look for

    :return similarities and and indices of k similar items
    """
    similarities = []
    indices = []
    impressions = impressions.T
    loc = impressions.index.get_loc(bookId)
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(impressions)

    distances, indices = model_knn.kneighbors(impressions.iloc[loc, :].values.reshape(1, -1), n_neighbors=k + 1)
    similarities = 1 - distances.flatten()

    return similarities, indices

def predict_user_based(user, bookId, impressions, metric=metric, k=k):
    """
    Predict impression for specified user-item combination based on user-based approach

    """
    prediction = 0
    similarities, indices = find_k_users(user, impressions, metric, k)
    mean_impression = impressions.loc[user-1,:].mean()
    sum_wt = np.sum(similarities)-1
    product = 1
    wtd_sum = 0

    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user:
            continue
        else:
            impressions_diff = impressions.iloc[indices.flatten()[i], bookId-1]-np.mean(impressions.iloc[indices.flatten()[i],:])
            product = impressions_diff * similarities[i]
            wtd_sum = wtd_sum + product

    prediction = int(round(mean_impression + (wtd_sum/sum_wt)))

    if prediction < 0:
        prediction = 0
    elif prediction > 6:
        prediction = 6


    print('\nPredicted rating for user {0} -> item {1}: {2}'.format(user, bookId, prediction))

    return prediction

def predict_item_based(user, bookId, impressions, metric=metric, k=k):
    """
    Predict impression for specified user-item combination based on item-based approach

    """
    prediction = wtd_sum = 0
    user_loc = impressions.index.get_loc(user)
    item_loc = impressions.columns.get_loc(bookId)
    similarities, indices = find_k_items(user, impressions)
    sum_wt = np.sum(similarities) - 1
    product = 1

    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == item_loc:
            continue
        else:
            product = impressions.iloc[user_loc, indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product
    prediction = int(round(wtd_sum / sum_wt))

    if prediction < 0:
        prediction = 0
    elif prediction > 6:
        prediction = 6

    print('\nPredicted rating for user {0} -> item {1}: {2}'.format(user, bookId, prediction))

    return prediction

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def recommend_item(user, impressions, metric=metric):
    if(user not in impressions.index.values) or type(user) is not int:
        print("User should be a valid integer from this list :\n\n {}" .format(re.sub('[\[\]]', '',
                                                                                      np.array_str(impressions.index.values))))
    else:
        ids = ['Item-based (correlation)','Item-based (cosine)','User-based (correlation)','User-based (cosine)']
        select = widgets.Dropdown(options=ids, value=ids[0], description='Select Approach', width='1000px')
        def on_change(change):
            clear_output(wait=True)
            prediction = []
            if change['type'] == 'change' and change['name'] == 'value':
                if (select.value == 'Item-based (correlation)') | (select.value == 'User-based (correlation)'):
                    metric = 'correlation'
                else:
                    metric = 'cosine'
                with suppress_stdout():
                    if (select.value == 'Item-based (correlation)') | (select.value == 'Item-based (cosine)'):
                        for i in range(impressions.shape[1]):
                            if (impressions[str(impressions.columns[i])][user] != 0):  # not rated already
                                prediction.append(predict_item_based(user, str(impressions.columns[i]), impressions, metric))
                            else:
                                prediction.append(-1)  # for already rated items
                    else:
                        for i in range(impressions.shape[1]):
                            if (impressions[str(impressions.columns[i])][user] != 0):  # not rated already
                                prediction.append(predict_user_based(user, str(impressions.columns[i]), impressions, metric))
                            else:
                                prediction.append(-1)  # for already rated items
                prediction = pd.Series(prediction)
                prediction = prediction.sort_values(ascending=False)
                recommended = prediction[:10]
                print("As per {0} approach....Following books are recommended...".format(select.value))
                for i in range(len(recommended)):
                    print("{0}. {1}".format(i + 1, books_df.bookTitle[recommended.index[i]].encode('utf-8')))

        select.observe(on_change)
        display(select)

if __name__ == '__main__':
    # To run script enter 'python model.py user path/to/matrix' in commandline
    user = sys.argv[1]
    impressions = sys.argv[2]
    recommend_item(user, impressions)