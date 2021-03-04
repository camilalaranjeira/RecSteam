import sys
import pandas as pd

# stem keys: {'text', 'early_access', 'page', 'compensation', 'page_order', 'date', 'product_id', 'hours', 'products', 'user_id', 'username', 'found_funny'}


def process_json(content, filename):
    if filename == 'steam_reviews.json':
        content = content.replace('u\'', '\'')
        content = eval(content)
        if 'user_id' in content:
            user_id = content['user_id']
        else:
            user_id = content['username']
            
        if 'hours' not in content:
            content['hours'] = 0
            
        play_times = [(user_id, content['product_id'], content['hours'])]
    else:
        content = eval(content)
        user_id = content['user_id']
        play_times = []
        for item in content['items']:
            item_id = item['item_id']
            playtime = item['playtime_forever']
            play_times.append((user_id, item_id, playtime))
        
    return play_times

reviews_file = sys.argv[1]

list_reviews = []
with open(reviews_file, 'r') as f:
    for review in f:
        review = process_json(review, reviews_file)
        list_reviews.extend(review)

df = pd.DataFrame(list_reviews, columns=['UserId','ItemId','Playtime'])
print(df)
