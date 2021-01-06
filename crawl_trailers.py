#!/usr/bin/env python
# coding: utf-8

# In[40]:


from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
import os, time
import json, ast

root_path   = Path('../steam_data/')
games_path  = Path(root_path, 'steam_games.json')

output_path = Path(root_path, 'media')
if not os.path.isdir(output_path):
    os.mkdir(output_path)

data = []
with open(games_path) as fp:
    lines = fp.read().split('\n')
    for line in lines[:-1]:
        data.append(ast.literal_eval(line))


# In[2]:


data[0]


# In[10]:


import requests

api_base_url = "http://store.steampowered.com/api/appdetails/"

gameid = 730
endpoint = f'{api_base_url}?appids={gameid}'
r = requests.get(endpoint)
print(r.json()['730']['data'].keys())


# In[46]:


def download_media(content, gameid):

    game_output_path = Path(output_path, str(gameid))
    if not os.path.isdir(game_output_path):
        os.mkdir(game_output_path)
    else:
        print(f'Game id {gameid} processed twice!')
    
    #### HEADER IMAGE
    try:
        url = content['header_image']
        response = requests.get(url)
        im = Image.open(BytesIO(response.content))    

        with open(Path(game_output_path, 'header_image.jpg'), 'w') as fp:
            im.save(fp)
    except:
        print(f'\ncould not process media {url} from game {gameid}')
        pass

    #### SCREENSHOTS

    if 'screenshots' in content.keys():
        for k, screenshot in enumerate(content['screenshots']):

            try:
                url = screenshot['path_full']
                response = requests.get(url)
                im = Image.open(BytesIO(response.content)) 

                with open(os.path.join(game_output_path, 'screenshot_{:02d}.jpg'.format(k)), 'w') as fp:
                    im.save(fp)

            except:
                print(f'\ncould not process screenshot {url} from game {gameid}')
                pass
        
    #### MOVIES
    if 'movies' in content.keys():
        for k, movie in enumerate(content['movies']):
            if k > 2: break

            if 'mp4' in movie.keys():
                url = movie['mp4']['480']
                save_name = f'movie_{k}_trailer.mp4' if 'trailer' in movie['name'].lower() else f'movie_{k}.mp4'
            else:
                url = movie['webm']['480']
                save_name = f'movie_{k}_trailer.webm' if 'trailer' in movie['name'].lower() else f'movie_{k}.webm'

            try:
                stream = requests.get(url, stream = True) 

                with open(Path(game_output_path, save_name), 'wb') as f: 
                    for chunk in stream.iter_content(chunk_size = 1024*1024): 
                        if chunk: 
                            f.write(chunk)
            except:
                print(f'\ncould not process movie {url} from game {gameid}')
                pass
    


# In[47]:


for i, game in enumerate(data):
    if 'id' not in game.keys(): 
        print('\n##### ID missing\n')
        print(game)
        continue

    print('\r{}/{} - {}: {} ######'.format(i, len(data), game['id'], game['app_name']), end='', flush=True)
    if i < 72: continue
    gameid = game['id']
    endpoint = f'{api_base_url}?appids={gameid}'

    r = requests.get(endpoint)
    retrieved = False
    attempts = 0
    while not retrieved:
        try:
            content = r.json()[str(gameid)]['data']
            retrieved = True
        except:
            print(f"\nCould not retrive data from {gameid}")
            attempts += 1
            if attempts > 3: break
            time.sleep(2)
 
    if retrieved:
        download_media(content, gameid)

