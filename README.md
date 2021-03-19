# RecSteam
Recommendation Systems 2020.2 PPGCC/UFMG - Final Project

Dataset: https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data <br>
Steam API: https://developer.valvesoftware.com/wiki/Steam_Web_API

## Folds
Link: https://drive.google.com/file/d/1ra83yrZOjc_z8YasoBIq-ZA4pvSXlBM6/view?usp=sharing <br>
Descrição:
* `playtime`: Normalização A (vide baselines)
* `score`: Normalização B (vide baselines)

## Baselines

### Método A
- Filtrando usuários e itens que apresentam < 10 interações.
- Rating: **Playtime** [0:'inf']. Tempo em minutos de jogo.
- Normalizado playtime para o intervalo [0:10] mapeando playtimes em um intervalo truncado [0:100:10].
```python
playtime = playtime/60
  if playtime > 100:
      playtime = 100
  playtime = playtime // 10
```
#### Results
4035113 interactions, 53870 users and 7697 items after filtering. <br>
5-fold

Fold | Method     | RSME  |Fold | Method     | RSME  |Fold | Method     | RSME  |Fold | Method     | RSME  |Fold | Method     | RSME  |
-----|------------|-------|-----|------------|-------|-----|------------|-------|-----|------------|-------|-----|------------|-------|
0    | user_mean  | 2.063 |1    | user_mean  | 2.063 |2    | user_mean  | 2.059 |3    | user_mean  | 2.065 |4    | user_mean  | 2.062 |
0    | item_mean  | 1.753 |1    | item_mean  | 1.752 |2    | item_mean  | 1.747 |3    | item_mean  | 1.753 |4    | item_mean  | 1.753 |
0    | random     | 2.627 |1    | random     | 2.622 |2    | random     | 2.620 |3    | random     | 2.624 |4    | random     | 2.625 |
0    | item-based | 1.776 |1    | item-based | 1.774 |2    | item-based | 1.771 |3    | item-based | 1.777 |4    | item-based | 1.775 |
0    | SVD        | 1.753 |1    | SVD        | 1.756 |2    | SVD        | 1.749 |3    | SVD        | 1.756 |4    | SVD        | 1.753 |
0    | SVD-spotlight | 2.103 |1    | SVD-spotlight | 2.101 |2    | SVD-spotlight | 2.099 |3    | SVD-spotlight | 2.303 |4    | SVD-spotlight | 2.318 |
0    | Co-Clustering | 1.900 |1    | Co-Clustering | 1.913 |2    | Co-Clustering | 1.912 |3    | Co-Clustering | 1.912 |4    | Co-Clustering | 1.907 |
0    | i2v tag-based | 2.267 |1    | i2v tag-based | 2.256 |2    | i2v tag-based | 2.241 |3    | i2v tag-based | 2.242 |4    | i2v tag-based | 2.214 |
0    | genre tag-based | 2.227 |1    | genre tag-based | 2.221 |2    | genre tag-based | 2.209 |3    | genre tag-based | 2.220 |4    | genre tag-based | 2.204 | 
0    | genre+i2v tag-based | 2.209 |1    | genre+i2v tag-based | 2.230 |2    | genre+i2v tag-based | 2.218 |3    | genre+i2v tag-based | 2.216 |4    | genre+i2v tag-based | 2.214 | 
0    | i2v tag-based (50T) | 2.127 |1    | i2v tag-based (50T) | 2.128 |2    | i2v tag-based (50T) | 2.124 |3    | i2v tag-based | 2.130 |4    | i2v tag-based | 2.125 |
0    | genre tag-based (50T) | 2.144 |1    | genre tag-based (50T) | 2.144 |2    | genre tag-based (50T) | 2.139 |3    | genre tag-based (50T) | 2.146 |4    | genre tag-based (50T) | 2.143 | 
0    | genre+i2v tag-based (50T) | 2.143 |1    | genre+i2v tag-based (50T) | 2.144 |2    | genre+i2v tag-based (50T) | 2.140 |3    | genre+i2v tag-based (50T) | 2.147 |4    | genre+i2v tag-based (50T) | 2.144 | 
0    | Gram | 2.412 |1    | Gram | 2.410 |2    | Gram | 2.406 |3    | Gram | 2.412 |4    | Gram | 2.411 | 

item-based: KNN with Means, k=40.

SVD: epochs=100, lr=0.002, reg_all=0.4.

SVD-spotlight: epochs=10, learning_rate=1e-3, l2=1e-9, embedding_dim=128, batch_size=1024.

Co-Clustering: n_cltr_u=40, n_cltr_i=40, n_epochs=100.

### Método B
- Filtrando usuários e itens que apresentam < 10 interações.
- Rating: **Playtime** [0:'inf']. Tempo em minutos de jogo.
- Normalizado playtime para o intervalo [0:10] 
```python
if rating['playtime'] == user_median:  score = 5
elif rating['playtime'] > user_median: score = 5 + 5*(rating['playtime']-user_median)/(user_max-user_median)
else: score = 5 - 5*(rating['playtime']-user_min)/(user_median-user_min)
```
#### Results
4035113 interactions, 53870 users and 7697 items after filtering. <br>
5-fold

Fold | Method     | RSME  |Fold | Method     | RSME  |Fold | Method     | RSME  |Fold | Method     | RSME  |Fold | Method     | RSME  |
-----|------------|-------|-----|------------|-------|-----|------------|-------|-----|------------|-------|-----|------------|-------|
0    | user_mean  | 1.183 |1    | user_mean  | 1.183 |2    | user_mean  | 1.184 |3    | user_mean  | 1.186 |4    | user_mean  | 1.184 |
0    | item_mean  | 1.108 |1    | item_mean  | 1.108 |2    | item_mean  | 1.110 |3    | item_mean  | 1.111 |4    | item_mean  | 1.109 |
0    | random     | 4.486 |1    | random     | 4.485 |2    | random     | 4.487 |3    | random     | 4.485 |4    | random     | 4.486 |
0    | item-based | 1.262 |1    | item-based | 1.260 |2    | item-based | 1.261 |3    | item-based | 1.259 |4    | item-based | 1.260 |
0    | SVD        | 1.228 |1    | SVD        | 1.227 |2    | SVD        | 1.229 |3    | SVD        | 1.227 |4    | SVD        | 1.227 |
0    | Co-Clustering | 1.350 |1    | Co-Clustering | 1.351 |2    | Co-Clustering | 1.357 |3    | Co-Clustering | 1.350 |4    | Co-Clustering | 1.350 |
0    | i2v tag-based | 1.257 |1    | i2v tag-based | 1.239 |2    | i2v tag-based | 1.223 |3    | i2v tag-based | 1.200 |4    | i2v tag-based | 1.332 | 
0    | genre tag-based | 1.233 |1    | i2v tag-based | 1.220 |2    | i2v tag-based | 1.209 |3    | i2v tag-based | 1.233 |4    | i2v tag-based | 1.222 | 
0    | genre+i2v tag-based | 1.235 |1    | genre+i2v tag-based | 1.277 |2    | genre+i2v tag-based | 1.259 |3    | genre+i2v tag-based | 1.235 |4    | genre+i2v tag-based | 1.208 | 
0    | i2v tag-based (50T) |  |1    | i2v tag-based (50T) |  |2    | i2v tag-based (50T) |  |3    | i2v tag-based |  |4    | i2v tag-based |  |
0    | genre tag-based (50T) | |1    | genre tag-based (50T) |  |2    | genre tag-based (50T) |  |3    | genre tag-based (50T) |  |4    | genre tag-based (50T) |  | 
0    | genre+i2v tag-based (50T) | 1.191 |1    | genre+i2v tag-based (50T) | 1.188 |2    | genre+i2v tag-based (50T) | 1.183 |3    | genre+i2v tag-based (50T) | 1.177 |4    | genre+i2v tag-based (50T) | 1.176 | 
0    | Gram | 1.332 |1    | Gram | 1.303 |2    | Gram | 1.332 |3    | Gram | 1.329 |4    | Gram | 1.330 | 

item-based: KNN with Means, k=60.

SVD: epochs=100, lr=0.002, reg_all=0.4.

Co-Clustering: n_cltr_u=40, n_cltr_i=40, n_epochs=100.
