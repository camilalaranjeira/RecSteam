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

item-based: KNN with Means, k=40.

SVD: epochs=100, lr=0.002, reg_all=0.4.

SVD-spotlight: epochs=10, learning_rate=1e-3, l2=1e-9, embedding_dim=128, batch_size=1024,.


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
