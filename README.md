# RecSteam
Recommendation Systems 2020.2 PPGCC/UFMG - Final Project

Dataset: https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data <br>
Steam API: https://developer.valvesoftware.com/wiki/Steam_Web_API

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
