# Prophet prévisions pour les préparation des collis(en cours de dev)
![](https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_1455/https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_300/https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_512/https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_768/https://www.relataly.com/wp-content/uploads/2023/03/stock-market-forecasting-python-relataly-midjourney-3-min.png)
## la stratégie du modèle:
- On considère ici que chaque entrepôt est une série agrégée au niveau enseigne par P,entrepôt.
- Chaque entrepôt prend un model différent le modèle s’appelle Prophet (auto),
- Le model s'entraînent pour trouver à chaque fois (freq) la meilleur combinaison de paramètres (tendance et saisonnalité…), pour chaque entrepôt tout cela se fait par deux algo:
- le 1er fichier c’est un backtest pour choisir le meilleur modèle pour chaque entrepôt.
- le 2ème fichier c’est algorithme qui prend les résultats du 1er fichier et applique chaque modèle pour chaque entrepôt pour avoirs les prévision pour un horizon donnée (s+1,s+2 …)

le schéma suivanat explique la stratégie:
![](https://i.ibb.co/DDmdKxM/Capture-d-cran-2024-04-27-114301.png)

la solution est en cours de dev, bigQuery remplacera la partie Excel







