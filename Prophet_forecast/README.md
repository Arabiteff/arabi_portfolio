# Prophet prévisions pour les préparation des collis(en cours de dev)
![](https://i.ibb.co/9hdDxwv/Capture-d-cran-2024-04-27-114847.png)
## la stratégie du modèle:
- On considère ici que chaque entrepôt est une série agrégée au niveau enseigne par P,entrepôt.
- Chaque entrepôt prend un model différent le modèle s’appelle Prophet (auto),
- Le model s'entraînent pour trouver à chaque fois (freq) la meilleur combinaison de paramètres (tendance et saisonnalité…), pour chaque entrepôt tout cela se fait par deux algo:
- le 1er fichier c’est un backtest pour choisir le meilleur modèle pour chaque entrepôt.
- le 2ème fichier c’est algorithme qui prend les résultats du 1er fichier et applique chaque modèle pour chaque entrepôt pour avoirs les prévision pour un horizon donnée (s+1,s+2 …)

le schéma suivanat explique la stratégie:
![](https://i.ibb.co/DDmdKxM/Capture-d-cran-2024-04-27-114301.png)

la solution est en cours de dev, bigQuery remplacera la partie Excel







