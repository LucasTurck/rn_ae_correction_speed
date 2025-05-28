# rn_ae_correction_speed
Apprentissage non supervisé pour la correction des mesures de vitesse en écoulement turbulent  par anémométrie thermique deux composantes

# Installation 
Avant de lancer le script, il faut installer les bibliothèques nécessaires (le script a été testé avec Python 3.8.10) :

- tensorflow
    pip install tensorflow
- keras
    pip install keras
- tqdm ( pour la barre de progression)
    pip install tqdm

# Lancement du script
Pour lancer le script, il suffit de se placer dans le repertoire du projet : rn_ae_correction_speed et de lancer la commande suivante :

- pour l'apprentissage du réseau de neurones
    python .\src\pred_f\reseau_neurones.py
- pour le random forest
    python .\src\pred_f\random_forest.py

## paramètres
tous les paramètres sont à rajouter après la commande python avec '--' avant le nom du paramètre, par exemple :
python .\src\pred_f\reseau_neurones.py --nb_training 100

- nb_training : taille de l'échantillon d'apprentissage (par défaut 100)
- E : nom de la base de donnée ( par défaut '145')
- num_sonde : numéro du sonde (par défaut 0)
- epochs : nombre d'epochs pour l'apprentissage du réseau de neurones (par défaut 100)
- batch_size : taille du batch pour l'apprentissage du réseau de neurones (par défaut 64)
- architecture : architecture du réseau de neurones (par défaut 'simple_dense')