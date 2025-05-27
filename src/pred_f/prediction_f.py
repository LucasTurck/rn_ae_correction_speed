import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reed_data import lire_fichier_U
from dir import DATA_DIRECTORY
import numpy as np

class ModelePrediction:
    def __init__(self, nom_modele, prediction = [0,0], X_train = None, y_train = None, X_test=None, y_test=None):
        self.nom_modele = nom_modele
        self.prediction = prediction
        self.X_train = X_train
        self.y_train = y_train
        self.r2train = -1
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.y_pred_train = None
        self.y_pred_test = None
    
    def remplissage_donnees(self, e = 145, num_sonde = 0, nb_training=-1, nb_test=-1):
        # Lire les données
        times, sondes = lire_fichier_U(os.path.join(DATA_DIRECTORY, f'E_{e}', 'U'))
        if not sondes:
            print("Aucune donnée de sonde trouvée.")
            exit(1)
        
        # Préparer les données pour l'entraînement
        if num_sonde not in sondes:
            print(f"Sonde {num_sonde} non trouvée dans les données.")
            exit(1)
        pos, col = sondes[num_sonde]
        if col is None:
            print(f"Pas de données pour la sonde {num_sonde}.")
            exit(1)
        
        if nb_training > 0 and nb_training < len(col):
            # Limiter le nombre d'échantillons d'entraînement
            col = col[:nb_training]
            
        # Par exemple, pour prediction = [1,1] (utilise X_{i-1}, X_{i} et y_{i-1})
        p = self.prediction[0]
        f = self.prediction[1]
        X_train = []
        for i in range(max(p, f), len(col)):
            features = [col[i][0]]
            # Ajoute X_{i-p} ... X_{i}
            for k in range(p):
                features.append(col[i-k-1][0])
            # Ajoute X_{i-1-f} ... X_{i-1}
            for k in range(f):
                features.append(col[i-k-1][1])
            X_train.append(features)
        self.X_train = np.array(X_train)
        self.y_train = np.array([col[i][1] for i in range(max(p, f), len(col))])
        
        # print(f"Nombre d'échantillons d'entraînement : {len(self.X_train)}")
        # print(f"Nombre de caractéristiques d'entraînement : {self.X_train.shape[1]}")
        # print(f"Nombre de cibles d'entraînement : {len(self.y_train)}")
        if len(self.X_train) != len(self.y_train):
            print("Erreur : Le nombre d'échantillons d'entraînement ne correspond pas aux cibles.")
            exit(1)
    
    def entrainer(self):
        raise NotImplementedError

    def predire(self, X):
        raise NotImplementedError
    
    def R2entrainement(self):
        if self.y_pred_train is not None:
            
            if len(self.y_pred_train) != len(self.y_train):
                print("Erreur : Le nombre de prédictions d'entraînement ne correspond pas aux cibles.")
                return None
            # print("Nombre de prédictions d'entraînement :", len(self.y_pred_train))
            # print("Nombre de cibles d'entraînement :", len(self.y_train))
            self.r2train = r2_score(self.y_train, self.y_pred_train)
            return self.r2train
        return None

    def evaluer(self):
        if self.X_test is not None and self.y_test is not None:
            y_pred = self.predire(self.X_test)
            return r2_score(self.y_test, y_pred)
        return None

    def afficher(self):
        raise NotImplementedError


def plot_predictions(X, y, y_pred, title = "Prédictions vs Réel", block=True):
    if len(X[0]) > 1:
        X = X[:, 0]  # Utiliser seulement la première composante pour l'affichage
    # print("Nombre de prédictions :", len(y_pred))
    # print("Nombre de cibles :", len(y))
    # print("Nombre de caractéristiques :", X.shape)
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Données réelles', alpha=0.5)
    plt.scatter(X, y_pred, color='red', label='Prédictions', alpha=0.5)
    plt.xlabel("Première composante (X)")
    plt.ylabel("Deuxième composante (Y)")
    plt.title(title)
    plt.legend(title=f"R2 = {r2_score(y, y_pred):.3f}")
    plt.grid(True)
    plt.show(block=block)
        
def scatter_plot(X, y, sonde_num):
    # Affichage d'un nuage de points pour la sonde choisie
    plt.figure(figsize=(6, 6))
    plt.scatter(X, y, alpha=0.5, label="Données réelles")
    plt.xlabel("Première composante (X)")
    plt.ylabel("Deuxième composante (Y)")
    plt.title(f"Sonde {sonde_num} : relation X/Y")
    plt.legend()
    plt.grid(True)
    plt.show()
