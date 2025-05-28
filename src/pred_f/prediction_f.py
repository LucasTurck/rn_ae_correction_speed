import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reed_data import lire_fichier_U
from dir import DATA_DIRECTORY
import numpy as np


"""
Module de prédiction pour les données de sonde.

Ce module définit la classe de base ModelePrediction pour l'entraînement, 
la prédiction et l'évaluation de modèles de régression sur des données de sondes.

"""
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
    
    def remplissage_donnees(self, E = 145, num_sonde = 0, nb_training=-1, nb_test=-1):
        """
        Remplit les données d'entraînement pour la prédiction à partir des fichiers de données de sonde.
        Cette méthode lit les données d'une sonde spécifique à partir d'un fichier, prépare les caractéristiques (features) et les cibles (targets) pour l'entraînement d'un modèle de prédiction, en tenant compte des paramètres de fenêtre de prédiction définis dans `self.prediction`.
        Args:
            e (int, optional): Numéro de l'expérience ou identifiant du dossier de données. Par défaut à 145.
            num_sonde (int, optional): Numéro de la sonde à utiliser pour l'extraction des données. Par défaut à 0.
            nb_training (int, optional): Nombre maximal d'échantillons d'entraînement à utiliser. Si négatif, utilise toutes les données disponibles. Par défaut à -1.
            nb_test (int, optional): Nombre maximal d'échantillons de test à utiliser (non utilisé dans cette méthode). Par défaut à -1.
        Side Effects:
            Remplit les attributs `self.X_train` (features d'entraînement) et `self.y_train` (cibles d'entraînement) sous forme de tableaux NumPy.
        Raises:
            Affiche un message d'erreur et termine le programme si :
                - Aucune donnée de sonde n'est trouvée.
                - La sonde spécifiée n'est pas présente dans les données.
                - Il n'y a pas de données pour la sonde spécifiée.
                - Le nombre d'échantillons d'entraînement ne correspond pas au nombre de cibles.
        """
        # Lire les données
        times, sondes = lire_fichier_U(os.path.join(DATA_DIRECTORY, f'E_{E}', 'U'))
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
        """
        Calculate and return the R² (coefficient of determination) score for the training predictions.
        Returns:
            float or None: The R² score of the training predictions if available and valid,
            otherwise None. Prints an error message if the number of predictions does not
            match the number of targets.
        """
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
    """
    Plots predicted values against true values for a regression task.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features) or (n_samples,)
        Input features. If multidimensional, only the first feature is used for the x-axis.

    y : array-like, shape (n_samples,)
        True target values.

    y_pred : array-like, shape (n_samples,)
        Predicted target values.

    title : str, optional (default="Prédictions vs Réel")
        Title of the plot.

    block : bool, optional (default=True)
        Whether to block execution until the plot window is closed.

    Returns
    -------
    None

    Notes
    -----
    - Displays a scatter plot of the true values and predictions.
    - The R2 score is shown in the legend.
    - Only the first feature of X is used for plotting if X is multidimensional.
    """
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
