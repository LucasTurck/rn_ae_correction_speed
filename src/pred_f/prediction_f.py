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
    def __init__(self, nom_modele, parameters=None):
        self.nom_modele = nom_modele
        self.model = None
        self.parameters = parameters if parameters is not None else {}
        self.X_train = None
        self.y_train = None
        self.y_pred_train = None
        self.r2train = -1
        self.X_test = None
        self.y_test = None
        self.y_pred_test = None
        self.r2test = -1
        # self.init_parameters()

    def init_parameters(self):
        """
        Initialise les paramètres du modèle de prédiction.
        Définit les paramètres par défaut pour l'entraînement et la prédiction.
        """
        self.parameters['E'] = self.parameters.get('E', 125)
        self.parameters['num_sonde_train'] = self.parameters.get('num_sonde_train', 0)
        self.parameters['num_sonde_test'] = self.parameters.get('num_sonde_test', 1)
        self.parameters['nb_training'] = self.parameters.get('nb_training', -1)
        self.parameters['nb_test'] = self.parameters.get('nb_test', 100)
        self.parameters['prediction'] = self.parameters.get('prediction', [0, 0, 0])
        self.parameters['y'] = self.parameters.get('y', 1)

    def set_parameters(self, parameters):
        """
        Met à jour les paramètres du modèle de prédiction.
        Args:
            parameters (dict): Dictionnaire contenant les paramètres à mettre à jour.
        """
        for key, value in parameters.items():
            self.parameters[key] = value
        self.init_parameters()

    def remplissage_donnees(self, train=True):
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
        times, sondes = lire_fichier_U(os.path.join(DATA_DIRECTORY, f"E_{self.parameters['E']}", 'U'))
        if not sondes:
            print("Aucune donnée de sonde trouvée.")
            exit(1)
        
        if train:
            # Utiliser les données d'entraînement
            num_sonde = self.parameters['num_sonde_train']
            nb_samples = self.parameters['nb_training']
        else:
            # Utiliser les données de test
            num_sonde = self.parameters['num_sonde_test']
            nb_samples = self.parameters['nb_test']
        
        # Préparer les données pour l'entraînement
        if num_sonde not in sondes:
            print(f"Sonde {num_sonde} non trouvée dans les données.")
            exit(1)
        col = sondes[num_sonde][1]
        if col is None:
            print(f"Pas de données pour la sonde {num_sonde}.")
            exit(1)

        
        if nb_samples > 0 and nb_samples < len(col):
            # Limiter le nombre d'échantillons d'entraînement
            col = col[:nb_samples]

        # Par exemple, pour prediction = [1,1] (utilise X_{i-1}, X_{i} et y_{i-1})
        p = self.parameters['prediction'][0]
        f = self.parameters['prediction'][1]
        h = self.parameters['prediction'][2]
        X = []
        for i in range(max(p, f, h), len(col)):
            features = [col[i][0]]
            # Ajoute X_{i-p} ... X_{i}
            for k in range(p):
                features.append(col[i-k-1][0])
            # Ajoute X_{i-1-f} ... X_{i-1}
            for k in range(f):
                features.append(col[i-k-1][1])
            # Ajoute X_{i-1-h} ... X_{i-1}
            for k in range(h):
                features.append(col[i-k-1][2])
            X.append(features)
        if train:
            # Entraînement
            self.X_train = np.array(X)
            self.y_train = np.array([col[i][self.parameters['y']] for i in range(max(p, f, h), len(col))])
        else:
            # Test
            self.X_test = np.array(X)
            self.y_test = np.array([col[i][self.parameters['y']] for i in range(max(p, f, h), len(col))])
            self.y_pred_test = None

        # print(f"Nombre d'échantillons d'entraînement : {len(self.X_train)}")
        # print(f"Nombre de caractéristiques d'entraînement : {self.X_train.shape[1]}")
        # print(f"Nombre de cibles d'entraînement : {len(self.y_train)}")
        # print(f"Nombre d'échantillons de test : {len(self.X_test) if self.X_test is not None else 'Aucun'}")
        # print(f"Nombre de caractéristiques de test : {self.X_test.shape[1] if self.X_test is not None else 'Aucune'}")
        if train and len(self.X_train) != len(self.y_train):
            print("remplissage_donnees - Erreur : Le nombre d'échantillons d'entraînement ne correspond pas au nombre de cibles.")
            exit(1)
        if not train and self.X_test is not None and len(self.X_test) != len(self.y_test):
            print("remplissage_donnees - Erreur : Le nombre d'échantillons de test ne correspond pas au nombre de cibles.")
            exit(1)
        if self.X_train is not None and self.X_test is not None and not train:
            if self.X_train.shape[1] != self.X_test.shape[1]:
                print("remplissage_donnees - Erreur : Le nombre de caractéristiques d'entraînement ne correspond pas au nombre de caractéristiques de test.")
                exit(1)
            
    def entrainer(self):
        raise NotImplementedError
    
    def sauvegarder_apprentissage(self, dir_path):
        raise NotImplementedError
    
    def charger_apprentissage(self, dir_path):
        raise NotImplementedError

    def predire(self, X):
        raise NotImplementedError

    def evaluer(self, train=True):
        # print("evaluer - Nombre de prédictions de test :", len(self.y_pred_test) if self.y_pred_test is not None else "Aucune")
        # print("evaluer - Nombre de cibles de test :", len(self.y_test) if self.y_test is not None else "Aucune")
        if train:
            if self.y_pred_train is not None and self.y_train is not None:
                if len(self.y_pred_train) == len(self.y_train):
                    self.r2train = r2_score(self.y_train, self.y_pred_train)
                    return self.r2train
            print("evaluer - Erreur : Le nombre de prédictions d'entraînement ne correspond pas aux cibles.")
            return None
        else:
            if self.y_pred_test is not None and self.y_test is not None:
                if len(self.y_pred_test) == len(self.y_test):
                    self.r2test = r2_score(self.y_test, self.y_pred_test)
                    return self.r2test
            print("evaluer - Erreur : Le nombre de prédictions de test ne correspond pas aux cibles.")
            return None

    def afficher(self):
        raise NotImplementedError


def plot_predictions(X, y, y_pred, title = "Prédictions vs Réel", block=True, ax=None):
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
    # print("plot_predictions - Nombre de prédictions :", len(y_pred))
    # print("plot_predictions - Nombre de cibles :", len(y))
    # print("plot_predictions - Nombre de caractéristiques :", X.shape)
    if ax is None:
        plt.figure(figsize=(8, 6))
    else:
        plt.sca(ax)
    plt.scatter(X, y, color='blue', label='Données réelles', alpha=0.5)
    plt.scatter(X, y_pred, color='red', label='Prédictions', alpha=0.5)
    plt.xlabel("Première composante (u)")
    plt.ylabel("Deuxième composante (v)")
    plt.title(title)
    plt.legend(title=f"R2 = {r2_score(y, y_pred):.5f}")
    plt.grid(True)
    if ax is None:
        plt.tight_layout()
        plt.show()
    else:
        ax.set_title(title)
        ax.legend(title=f"R2 = {r2_score(y, y_pred):.5f}")
        ax.grid(True)
        plt.draw()
    # plt.show(block=block)

def args_to_dict(args):
    """
    Convert command line arguments to a dictionary.
    
    Parameters
    ----------
    args : Namespace
        The parsed command line arguments.
    
    Returns
    -------
    dict
        A dictionary with argument names as keys and their values.
    """
    return {k: v for k, v in vars(args).items() if v is not None}
