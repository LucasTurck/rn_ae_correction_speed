import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reed_data import lire_fichier_U
from dir import DATA_DIRECTORY
import numpy as np
import copy


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Évite la division par zéro
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

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

    def copy_model(self, new_model = None):
        """
        Copie les paramètres du modèle de prédiction dans un nouveau modèle.
        Retourne une nouvelle instance de ModelePrediction avec les mêmes paramètres.
        """
        parameters_copy = self.copy_parameters()
        new_model.set_parameters(parameters_copy)
        new_model.X_train = copy.deepcopy(self.X_train)
        new_model.y_train = copy.deepcopy(self.y_train)
        new_model.y_pred_train = copy.deepcopy(self.y_pred_train)
        new_model.r2train = self.r2train
        new_model.X_test = copy.deepcopy(self.X_test)
        new_model.y_test = copy.deepcopy(self.y_test)
        new_model.y_pred_test = copy.deepcopy(self.y_pred_test)
        new_model.r2test = self.r2test

        return new_model

    def copy_parameters(self):
        new_parameters = {copy.deepcopy(key): copy.deepcopy(value) for key, value in self.parameters.items()}
        return new_parameters

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
        self.parameters['prediction_start'] = self.parameters.get('prediction_start', [0, 0, 1])
        self.parameters['prediction_end'] = self.parameters.get('prediction_end', [1, 0, 2])
        self.parameters['timesteps'] = self.parameters.get('timesteps', 5)
        self.parameters['y'] = self.parameters.get('y', 2)

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
        
        # lire les données d'entraînement ou de test
        times, sondes = lire_fichier_U(os.path.join(DATA_DIRECTORY, f"E_{self.parameters['E']}", 'U'))
        if not sondes:
            raise ValueError(f"Aucune donnée trouvée pour E_{self.parameters['E']} dans le fichier U.")
        
        if train:
            num_sonde = self.parameters.get("num_sonde_train", 0)
            nb_samples = self.parameters.get("nb_training", -1)
        else:
            num_sonde = self.parameters.get("num_sonde_test", 0)
            nb_samples = self.parameters.get("nb_test", -1)
        
        if num_sonde not in sondes:
            raise ValueError(f"La sonde {num_sonde} n'est pas disponible dans les données pour E_{self.parameters['E']}.")
        
        # Sélectionner les données de la sonde spécifiée
        data = sondes[num_sonde][1]
        if data is None or len(data) == 0:
            raise ValueError(f"Aucune donnée disponible pour la sonde {num_sonde} dans E_{self.parameters['E']}.")
        
        if nb_samples > 0 and nb_samples < len(data):
            # Limiter le nombre d'échantillons d'entraînement
            data = data[:nb_samples]
            
        X, y = [], []
        p_start, f_start, h_start = self.parameters['prediction_start']
        p_end, f_end, h_end = self.parameters['prediction_end']
        timesteps = self.parameters['timesteps']
        y_index = self.parameters['y']
        for i in range(max(p_start, f_start, h_start) * timesteps, len(data)):
            feature_vector = []
            for j in range(timesteps):
                timesteps_vector = []
                for k in range(p_start, p_end):
                    timesteps_vector.append(data[i - j - k][0])  # u
                for k in range(f_start, f_end):
                    timesteps_vector.append(data[i - j - k][1])  # v
                for k in range(h_start, h_end):
                    timesteps_vector.append(data[i - j - k][2])  # w
                feature_vector.append(timesteps_vector)
            X.append(feature_vector)  # Inverser l'ordre des features
            y.append(data[i][y_index])  # Prédiction de w ou v selon y_index
        
        if train:
            self.X_train = np.array(X, dtype=np.float32)
            self.y_train = np.array(y, dtype=np.float32)
        else:
            self.X_test = np.array(X, dtype=np.float32)
            self.y_test = np.array(y, dtype=np.float32)

        print(f"Données {'d entraînement' if train else 'de test'} chargées : {len(X)} échantillons, forme X = {self.X_train.shape if train else self.X_test.shape}, forme y = {self.y_train.shape if train else self.y_test.shape}")

    def remplissage_donnees2(self, train=True):
        raise NotImplementedError("La méthode remplissage_donnees doit être implémentée dans la sous-classe.")
    
    def entrainer(self):
        raise NotImplementedError
    
    def sauvegarder_apprentissage(self, dir_path):
        raise NotImplementedError
    
    def charger_apprentissage(self, dir_path):
        raise NotImplementedError

    def predire(self, X=None):
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
    n = 0
    if len(X[0]) > 1:
        X = X[:, n]  # Utiliser seulement la première composante pour l'affichage
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
