from pred_f.prediction_f import ModelePrediction, plot_predictions, args_to_dict
from reed_data import lire_fichier_U
from simu_X_probes import calcul_U_effective, calculate_speed_vector_using_U_eff, calcul_U_effective_tf
from dir import MOD_PERSO_DIRECTORY, DATA_DIRECTORY
import datetime
import hashlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tqdm.keras import TqdmCallback
import re
import json
import os
import tensorflow as tf
import copy
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Ignore TensorFlow warnings
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

NAMES_METRICS = ["mse", "mae"]
NAMES_LOSSES = ["mse", "mae", "huber_loss"]



def get_architecture_by_name(name, architectures):
    """
    Retrieve the layer configuration of a neural network architecture by its name.

    Args:
        name (str): The name of the architecture to search for.
        architectures (list of dict): A list of architecture dictionaries, each containing at least
            a "name" key and a "layers" key.

    Returns:
        list: The list of layers corresponding to the architecture with the given name.

    Raises:
        ValueError: If no architecture with the specified name is found in the list.
    """
    for arch in architectures:
        if arch["name"] == name:
            return arch["layers"]
    raise ValueError(f"Architecture '{name}' non trouvée.")

def get_run_id(parameters):
    # Utilise un hash court des paramètres pour l’identifiant
    param_str = json.dumps(parameters, sort_keys=True)
    hash_id = hashlib.md5(param_str.encode()).hexdigest()[:8]
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{date_str}_{hash_id}"

class ModeleReseauNeurones(ModelePrediction):
    """
    Classe pour entraîner et utiliser un modèle de réseau de neurones pour la prédiction.
    Attributs
    ----------
    architecture : str
        Le nom de l'architecture à utiliser, par défaut "dense_simple".
    parametres : dict
        Dictionnaire contenant les paramètres d'entraînement (epochs, batch_size).
    reseau : keras.Model
        Le modèle de réseau de neurones construit.
    model : keras.callbacks.History
        L'historique de l'entraînement du modèle.
    y_pred_train : np.ndarray
        Prédictions du modèle sur les données d'entraînement.
    y_pred_test : np.ndarray
        Prédictions du modèle sur les données de test.
    Méthodes
    --------
    create_reseau():
        Crée le réseau de neurones selon l'architecture spécifiée dans le fichier 'architectures.json'.
    entrainer():
        Compile et entraîne le réseau de neurones sur les données d'entraînement.
    predire():
        Effectue des prédictions sur les données de test si elles sont disponibles.
    afficher():
        Affiche les prédictions du modèle sur les ensembles d'entraînement et de test à l'aide de graphiques.
    """
    def __init__(self, parameters=None):
        super().__init__('reseau_neurones', parameters)
        self.history = None
        self.run_path = None

    def clear(self):
        self.model = None
        self.data_train = None
        self.X_train = None
        self.y_train = None
        self.data_test = None
        self.X_test = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.history = None
        import gc
        gc.collect()
        tf.keras.backend.clear_session()

    def __str__(self):
        return f"(architecture={self.parameters['architecture']}, epochs={self.parameters['epochs']}, batch_size={self.parameters['batch_size']})"
    
    def copy_model(self):
        new_model = ModeleReseauNeurones()
        new_model = super().copy_model(new_model)
        new_model.create_reseau()
        new_model.model = keras.models.clone_model(self.model)
        new_model.model.set_weights(self.model.get_weights())
        new_model.history = copy.deepcopy(self.history) if self.history is not None else None
        return new_model

    def init_parameters(self):
        super().init_parameters()
        self.parameters["architecture"] = self.parameters.get("architecture", "dense_simple")
        self.parameters["epochs"] = self.parameters.get("epochs", 100)
        self.parameters["batch_size"] = self.parameters.get("batch_size", 64)
        self.parameters['loss'] = self.parameters.get('loss', 'mse')
        self.parameters['target_shape'] = self.parameters.get('target_shape', (self.parameters['timesteps'], 1))
        # print(f"Paramètres du modèle : {self.parameters}")

    def remplissage_donnees(self, train=True, simulate=False):
        
        data = self.simulate_probes(train=train, simulate=simulate)

        X, y = [], []
        p_start, f_start, h_start = self.parameters['prediction_start']
        p_end, f_end, h_end = self.parameters['prediction_end']
        timesteps = self.parameters['timesteps']
        y_index = self.parameters['y']
        n_start = max(p_start + (p_end - p_start) * timesteps,
                      f_start + (f_end - f_start) * timesteps,
                      h_start + (h_end - h_start) * timesteps)
        for i in range(n_start, len(data)):
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
            X.append(feature_vector[::-1])  # Inverser l'ordre des features
            y.append(data[i][y_index])  # Prédiction de w ou v selon y_index
        
        if train:
            self.X_train = np.array(X, dtype=np.float32)
            self.y_train = np.array(y, dtype=np.float32)
        else:
            self.X_test = np.array(X, dtype=np.float32)
            self.y_test = np.array(y, dtype=np.float32)

        print(f"Données {'d entraînement' if train else 'de test'} chargées : {len(X)} échantillons, forme X = {self.X_train.shape if train else self.X_test.shape}, forme y = {self.y_train.shape if train else self.y_test.shape}")

        # Vérifier que les données sont correctement remplies
        if train:
            if len(self.X_train) != len(self.y_train):
                raise ValueError("Les données d'entraînement X et y doivent avoir la même longueur.")
        if not train:
            if len(self.X_test) != len(self.y_test):
                raise ValueError("Les données de test X et y doivent avoir la même longueur.")
        if self.X_train is not None and self.X_test is not None:
            if self.X_train.shape[1] != self.X_test.shape[1]:
                print(f"Attention : Les données d'entraînement et de test ont des formes différentes : {self.X_train.shape} vs {self.X_test.shape}")
                raise ValueError("Les données d'entraînement et de test doivent avoir la même forme.")

    def simulate_probes(self, train=True, simulate=False):
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
        
        if train:
            self.data_train = data
        else:
            self.data_test = data
        if nb_samples > 0 and nb_samples < len(data):
            # Limiter le nombre d'échantillons d'entraînement
            data = data[:nb_samples]
        
        if not simulate:
            # Si on ne simule pas, on utilise les données réelles
            return data

        U_2_eff_1, U_2_eff_2 = calcul_U_effective(data, self.parameters['k'], self.parameters['phi'], self.parameters['U_mean'], b_coord=self.parameters['y'])

        u, v, w = calculate_speed_vector_using_U_eff(U_2_eff_1, U_2_eff_2, self.parameters['k'], self.parameters['phi'], self.parameters['U_mean'], b_coord=self.parameters['y'])

        print(f"Nombre d'échantillons simulés : {len(u)}")
        return list(zip(u, v, w))  # Retourne une liste de tuples (u, v, w) pour chaque échantillon

    def build_model(self, architecture):
        """
        Builds a Keras model based on the provided architecture.
        
        Args:
            architecture (list): A list of dictionaries defining the layers of the model.
        """
        if architecture is None:
            raise ValueError("L'architecture ne peut pas être None.")
        
        self.model = keras.Sequential()
        self.model.add(layers.Input(shape=self.X_train.shape[1:]))
        
        for layer in architecture:
            layer_type = layer["type"]
            if layer_type == "Dense":
                self.model.add(layers.Dense(layer["units"], activation=layer.get("activation")))
            elif layer_type == "Conv1D":
                self.model.add(layers.Conv1D(filters=layer["filters"], 
                                            kernel_size=layer["kernel_size"],
                                            activation=layer.get("activation"), 
                                            padding=layer.get("padding", "valid")))
            elif layer_type == "Flatten":
                self.model.add(layers.Flatten())
            elif layer_type == "Reshape":
                target_shape = layer.get("target_shape")
                if target_shape is None:
                    self.model.add(layers.Reshape(self.parameters.get("target_shape", (self.X_train.shape[1], 1))))
                elif isinstance(target_shape, (list, tuple)):
                    self.model.add(layers.Reshape(tuple(target_shape)))
            elif layer_type == "Dropout":
                self.model.add(layers.Dropout(layer.get("rate", 0.5)))
            elif layer_type == "BatchNormalization":
                self.model.add(layers.BatchNormalization())
            elif layer_type == "LSTM":
                self.model.add(layers.LSTM(layer["units"], 
                                        activation=layer.get("activation"), 
                                        return_sequences=layer.get("return_sequences", False)))
            elif layer_type == "GRU":
                self.model.add(layers.GRU(layer["units"], 
                                        activation=layer.get("activation"), 
                                        return_sequences=layer.get("return_sequences", False)))
            elif layer_type == "Bidirectional":
                inner = dict(layer["layer"])  # Copie pour ne pas modifier l'original
                inner_type = inner.pop("type")  # Retire "type"
                if inner_type == "LSTM":
                    inner_layer = layers.LSTM(**inner)
                elif inner_type == "GRU":
                    inner_layer = layers.GRU(**inner)
                else:
                    raise ValueError(f"Type de couche bidirectionnelle non supporté: {inner_type}")
                self.model.add(layers.Bidirectional(inner_layer))
            elif layer_type == "Add":
                # Les couches Add nécessitent un modèle fonctionnel, pas séquentiel.
                raise NotImplementedError("La couche 'Add' nécessite un modèle fonctionnel (Functional API).")

    def create_reseau(self):
        if self.X_train is None:
            print("Erreur : X_train n'est pas défini.")
        
        json_path = os.path.join(os.path.dirname(__file__), "architectures.json")
        with open(json_path) as f:
            architectures = json.load(f)

        architecture = get_architecture_by_name(self.parameters["architecture"], architectures)

        if architecture is None:
            print(f"Erreur : L'architecture '{architecture}' n'est pas définie dans 'architectures.json'.")

        self.build_model(architecture)


    def physical_loss(self, y_true, y_pred):
        shape = tf.shape(y_pred)[0] - 1
        diff = tf.zeros([shape], dtype=y_pred.dtype)
        diff = y_pred[1:] - y_pred[:-1]
        diff_2 = (y_pred[2:] - 2 * y_pred[1:-1] + y_pred[:-2])/2
        return tf.reduce_mean(tf.square(diff)) + tf.reduce_mean(tf.square(diff_2))

    def combined_loss(self, y_true, y_pred):
        if self.parameters['loss'] == 'mse':
            loss_data = tf.reduce_mean(tf.square(y_true - y_pred))  # MSE
        elif self.parameters['loss'] == 'mae':
            loss_data = tf.reduce_mean(tf.abs(y_true - y_pred))
        loss_phys = self.physical_loss(y_true, y_pred)
        loss_phys = 0
        return loss_data + 0.1 * loss_phys


    def entrainer(self):
        self.model.compile(optimizer='adam', loss=self.combined_loss, metrics=NAMES_METRICS)
        self.history = self.model.fit(
            self.X_train, self.y_train, 
            epochs=self.parameters["epochs"], 
            batch_size=self.parameters["batch_size"], 
            verbose=0,
            callbacks=[TqdmCallback(verbose=1)])
        self.y_pred_train = self.model.predict(self.X_train).flatten()

    def predire(self, test = True):
        if test:
            if self.X_test is not None:
                self.y_pred_test = self.model.predict(self.X_test).flatten()
            else:
                self.y_pred_test = None
        else:
            if self.X_train is not None:
                self.y_pred_train = self.model.predict(self.X_train).flatten()
            else:
                self.y_pred_train = None
        # print(f"Nombre de données d'entraînement : {self.X_train.shape[0]}")
        # print(f"Nombre de données de prediction d'entraînement : {self.y_pred_train.shape[0] if self.y_pred_train is not None else 0}")

        # print(f"Nombre de données de test : {self.X_test.shape[0] if self.X_test is not None else 0}")
        # print(f"Nombre de données de prediction de test : {self.y_pred_test.shape[0] if self.y_pred_test is not None else 0}")

    def afficher(self):
        plot_predictions(self.X_train, self.y_train, self.y_pred_train, 
                         f"Train - RdN : {self.parameters['architecture']}, predictions : {self.parameters['prediction']}, epochs = {self.parameters['epochs']}, batch_size = {self.parameters['batch_size']}", block=False)
        if self.X_test is not None and self.y_test is not None:
            plot_predictions(self.X_test, self.y_test, self.y_pred_test, 
                             f"Test - RdN : {self.parameters['architecture']}, predictions : {self.parameters['prediction']}, epochs = {self.parameters['epochs']}, batch_size = {self.parameters['batch_size']}", block=True)

    def sauvegarder_apprentissage(self, dossier="../mod"):
        """
        Sauvegarde les poids du modèle et l'historique d'entraînement dans un dossier.
        """
        run_id = get_run_id(self.parameters)
        dossier = os.path.join(dossier, f"{self.parameters['architecture']}", run_id)
        if not os.path.exists(dossier):
            os.makedirs(dossier)
        
        self.run_path = dossier  # Stocker le chemin du run pour une utilisation ultérieure

        # Sauvegarde des poids
        chemin_modele = os.path.join(dossier, "poids.h5")
        self.model.save_weights(chemin_modele)

        # Sauvegarde de l'historique d'entraînement
        chemin_historique = os.path.join(dossier, "historique.json")
        if self.model is not None:
            with open(chemin_historique, "w") as f:
                json.dump(self.history.history, f, indent=2)
        
                
        # Reformater les listes sur une ligne
        with open(chemin_historique, "r") as f:
            contenu = f.read()

        # Remplace les listes multilignes par une seule ligne
        contenu = re.sub(r'\[\s+([^\]]+?)\s+\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']', contenu)

        with open(chemin_historique, "w") as f:
            f.write(contenu)
        
        # Sauvegarde des paramètres du modèle
        chemin_parametres = os.path.join(dossier, "config.json")
        with open(chemin_parametres, "w") as f:
            json.dump(self.parameters, f, indent=2)
        
        #sauvegarde des scores R²
        scores = {
            "r2train": float(self.r2train) if self.r2train is not None else None,
            "r2test": float(self.r2test) if self.r2test is not None else None
        }
        chemin_scores = os.path.join(dossier, "scores.json")
        with open(chemin_scores, "w") as f:
            json.dump(scores, f, indent=2)

        print(f"Modèle et historique sauvegardés dans {dossier}")

    def charger_apprentissage(self, dossier):
        """
        Charge les poids, les paramètres et l'historique d'un modèle sauvegardé.
        """
        self.clear()  # Nettoyer l'état actuel du modèle
        self.run_path = dossier  # Stocker le chemin du run pour une utilisation ultérieure
        
        # Charger les paramètres du modèle
        chemin_parametres = os.path.join(dossier, "config.json")
        with open(chemin_parametres, "r") as f:
            self.parameters = json.load(f)
        self.init_parameters()
        
        # Récupérer les données d'entraînement
        self.remplissage_donnees(train=True)
    
        # Recréer l'architecture et le modèle
        json_path = os.path.join(os.path.dirname(__file__), "architectures.json")
        with open(json_path) as f:
            architectures = json.load(f)
        architecture = get_architecture_by_name(self.parameters["architecture"], architectures)
        self.build_model(architecture)
    
        # Charger les poids
        chemin_modele = os.path.join(dossier, "poids.h5")
        self.model.load_weights(chemin_modele)
        
        self.predire(test = False)  # Prédire les valeurs sur les données d'entraînement
    
        # Charger l'historique d'entraînement (optionnel)
        chemin_historique = os.path.join(dossier, "historique.json")
        if os.path.exists(chemin_historique):
            import types
            with open(chemin_historique, "r") as f:
                history_dict = json.load(f)
            # Créer un objet factice avec un attribut .history
            self.history = types.SimpleNamespace(history=history_dict)

        # Charger les scores R² (optionnel)
        chemin_scores = os.path.join(dossier, "scores.json")
        if os.path.exists(chemin_scores):
            with open(chemin_scores, "r") as f:
                scores = json.load(f)
                self.r2train = scores.get("r2train")
                self.r2test = scores.get("r2test")

    def afichage_historique(self, erreur = 'loss'):
        """
        Affiche l'historique d'entraînement du modèle.
        """
        if self.history is not None:
            if erreur not in self.history.history:
                print(f"Erreur : '{erreur}' n'est pas une clé valide dans l'historique d'entraînement.")
                return
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(self.history.history[erreur], label=erreur)
            plt.title('Historique d\'entraînement')
            plt.xlabel('Epochs')
            plt.ylabel('Valeurs')
            plt.yscale('log')
            plt.legend()
            plt.show(block = False)
        else:
            print("Aucun historique d'entraînement disponible.")

