from pred_f.prediction_f import ModelePrediction, plot_predictions, args_to_dict
from dir import MOD_PERSO_DIRECTORY
import datetime
import hashlib
try:
    import intel_extension_for_tensorflow as itex
except ImportError:
    print("intel_extension_for_tensorflow non installé, optimisation Intel désactivée.")
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback
import re
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Ignore TensorFlow warnings
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

NAMES_METRICS = ["mse", "mae"]

def build_model(input_shape, layer_defs):
    """
    Builds a Keras Sequential model based on the provided input shape and layer definitions.

    Args:
        input_shape (tuple): Shape of the input data (excluding batch size).
        layer_defs (list of dict): List of dictionaries, each specifying a layer to add to the model.
            Each dictionary must have a "type" key indicating the layer type (e.g., "Dense", "Conv1D", "Flatten", "Reshape").
            Additional keys specify layer-specific parameters:
                - For "Dense": "units" (int), "activation" (str, optional)
                - For "Conv1D": "filters" (int), "kernel_size" (int), "activation" (str, optional), "padding" (str, optional)
                - For "Flatten": no additional parameters
                - For "Reshape": "target_shape" (tuple or int or list)

    Returns:
        keras.Sequential: The constructed Keras Sequential model.

    Raises:
        KeyError: If required keys for a layer type are missing in the layer definition.
        ValueError: If an unsupported layer type is specified in layer_defs.

    Note:
        Extend the function to support additional layer types as needed.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    for layer in layer_defs:
        if layer["type"] == "Dense":
            model.add(layers.Dense(layer["units"], activation=layer.get("activation")))
        elif layer["type"] == "Conv1D":
            model.add(layers.Conv1D(
                filters=layer["filters"],
                kernel_size=layer["kernel_size"],
                activation=layer.get("activation"),
                padding=layer.get("padding", "valid")
            ))
        elif layer["type"] == "Flatten":
            model.add(layers.Flatten())
        elif layer["type"] == "Reshape":
            if not isinstance(layer["target_shape"], list):
                model.add(layers.Reshape((input_shape[0], layer["target_shape"])))
            else:
                model.add(layers.Reshape(tuple(layer["target_shape"])))
        # Ajoute d'autres types de couches si besoin
    return model

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
        
    def init_parameters(self):
        super().init_parameters()
        self.parameters["architecture"] = self.parameters.get("architecture", "dense_simple")
        self.parameters["epochs"] = self.parameters.get("epochs", 100)
        self.parameters["batch_size"] = self.parameters.get("batch_size", 64)
        self.parameters['loss'] = self.parameters.get('loss', 'mse')
        print(f"Paramètres du modèle : {self.parameters}")

    def create_reseau(self):
        if self.X_train is None:
            print("Erreur : X_train n'est pas défini.")
        
        json_path = os.path.join(os.path.dirname(__file__), "architectures.json")
        with open(json_path) as f:
            architectures = json.load(f)

        architecture = get_architecture_by_name(self.parameters["architecture"], architectures)

        if architecture is None:
            print(f"Erreur : L'architecture '{architecture}' n'est pas définie dans 'architectures.json'.")
            
        self.model = build_model(self.X_train.shape[1:], architecture)

    def entrainer(self):
        self.model.compile(optimizer='adam', loss=self.parameters['loss'], metrics=NAMES_METRICS)
        self.history = self.model.fit(
            self.X_train, self.y_train, 
            epochs=self.parameters["epochs"], 
            batch_size=self.parameters["batch_size"], 
            verbose=0,
            callbacks=[TqdmCallback(verbose=1)])
        self.y_pred_train = self.model.predict(self.X_train).flatten()

    def predire(self):
        if self.X_test is not None:
            self.y_pred_test = self.model.predict(self.X_test).flatten()
        else:
            self.y_pred_test = None

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
        # Charger les paramètres du modèle
        chemin_parametres = os.path.join(dossier, "config.json")
        with open(chemin_parametres, "r") as f:
            self.parameters = json.load(f)
        self.init_parameters()
    
        # Recréer l'architecture et le modèle
        json_path = os.path.join(os.path.dirname(__file__), "architectures.json")
        with open(json_path) as f:
            architectures = json.load(f)
        architecture = get_architecture_by_name(self.parameters["architecture"], architectures)
        self.model = build_model(self.X_train.shape[1:], architecture)
    
        # Charger les poids
        chemin_modele = os.path.join(dossier, "poids.h5")
        self.model.load_weights(chemin_modele)
    
        # Charger l'historique d'entraînement (optionnel)
        chemin_historique = os.path.join(dossier, "historique.json")
        if os.path.exists(chemin_historique):
            with open(chemin_historique, "r") as f:
                self.history = json.load(f)
    
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
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Réseau de Neurones Model Training and Evaluation')
    parser.add_argument('--config', type=str, help='Path to the configuration file for the model parameters')
    parser.add_argument('--load', type=str, help='Path to the directory containing a saved model to load')
    parser.add_argument('--E', type=int, help='Experiment number to load data for')
    parser.add_argument('--num_sonde_train', type=int, help='Number of the probes to use in the model')
    parser.add_argument('--num_sonde_test', type=int, help='Number of the probes to use in the model for testing')
    parser.add_argument('--nb_training', type=int, help='Number of training samples to use, -1 for all')
    parser.add_argument('--nb_test', type=int, help='Number of test samples to use, -1 for all')
    parser.add_argument('--prediction', type=int, nargs=2, help='Prediction parameters [p, f] where p is the number of previous samples of u and f is the number of previous samples of v to use')
    parser.add_argument('--architecture', type=str, help='Name of the neural network architecture to use')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--y', type=int, help='Index of the column to predict (0 for u, 1 for v)')
    args = parser.parse_args()
    args_dict = args_to_dict(args)
    
    # Charger les paramètres depuis un fichier si fourni
    if args.config:
        with open(args.config, "r") as f:
            file_params = json.load(f)
        # Fusionner les paramètres du fichier et ceux de la ligne de commande (priorité à la ligne de commande)
        file_params.update({k: v for k, v in args_dict.items() if v is not None})
        params = file_params
    else:
        params = args_dict

    # Créer
    model_nn = ModeleReseauNeurones(parameters=params)
    # Remplir les données d'entraînement
    model_nn.remplissage_donnees()

    model_nn.create_reseau()

    model_nn.entrainer()


    print(f"R² pour l'entraînement : {model_nn.evaluer(train=True)}")
    
    # Remplir les données de test
    model_nn.remplissage_donnees(train=False)

    # Prédire les valeurs sur les données de test
    model_nn.predire()

    print(f"R² pour le test : {model_nn.evaluer(train=False)}")

    # Sauvegarder l'apprentissage
    model_nn.sauvegarder_apprentissage(MOD_PERSO_DIRECTORY)
    
    # Afficher l'historique d'entraînement
    model_nn.afichage_historique()
    
    # Afficher les résultats
    model_nn.afficher()