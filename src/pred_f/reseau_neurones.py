from prediction_f import ModelePrediction, plot_predictions
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Ignore TensorFlow warnings


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
    def __init__(self, prediction, architecture = "dense_simple", epochs=1000, batch_size=64):
        super().__init__('reseau_neurones', prediction)
        self.architecture = architecture
        self.parametres = {
            "epochs": epochs,
            "batch_size": batch_size
        }

    def create_reseau(self):
        if self.X_train is None:
            print("Erreur : X_train n'est pas défini.")
        
        json_path = os.path.join(os.path.dirname(__file__), "architectures.json")
        with open(json_path) as f:
            architectures = json.load(f)
            
        self.architecture = get_architecture_by_name(self.architecture, architectures)
        
        if self.architecture is None:
            print(f"Erreur : L'architecture '{self.architecture}' n'est pas définie dans 'architectures.json'.")
            
        self.reseau = build_model(self.X_train.shape[1:], self.architecture)

    def entrainer(self):
        self.reseau.compile(optimizer='adam', loss='mse')
        self.model = self.reseau.fit(self.X_train, self.y_train, epochs=self.parametres["epochs"], batch_size=self.parametres["batch_size"], verbose=0, 
                                  callbacks=[TqdmCallback(verbose=1)])
        self.y_pred_train = self.reseau.predict(self.X_train).flatten()

    def predire(self):
        if self.X_test is not None:
            self.y_pred_test = self.model.predict(self.X_test).flatten()
        else:
            self.y_pred_test = None

    def afficher(self):
        plot_predictions(self.X_train, self.y_train, self.y_pred_train, "Réseau de Neurones - Train")
        if self.X_test is not None and self.y_test is not None:
            plot_predictions(self.X_test, self.y_test, self.y_pred_test, "Réseau de Neurones - Test")
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Réseau de Neurones Model Training and Evaluation')
    parser.add_argument('--E', type=int, default=145, help='Experiment number to load data for')
    parser.add_argument('--num_sonde', type=int, default=0, help='Number of the probes to use in the model')
    parser.add_argument('--nb_training', type=int, default=100, help='Number of training samples to use, -1 for all')
    parser.add_argument('--nb_test', type=int, default=-1, help='Number of test samples to use, -1 for all')
    parser.add_argument('--prediction', type=int, nargs=2, default=[0, 0], help='Prediction parameters [p, f] where p is the number of previous samples of u and f is the number of previous samples of v to use')
    parser.add_argument('--architecture', type=str, default='dense_simple', help='Name of the neural network architecture to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    args = parser.parse_args()

    # Créer
    model_nn = ModeleReseauNeurones(prediction =args.prediction, architecture=args.architecture, epochs=args.epochs, batch_size=args.batch_size)
    # Remplir les données
    model_nn.remplissage_donnees(E=args.E, num_sonde=args.num_sonde, nb_training=args.nb_training, nb_test=args.nb_test)

    model_nn.create_reseau()

    model_nn.entrainer()
    
    print(f"R² pour l'entraînement : {model_nn.R2entrainement()}")
    
    # Afficher les résultats
    model_nn.afficher()