import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reed_data import lire_fichier_U
import copy
from dir import DATA_DIRECTORY

from tensorflow import keras
from tensorflow.keras import layers
import json
from tqdm.keras import TqdmCallback
import numpy as np
import matplotlib.pyplot as plt
import datetime
import hashlib
from sklearn.metrics import mean_squared_error, r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Ignore TensorFlow warnings


def get_run_id(parameters):
    # Utilise un hash court des paramètres pour l’identifiant
    param_str = json.dumps(parameters, sort_keys=True)
    hash_id = hashlib.md5(param_str.encode()).hexdigest()[:8]
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{date_str}_{hash_id}"


def get_architecture_by_name(name, architectures):
    for arch in architectures:
        if arch["name"] == name:
            return arch["layers"]
    raise ValueError(f"Architecture '{name}' non trouvée.")


class CNN:
    def __init__(self, parameters):
        self.name = "CNN"
        self.parameters = parameters if parameters is not None else {}
        self.data = None
        self.model = None
        self.history = None
        
    def load_data(self):
        # lire les données
        times, sondes = lire_fichier_U(os.path.join(DATA_DIRECTORY, f"E_{self.parameters['E']}", 'U'))
        self.data = [sondes[i][1] for i in range(len(sondes))]
        
    def init_parameters(self):
        """
        Initialise les paramètres du modèle de prédiction.
        Définit les paramètres par défaut pour l'entraînement et la prédiction.
        """
        self.parameters['E'] = self.parameters.get('E', 125) # auteur de la sonde
        self.parameters['num_sonde_train'] = self.parameters.get('num_sonde_train', 0) # numéro de la sonde d'entraînement
        self.parameters['num_sonde_test'] = self.parameters.get('num_sonde_test', 1) # numéro de la sonde de test
        self.parameters['nb_training'] = self.parameters.get('nb_training', -1) # nombre d'exemples d'entraînement (-1 pour tout utiliser)
        self.parameters['nb_test'] = self.parameters.get('nb_test', 100) # nombre d'exemples de test
        self.parameters['timesteps_before'] = self.parameters.get('timesteps_before', 5) # nombre de pas de temps avant l'événement
        self.parameters['timesteps_after'] = self.parameters.get('timesteps_after', 5) # nombre de pas de temps après l'événement
        self.parameters['y_col'] = self.parameters.get('y_col', 1) # colonne de la variable cible
        self.parameters['prediction'] = self.parameters.get('prediction', [1, 1, 0])
        sum = self.parameters['prediction'][0] + self.parameters['prediction'][1] + self.parameters['prediction'][2]
        self.parameters['input_shape_before'] = self.parameters.get('input_shape_before', (self.parameters['timesteps_before'], sum)) # forme de l'entrée du modèle
        self.parameters['input_shape_after'] = self.parameters.get('input_shape_after', (self.parameters['timesteps_after'], sum)) # forme de l'entrée du modèle

        self.parameters['architecture'] = self.parameters.get('architecture', 'cnn_lstm') # architecture du modèle
        self.parameters['batch_size'] = self.parameters.get('batch_size', 512) # taille du batch
        self.parameters['epochs'] = self.parameters.get('epochs', 100) # nombre de fois que le reseau de neurones est entraîné sur l'ensemble de données
        print(f"Paramètres du modèle de prédiction : {self.parameters}")
        
    def set_parameters(self, parameters):
        """
        Met à jour les paramètres du modèle de prédiction.
        Args:
            parameters (dict): Dictionnaire contenant les paramètres à mettre à jour.
        """
        for key, value in parameters.items():
            self.parameters[key] = value
        self.init_parameters()
        
    def build_model(self, architecture=None, input_shape=None):
        """
        Construit le modèle de prédiction en fonction des paramètres définis.
        """

        if architecture is None:
            raise ValueError("L'architecture doit être spécifiée pour construire le modèle.")

        model = keras.Sequential()
        model.add(layers.Input(shape=input_shape))

        for layer in architecture:
            layer_type = layer["type"]
            if layer_type == "Dense":
                model.add(layers.Dense(units=layer.get("units", 32), 
                                       activation=layer.get("activation", "relu")))
            elif layer_type == "Conv1D":
                model.add(layers.Conv1D(filters=layer.get("filters", 32), 
                                        kernel_size=layer.get("kernel_size", 3),
                                        activation=layer.get("activation", "relu"),
                                        padding=layer.get("padding", "valid"),
                                        strides=layer.get("strides", 1)))
            
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            
            elif layer_type == "Reshape":
                target_shape = layer.get("target_shape")
                if target_shape is None:
                    model.add(layers.Reshape(input_shape))
                else:
                    model.add(layers.Reshape(target_shape))
            
            elif layer_type == "Dropout":
                model.add(layers.Dropout(rate=layer.get("rate", 0.5)))
            elif layer_type == "BatchNormalization":
                model.add(layers.BatchNormalization())
            elif layer_type == "LSTM":
                model.add(layers.LSTM(units=layer.get("units", 32), 
                                      activation=layer.get("activation", "tanh"),
                                      return_sequences=layer.get("return_sequences", False)))
            elif layer_type == "GRU":
                model.add(layers.GRU(units=layer.get("units", 32), 
                                      activation=layer.get("activation", "tanh"),
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
                model.add(layers.Bidirectional(inner_layer, 
                                               merge_mode=layer.get("merge_mode", "concat")))
            else:
                raise ValueError(f"Type de couche non supporté: {layer_type}")
        
        return model
        
    def create_reseau(self):
        if self.data is None:
            self.load_data()
        
        json_path = os.path.join(os.path.dirname(__file__), "architectures.json")
        with open(json_path, 'r') as f:
            architectures = json.load(f)
        
        architecture = get_architecture_by_name(self.parameters['architecture'], architectures)
        
        if architecture is None:
            raise ValueError(f"Architecture '{self.parameters['architecture']}' non trouvée dans le fichier JSON.")
        
        input_before = keras.Input(shape=self.parameters['input_shape_before'], name="input_before")
        input_after = keras.Input(shape=self.parameters['input_shape_after'], name="input_after")
        model_before = self.build_model(architecture, input_shape=self.parameters['input_shape_before'])
        model_after = self.build_model(architecture, input_shape=self.parameters['input_shape_after'])
        
        out_before = model_before(input_before)
        out_after = model_after(input_after)
        
        # Fusionne les sorties (concaténation)
        merged = layers.Concatenate()([out_before, out_after])
        output = layers.Dense(units=1, activation='linear')(merged)

        self.model = keras.Model(inputs=[input_before, input_after], outputs=output)

    def create_data(self, train = True):
        """
        Crée les données d'entraînement pour le modèle de prédiction.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé. Veuillez appeler create_reseau() avant d'entraîner le modèle.")
        
        # Préparer les données pour l'entraînement
        if train:
            data = copy.deepcopy(self.data[self.parameters['num_sonde_train']])
        else:
            data = copy.deepcopy(self.data[self.parameters['num_sonde_test']])

        X_before = []
        X_after = []
        Y = []
        y = self.parameters['y_col']
        prediction = self.parameters['prediction']
        features_before = []
        timesteps_before = self.parameters['timesteps_before']
        for i in range(timesteps_before):
            timesteps_vector = []
            if prediction[0] == 1:
                timesteps_vector.append(data[i+1][0])
            if y == 1:
                if prediction[1] == 1:
                    timesteps_vector.append(data[i][1])
                if prediction[2] == 1:
                    timesteps_vector.append(data[i+1][2])
            elif y == 2:
                if prediction[1] == 1:
                    timesteps_vector.append(data[i+1][1])
                if prediction[2] == 1:
                    timesteps_vector.append(data[i][2])
            features_before.append(timesteps_vector)
            
        features_after = []
        timesteps_after = self.parameters['timesteps_after']
        for i in range(timesteps_after):
            timesteps_vector = []
            if prediction[0] == 1:
                timesteps_vector.append(data[i+1+timesteps_before][0])
            if prediction[1] == 1:
                timesteps_vector.append(data[i+1+timesteps_before][1])
            if prediction[2] == 1:
                timesteps_vector.append(data[i+1+timesteps_before][2])
            features_after.append(timesteps_vector)

        X_before.append(copy.deepcopy(features_before))
        X_after.append(copy.deepcopy(features_after[::-1]))
        Y.append(data[timesteps_before][y])

        if train:
            nb = self.parameters['nb_training']
        else:
            nb = self.parameters['nb_test']
        if nb == -1:
            nb = len(data) - timesteps_before - timesteps_after - 1
        n = min(len(data) - timesteps_after - 1, timesteps_before + nb - 1)
        print(n)
        for i in range(timesteps_before, n):
            features_before.pop(0)
            timesteps_vector = []
            if prediction[0] == 1:
                timesteps_vector.append(data[i+1][0])
            if y == 1:
                if prediction[1] == 1:
                    timesteps_vector.append(data[i][1])
                if prediction[2] == 1:
                    timesteps_vector.append(data[i+1][2])
            elif y == 2:
                if prediction[1] == 1:
                    timesteps_vector.append(data[i+1][1])
                if prediction[2] == 1:
                    timesteps_vector.append(data[i][2])
            features_before.append(timesteps_vector)
            
            features_after.pop(0)
            timesteps_vector = []
            if prediction[0] == 1:
                timesteps_vector.append(data[i+1+timesteps_after][0])
            if prediction[1] == 1:
                timesteps_vector.append(data[i+1+timesteps_after][1])
            if prediction[2] == 1:
                timesteps_vector.append(data[i+1+timesteps_after][2])
            features_after.append(timesteps_vector)
            X_before.append(copy.deepcopy(features_before))
            X_after.append(copy.deepcopy(features_after[::-1]))
            Y.append(data[i][y])

        # print(f"X = {X[:5]}")
        # print(f"Nombre data : {np.array(data).shape}, n = {n}, timesteps_before = {timesteps_before}, timesteps_after = {timesteps_after}")
        # Y = data[timesteps_before: -timesteps_after][y]
        # print(f"Nombre d'exemples d'entraînement : {len(X)}")
        # print(f"Nombre d'exemples de test : {len(Y)}")
        if train:
            self.X_train_before = np.array(X_before)
            self.X_train_after = np.array(X_after)
            self.Y_train = np.array(Y)
        else:
            self.X_test_before = np.array(X_before)
            self.X_test_after = np.array(X_after)
            self.Y_test = np.array(Y)

    def train(self):
        """
        Entraîne le modèle de prédiction sur les données d'entraînement.
        """
        if self.X_train_before is None or self.Y_train is None:
            self.create_data(train=True)
        if self.model is None:
            self.create_reseau()


        self.model.compile(optimizer='adam', loss='mae', metrics=['mae','mse'])
        self.history = self.model.fit(
            x=[self.X_train_before, self.X_train_after],
            y=self.Y_train,
            batch_size=self.parameters['batch_size'],
            epochs=self.parameters['epochs'],#            validation_split=0.2,
            verbose=0,
            callbacks=[TqdmCallback(verbose=1)]
        )

        self.Y_pred_train = self.model.predict([self.X_train_before, self.X_train_after])
        self.Y_pred_train = self.Y_pred_train.reshape(-1, 1)
    
    def predict(self, train = True):
        """
        Prédit les valeurs cibles à partir des données d'entrée.
        """
        if train:
            if self.X_train_before is None or self.Y_train is None:
                self.create_data(train=True)
            if self.model is None:
                self.create_reseau()
            self.Y_pred_train = self.model.predict([self.X_train_before, self.X_train_after])
            self.Y_pred_train = self.Y_pred_train.reshape(-1, 1)
        else:
            if self.X_test_before is None or self.Y_test is None:
                self.create_data(train=False)
            self.Y_pred_test = self.model.predict([self.X_test_before, self.X_test_after])
            self.Y_pred_test = self.Y_pred_test.reshape(-1, 1)
    
    def r2_score(self, train =  True):
        """
        Calcule le coefficient de détermination R² pour les prédictions du modèle.
        Args:
            train (bool): Si True, calcule R² sur les données d'entraînement, sinon sur les données de test.
        Returns:
            float: Coefficient de détermination R².
        """
        
    
    def save_model(self, path = "../mod"):
        """
        Enregistre le modèle de prédiction dans un fichier.
        Args:
            path (str): Chemin du fichier où enregistrer le modèle.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé. Veuillez appeler create_reseau() avant d'enregistrer le modèle.")
        run_id = get_run_id(self.parameters)
        dossier = os.path.join(path, f"{self.parameters['architecture']}", run_id)
        self.model.save(dossier)

    def charge_model(self, path = "../mod"):
        """
        Charge le modèle de prédiction à partir d'un fichier.
        Args:
            path (str): Chemin du fichier à partir duquel charger le modèle.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier de modèle '{path}' n'existe pas.")
        self.model = keras.models.load_model(path)
        
    def affichage_history(self, loss):
        """
        Affiche l'historique de l'entraînement du modèle.
        Args:
            loss (str): Type de perte à afficher ('loss', 'val_loss', 'mae', 'mse').
        """
        if loss not in self.history.history:
            raise ValueError(f"L'historique '{loss}' n'est pas disponible.")
        plt.plot(self.history.history[loss], label=loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
    def affichage_prediction(self, train=True):
        """
        Affiche les prédictions du modèle par rapport aux valeurs réelles.
        Args:
            train (bool): Si True, affiche les prédictions sur les données d'entraînement, sinon sur les données de test.
        """
        if train:
            if self.Y_pred_train is None or self.Y_train is None:
                self.predict(train=True)
            plt.plot(self.Y_train, label='Valeurs réelles')
            plt.plot(self.Y_pred_train, label='Prédictions')
        else:
            if self.Y_pred_test is None or self.Y_test is None:
                self.predict(train=False)
            plt.plot(self.Y_test, label='Valeurs réelles')
            plt.plot(self.Y_pred_test, label='Prédictions')
        plt.xlabel("Index")
        plt.ylabel("Valeur")
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    reseau = CNN({})
    reseau.init_parameters()
    reseau.load_data()
    reseau.create_reseau()
    reseau.create_data(train=True)
    reseau.create_data(train=False)
    reseau.train()
    reseau.save_model()
    reseau.predict(train=True)
    reseau.predict(train=False)
    reseau.affichage_history('mae')
    reseau.affichage_prediction(train=True)
    reseau.affichage_prediction(train=False)
    