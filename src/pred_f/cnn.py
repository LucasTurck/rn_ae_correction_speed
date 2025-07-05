import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reed_data import lire_fichier_U
import copy
from dir import DATA_DIRECTORY

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from tqdm.keras import TqdmCallback
import numpy as np
import matplotlib.pyplot as plt
import datetime
import hashlib
from sklearn.metrics import mean_squared_error, r2_score
import re
from scipy import stats
from scipy.signal import csd, welch

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


def power_transform(X, power):
    """Applique une transformation de puissance à chaque élément de X."""
    return np.sign(X) * np.power(np.abs(X), power)

class CNN:
    def __init__(self, parameters=None):
        self.name = "CNN"
        self.parameters = parameters if parameters is not None else {}
        self.data = None
        self.model = None
        self.history = None
        self.X_train_before = None
        self.X_train_after = None
        self.y_train = None
        self.Y_pred_train = None
        self.X_test_before = None
        self.X_test_after = None
        self.y_test = None
        self.Y_pred_test = None
             
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
        self.parameters['timesteps_before'] = self.parameters.get('timesteps_before', 10) # nombre de pas de temps avant l'événement
        self.parameters['timesteps_after'] = self.parameters.get('timesteps_after', 10) # nombre de pas de temps après l'événement
        self.parameters['y_col'] = self.parameters.get('y_col', 1) # colonne de la variable cible
        self.parameters['prediction'] = self.parameters.get('prediction', [1, 1, 0])
        self.parameters['puisance'] = self.parameters.get('puisance', [1,2]) # puissance des variables de prédiction
        sum = self.parameters['prediction'][0] + self.parameters['prediction'][1] + self.parameters['prediction'][2]
        self.parameters['input_shape_before'] = self.parameters.get('input_shape_before', (self.parameters['timesteps_before'], sum*len(self.parameters['puisance']))) # forme de l'entrée du modèle
        self.parameters['input_shape_after'] = self.parameters.get('input_shape_after', (self.parameters['timesteps_after'], sum*len(self.parameters['puisance']))) # forme de l'entrée du modèle

        self.parameters['architecture'] = self.parameters.get('architecture', 'cnn_lstm') # architecture du modèle
        self.parameters['batch_size'] = self.parameters.get('batch_size', 512) # taille du batch
        self.parameters['epochs'] = self.parameters.get('epochs', 20) # nombre de fois que le reseau de neurones est entraîné sur l'ensemble de données
        self.parameters['loss'] = self.parameters.get('loss', 'mae') # fonction de perte
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
        
        if self.parameters['timesteps_after'] == 0:
            input_before = keras.Input(shape=self.parameters['input_shape_before'], name="input_before")
            model_before = self.build_model(architecture, input_shape=self.parameters['input_shape_before'])
            out_before = model_before(input_before)
            output = layers.Dense(units=1, activation='linear')(out_before)
            self.model = keras.Model(inputs=input_before, outputs=output)
            return self.model

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
        return self.model

    def create_data(self, train = True):
        """
        Crée les données d'entraînement pour le modèle de prédiction.
        Args:
            train (bool): Si True, crée les données d'entraînement, sinon crée les données de test.
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

        for i in range(timesteps_before, n):
            if timesteps_before > 0:
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
            
            if timesteps_after > 0:
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

    def add_power(self, train = True):
        if train:
            if self.X_train_before is None or self.Y_train is None:
                self.create_data(train=True)
            X_before = self.X_train_before
            X_after = self.X_train_after
            if len(self.parameters['puisance']) == 1 and self.parameters['puisance'][0] == 1:
                return
            X_before = np.concatenate((X_before, np.concatenate([power_transform(X_before, p) for p in self.parameters['puisance'] if p != 1], axis=-1)), axis=-1)
            X_after = np.concatenate((X_after, np.concatenate([power_transform(X_after, p) for p in self.parameters['puisance'] if p != 1], axis=-1)), axis=-1)
            self.X_train_before = X_before
            self.X_train_after = X_after
        else:
            if self.X_test_before is None or self.Y_test is None:
                self.create_data(train=False)
            X_before = self.X_test_before
            X_after = self.X_test_after
            if len(self.parameters['puisance']) == 1 and self.parameters['puisance'][0] == 1:
                return
            X_before = np.concatenate((X_before, np.concatenate([power_transform(X_before, p) for p in self.parameters['puisance'] if p != 1], axis=-1)), axis=-1)
            X_after = np.concatenate((X_after, np.concatenate([power_transform(X_after, p) for p in self.parameters['puisance'] if p != 1], axis=-1)), axis=-1)
            self.X_test_before = X_before
            self.X_test_after = X_after

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

    def train(self):
        """
        Entraîne le modèle de prédiction sur les données d'entraînement.
        """
        if self.X_train_before is None or self.Y_train is None:
            self.create_data(train=True)
        if self.model is None:
            self.create_reseau()


        self.model.compile(optimizer='adam', loss=self.combined_loss, metrics=['mae','mse'])
        if self.parameters['timesteps_after'] == 0:
            X = self.X_train_before
        else:
            X = [self.X_train_before, self.X_train_after]
        self.history = self.model.fit(
            x=X,
            y=self.Y_train,
            batch_size=self.parameters['batch_size'],
            epochs=self.parameters['epochs'],#            validation_split=0.2,
            verbose=0,
            callbacks=[TqdmCallback(verbose=1)]
        )
    
    def predict(self, train = True):
        """
        Prédit les valeurs cibles à partir des données d'entrée.
        """
        if train:
            if self.X_train_before is None or self.Y_train is None:
                self.create_data(train=True)
            if self.model is None:
                self.create_reseau()
            if self.parameters['timesteps_after'] == 0:
                X = self.X_train_before
            else:
                X = [self.X_train_before, self.X_train_after]
            self.Y_pred_train = self.model.predict(X)
            self.Y_pred_train = self.Y_pred_train.reshape(-1, 1)
        else:
            if self.X_test_before is None or self.Y_test is None:
                self.create_data(train=False)
            if self.model is None:
                self.create_reseau()
            if self.parameters['timesteps_after'] == 0:
                X = self.X_test_before
            else:
                X = [self.X_test_before, self.X_test_after]
            self.Y_pred_test = self.model.predict(X)
            self.Y_pred_test = self.Y_pred_test.reshape(-1, 1)
    
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
        
        # Sauvegarde des paramètres du modèle
        chemin_parametres = os.path.join(dossier, "assets", "config.json")
        with open(chemin_parametres, "w") as f:
            json.dump(self.parameters, f, indent=2)
            
        # Sauvegarde de l'historique d'entraînement
        chemin_historique = os.path.join(dossier, "assets", "historique.json")
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
            
        print(f"Modèle et historique sauvegardés dans {dossier}")        

    def load_model(self, path = "../mod"):
        """
        Charge le modèle de prédiction à partir d'un fichier.
        Args:
            path (str): Chemin du fichier à partir duquel charger le modèle.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier de modèle '{path}' n'existe pas.")

        chemin_parametres = os.path.join(path, "assets", "config.json")
        with open(chemin_parametres, "r") as f:
            self.parameters = json.load(f)
        self.init_parameters()
        
        
        self.model = keras.models.load_model(
            path,
            custom_objects={"combined_loss": self.combined_loss}
        )
        
        # Charger l'historique d'entraînement (optionnel)
        chemin_historique = os.path.join(path, "assets", "historique.json")
        if os.path.exists(chemin_historique):
            import types
            with open(chemin_historique, "r") as f:
                history_dict = json.load(f)
            # Créer un objet factice avec un attribut .history
            self.history = types.SimpleNamespace(history=history_dict)
 
    def affichage_history(self, loss, axis=None):
        """
        Affiche l'historique de l'entraînement du modèle.
        Args:
            loss (str): Type de perte à afficher ('loss', 'val_loss', 'mae', 'mse').
        """
        if loss not in self.history.history:
            raise ValueError(f"L'historique '{loss}' n'est pas disponible.")
        if axis is None:
            axis = plt.gca()
        axis.plot(self.history.history[loss], label=loss)
        axis.set_xlabel("Epochs")
        axis.set_ylabel("Loss")

        axis.legend(title=f"Valeur finale : {self.history.history[loss][-1]:.4f} (Epoch {len(self.history.history[loss])})")
        return axis
        
    def affichage_prediction(self, train=True, axis=None):
        """
        Affiche les prédictions du modèle par rapport aux valeurs réelles.
        Args:
            train (bool): Si True, affiche les prédictions sur les données d'entraînement, sinon sur les données de test.
        """
        if axis is None:
            axis = plt.gca()
        if train:
            if self.Y_pred_train is None or self.Y_train is None:
                self.predict(train=True)
            axis.plot(self.Y_train, label='Valeurs réelles')
            axis.plot(self.Y_pred_train, label='Prédictions')
        else:
            if self.Y_pred_test is None or self.Y_test is None:
                self.predict(train=False)
            axis.plot(self.Y_test, label='Valeurs réelles')
            axis.plot(self.Y_pred_test, label='Prédictions')
        axis.set_xlabel("time")
        axis.set_ylabel("vitesse [m/s]")
        axis.legend(title=f"R² : {self.r2_score(train)}")
        
        return axis

    def r2_score(self, train=True):
        """
        Calcule le coefficient de détermination R² pour les prédictions du modèle.
        Args:
            train (bool): Si True, calcule R² sur les données d'entraînement, sinon sur les données de test.
        Returns:
            float: Coefficient de détermination R².
        """
        if train:
            if self.Y_pred_train is None or self.Y_train is None:
                self.predict(train=True)
            try:
                self.r2_train = r2_score(self.Y_train, self.Y_pred_train)
            except ValueError as e:
                print(f"Erreur lors du calcul de R² pour les données d'entraînement : {e}")
                return None
            return self.r2_train
        else:
            if self.Y_pred_test is None or self.Y_test is None:
                self.predict(train=False)
            try:
                self.r2_test = r2_score(self.Y_test, self.Y_pred_test)
            except ValueError as e:
                print(f"Erreur lors du calcul de R² pour les données de test : {e}")
                return None
            return self.r2_test
    
    def affichage_statistiques(self, train=True, axis=None):
        """
        Affiche la RMS, les moments d'ordre 2, 3, 4 et le PDF croisé des prédictions du modèle.
        Args:
            train (bool): Si True, affiche les statistiques sur les données d'entraînement, sinon sur les données de test.
            axis: matplotlib axis (optionnel)
        """
        if axis is None:
            axis = plt.gca()
        if train:
            print("Affichage des statistiques sur les données d'entraînement")
            if self.Y_pred_train is None or self.Y_train is None:
                self.predict(train=True)
            y_true = self.Y_train.flatten()
            y_pred = self.Y_pred_train.flatten()
        else:
            print("Affichage des statistiques sur les données de test")
            if self.Y_pred_test is None or self.Y_test is None:
                self.predict(train=False)
            y_true = self.Y_test.flatten()
            y_pred = self.Y_pred_test.flatten()

        # RMS
        rms_true = np.sqrt(np.mean(y_true**2))
        rms_pred = np.sqrt(np.mean(y_pred**2))
        print(f"RMS (réel): {rms_true:.4f}")
        print(f"RMS (préd): {rms_pred:.4f}")

        # Moments d'ordre 2, 3, 4
        for order in [2, 3, 4]:
            m_true = stats.moment(y_true, moment=order)
            m_pred = stats.moment(y_pred, moment=order)
            print(f"Moment d'ordre {order} (réel): {m_true:.4f}")
            print(f"Moment d'ordre {order} (préd): {m_pred:.4f}")

        # PDF croisé (densité jointe)
        values = np.vstack([y_true, y_pred])
        kde = stats.gaussian_kde(values)
        xmin, xmax = y_true.min(), y_true.max()
        ymin, ymax = y_pred.min(), y_pred.max()
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde(positions).T, X.shape)

        pcm = axis.imshow(
            np.rot90(Z),
            extent=[xmin, xmax, ymin, ymax],
            aspect='auto',
            cmap='viridis'
        )
        axis.set_xlabel("Valeurs réelles")
        axis.set_ylabel("Prédictions")
        if train:
            axis.set_title("PDF croisé (réel, préd) - Entraînement")
        else:
            axis.set_title("PDF croisé (réel, préd) - Test")
        plt.colorbar(pcm, ax=axis, label="Densité")

        return axis
    
    def affichage_fft(self, train=True, axis=None, fs=1.0):
        """
        Affiche le spectre de puissance (FFT) des signaux réel et prédit.
        Args:
            train (bool): Si True, utilise les données d'entraînement, sinon de test.
            axis: matplotlib axis (optionnel)
            fs (float): Fréquence d'échantillonnage (par défaut 1.0)
        """
        if axis is None:
            axis = plt.gca()
        if train:
            if self.Y_pred_train is None or self.Y_train is None:
                self.predict(train=True)
            y_true = self.Y_train.flatten()
            y_pred = self.Y_pred_train.flatten()
        else:
            if self.Y_pred_test is None or self.Y_test is None:
                self.predict(train=False)
            y_true = self.Y_test.flatten()
            y_pred = self.Y_pred_test.flatten()

        # Calcul FFT
        N = len(y_true)
        freq = np.fft.rfftfreq(N, d=1/fs)
        fft_true = np.fft.rfft(y_true - np.mean(y_true))
        fft_pred = np.fft.rfft(y_pred - np.mean(y_pred))

        # Spectre de puissance
        power_true = np.abs(fft_true)**2
        power_pred = np.abs(fft_pred)**2

        axis.plot(freq, power_true, label="Réalité", color='blue')
        axis.plot(freq, power_pred, label="Prédiction", color='orange', linestyle='--')
        axis.set_xlabel("Fréquence")
        axis.set_ylabel("Spectre de puissance")
        if train:
            axis.set_title("Spectre de puissance (FFT) - Entraînement")
        else:
            axis.set_title("Spectre de puissance (FFT) - Test")
        axis.set_yscale('log')
        axis.set_xscale('log')
        axis.legend()
        return axis
    
    def affichage_welch(self, train=True, axis=None, fs=1.0, nperseg=1024):
        """
        Affiche le spectre de puissance (Welch) des signaux réel et prédit.
        Args:
            train (bool): Si True, utilise les données d'entraînement, sinon de test.
            axis: matplotlib axis (optionnel)
            fs (float): Fréquence d'échantillonnage (par défaut 1.0)
            nperseg (int): Taille des segments pour Welch (par défaut 1024)
        """
        if axis is None:
            axis = plt.gca()
        if train:
            if self.Y_pred_train is None or self.Y_train is None:
                self.predict(train=True)
            y_true = self.Y_train.flatten()
            y_pred = self.Y_pred_train.flatten()
        else:
            if self.Y_pred_test is None or self.Y_test is None:
                self.predict(train=False)
            y_true = self.Y_test.flatten()
            y_pred = self.Y_pred_test.flatten()

        # Spectre de puissance par Welch
        f_true, pxx_true = welch(y_true - np.mean(y_true), fs=fs, nperseg=min(nperseg, len(y_true)))
        f_pred, pxx_pred = welch(y_pred - np.mean(y_pred), fs=fs, nperseg=min(nperseg, len(y_pred)))

        axis.semilogy(f_true, pxx_true, label="Réalité", color='blue')
        axis.semilogy(f_pred, pxx_pred, label="Prédiction", color='orange', linestyle='--')
        axis.set_xlabel("Fréquence [Hz]")
        axis.set_ylabel("Spectre de puissance (Welch)")
        if train:
            axis.set_title("Spectre de puissance (Welch) - Entraînement")
        else:
            axis.set_title("Spectre de puissance (Welch) - Test")
        axis.legend()
        axis.grid(True, which="both", ls="--", alpha=0.5)
        return axis
    
    def affichage_co_spectre(self, train=True, axis=None, fs=1.0):
        """
        Affiche le co-spectre (densité spectrale croisée) entre le signal réel et prédit.
        Args:
            train (bool): Si True, utilise les données d'entraînement, sinon de test.
            axis: matplotlib axis (optionnel)
            fs (float): Fréquence d'échantillonnage (Hz)
        """
    
        if axis is None:
            axis = plt.gca()
        if train:
            if self.Y_pred_train is None or self.Y_train is None:
                self.predict(train=True)
            y_true = self.Y_train.flatten()
            y_pred = self.Y_pred_train.flatten()
        else:
            if self.Y_pred_test is None or self.Y_test is None:
                self.predict(train=False)
            y_true = self.Y_test.flatten()
            y_pred = self.Y_pred_test.flatten()
    
        # Calcul du co-spectre
        f, Pxy = csd(y_true, y_pred, fs=fs, nperseg=min(1024, len(y_true)))
        axis.semilogy(f, np.abs(Pxy), label="|Co-spectre|")
        axis.set_xlabel("Fréquence [Hz]")
        axis.set_ylabel("Amplitude")
        if train:
            axis.set_title("Co-spectre (Entraînement)")
        else:
            axis.set_title("Co-spectre (Test)")
        axis.legend()
        axis.grid(True, which="both", ls="--", alpha=0.5)
        return axis
    
    
if __name__ == "__main__":
    reseau = CNN()
    # reseau.load_model(os.path.join(MOD_PERSO_DIRECTORY, "cnn_lstm", "run_20250624_150753_5cd228d9"))
    reseau.init_parameters()#
    reseau.load_data()
    reseau.create_reseau()#
    reseau.create_data(train=True)
    reseau.create_data(train=False)
    reseau.train()#
    reseau.predict(train=True)
    reseau.predict(train=False)
    print(f"R² sur les données d'entraînement : {reseau.r2_score(train=True)}")
    print(f"R² sur les données de test : {reseau.r2_score(train=False)}")
    
    reseau.save_model()#
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    reseau.affichage_history('mae', axis=ax[0])
    reseau.affichage_prediction(train=True, axis=ax[1])
    reseau.affichage_prediction(train=False, axis=ax[2])
    
    plt.tight_layout()
    plt.show()
