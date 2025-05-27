from prediction_f import ModelePrediction, plot_predictions
from tensorflow import keras
from tensorflow.keras import layers

def reseau_neurones(X):
    reseau = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Reshape((X.shape[1], 1)),  # Ajoute une dimension pour la convolution
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.Conv1D(16, kernel_size=3, activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='tanh'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    return reseau

class ModeleReseauNeurones(ModelePrediction):
    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        super().__init__('reseau_neurones', X_train, y_train, X_test, y_test)
        self.reseau = reseau_neurones(X_train)

    def entrainer(self):
        self.reseau.compile(optimizer='adam', loss='mse')
        self.model = self.reseau.fit(self.X_train, self.y_train, epochs=300, batch_size=64, verbose=0)
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
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from reed_data import lire_fichier_U
    from dir import DATA_DIRECTORY
    import numpy as np

    parser = argparse.ArgumentParser(description='Réseau de Neurones Model Training and Evaluation')
    parser.add_argument('--e', type=int, default=145, help='Experiment number to load data for')
    parser.add_argument('--num_sonde', type=int, default=0, help='Number of the probes to use in the model')
    args = parser.parse_args()

    # Lire les données
    times, sondes = lire_fichier_U(os.path.join(DATA_DIRECTORY, f'E_{args.e}', 'U'))
    if not sondes:
        print("Aucune donnée de sonde trouvée.")
        exit(1)
    
    # Préparer les données pour l'entraînement
    pos, col = sondes[args.num_sonde]
    X_train = np.array([v[0] for v in col]).reshape(-1, 1)  # première composante
    y_train = np.array([v[1] for v in col])                 # deuxième composante

    # Créer et entraîner le modèle
    model_nn = ModeleReseauNeurones(X_train, y_train)
    model_nn.entrainer()
    
    print(f"R² pour l'entraînement : {model_nn.R2entrainement()}")
    
    # Afficher les résultats
    model_nn.afficher()