from prediction_f import ModelePrediction, plot_predictions
from tensorflow import keras
from tensorflow.keras import layers

def reseau_neurones(X):
    reseau = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Reshape((X.shape[1], 1)),
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
    ])
    
    return reseau

class ModeleReseauNeurones(ModelePrediction):
    def __init__(self, prediction, X_train=None, y_train=None, X_test=None, y_test=None):
        super().__init__('reseau_neurones', prediction, X_train, y_train, X_test, y_test)
        

    def entrainer(self):
        self.reseau.compile(optimizer='adam', loss='mse')
        self.model = self.reseau.fit(self.X_train, self.y_train, epochs=1000, batch_size=64, verbose=0)
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
    parser.add_argument('--e', type=int, default=145, help='Experiment number to load data for')
    parser.add_argument('--num_sonde', type=int, default=0, help='Number of the probes to use in the model')
    parser.add_argument('--nb_training', type=int, default=-1, help='Number of training samples to use, -1 for all')
    parser.add_argument('--nb_test', type=int, default=-1, help='Number of test samples to use, -1 for all')
    parser.add_argument('--prediction', type=int, nargs=2, default=[0, 0], help='Prediction parameters [p, f] where p is the number of previous samples of u and f is the number of previous samples of v to use')
    args = parser.parse_args()

    # Créer et entraîner le modèle
    model_nn = ModeleReseauNeurones(prediction =args.prediction)
    model_nn.remplissage_donnees(e=args.e, num_sonde=args.num_sonde, nb_training=args.nb_training, nb_test=args.nb_test)

    model_nn.reseau = reseau_neurones(model_nn.X_train)

    model_nn.entrainer()
    
    print(f"R² pour l'entraînement : {model_nn.R2entrainement()}")
    
    # Afficher les résultats
    model_nn.afficher()