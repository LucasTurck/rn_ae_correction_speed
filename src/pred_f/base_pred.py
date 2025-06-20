from prediction_f import ModelePrediction, plot_predictions, args_to_dict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class ModeleRegressionLineaire(ModelePrediction):
    def __init__(self, parameters = None):
        super().__init__('regression_lineaire', parameters)
        
    def init_parameters(self):
        super().init_parameters()
        print(f"Initialisation des paramètres pour la régression linéaire : {self.parameters}")

    def remplissage_donnees(self, train=True):
        super().remplissage_donnees(train)
        n = sum(self.parameters['prediction_end']) * self.parameters['timesteps'] - sum(self.parameters['prediction_start'])*self.parameters['timesteps']
        print(n)
        
        if train:
            X = np.zeros((self.X_train.shape[0], n))  # Crée un tableau vide pour stocker les données
            for i in range(self.X_train.shape[0]):
                X[i,:] = self.X_train[i,:].flatten()  # Assure que X_train est un tableau 2D
            self.X_train = np.array(X)
            print(f"X_train shape: {self.X_train.shape}")
        else:
            X = np.zeros((self.X_test.shape[0], n))  # Crée un tableau vide pour stocker les données
            for i in range(self.X_test.shape[0]):
                X[i,:] = self.X_test[i,:].flatten()  # Assure que X_test est un tableau 2D
            self.X_test = np.array(X)
            print(f"X_test shape: {self.X_test.shape}")

    def entrainer(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred_train = self.model.predict(self.X_train)

    def predire(self):
        if self.X_test is not None:
            self.y_pred_test = self.model.predict(self.X_test)
        else:
            self.y_pred_test = None

    def afficher(self):
        plot_predictions(self.X_train, self.y_train, self.y_pred_train, "Train - Regression Lineaire", block=False)
        if self.X_test is not None and self.y_test is not None:
            plot_predictions(self.X_test, self.y_test, self.y_pred_test, "Test - Regression Lineaire", block=True)

class ModeleRegressionPolynomiale(ModelePrediction):
    def __init__(self, parameters = None):
        super().__init__('regression_polynomiale', parameters)
        
    def init_parameters(self):
        super().init_parameters()
        self.parameters['degree'] = self.parameters.get('degree', 2)  # Default degree for polynomial regression is 2

    def entrainer(self):
        poly = PolynomialFeatures(degree=self.parameters['degree'])
        X_poly = poly.fit_transform(self.X_train)
        self.model = LinearRegression().fit(X_poly, self.y_train)
        self.y_pred_train = self.model.predict(X_poly)

    def predire(self):
        if self.X_test is not None:
            X_poly_test = PolynomialFeatures(degree=self.parameters['degree']).fit_transform(self.X_test)
            self.y_pred_test = self.model.predict(X_poly_test)
        else:
            self.y_pred_test = None
            
    def afficher(self):
        plot_predictions(self.X_train, self.y_train, self.y_pred_train, f"Train - Regression Polynomiale (degré {self.parameters['degree']})", block=False)
        if self.X_test is not None and self.y_test is not None:
            plot_predictions(self.X_test, self.y_test, self.y_pred_test, f"Test - Regression Polynomiale (degré {self.parameters['degree']})", block=True)

class ModeleNaif(ModelePrediction):
    def __init__(self, parameters = None):
        super().__init__('naif', parameters)

    def entrainer(self):
        self.y_pred_train = self.X_train[:, -1].flatten()  # Utilise la dernière colonne de X_train comme prédiction

    def predire(self):
        if self.X_test is not None:
            self.y_pred_test = self.X_test[:, -1].flatten()  # Utilise la dernière colonne de X_test comme prédiction
        else:
            self.y_pred_test = None

    def afficher(self):
        plot_predictions(self.X_train, self.y_train, self.y_pred_train, "Train - Méthode Naïve", block=False)
        if self.X_test is not None and self.y_test is not None:
            plot_predictions(self.X_test, self.y_test, self.y_pred_test, "Test - Méthode Naïve", block=True)

class ModeleAleatoire(ModelePrediction):
    def __init__(self, parameters = None):
        super().__init__('aleatoire', parameters)

    def entrainer(self):
        self.y_pred_train = np.random.uniform(-1.5, 1.5, size=self.y_train.shape)

    def predire(self):
        if self.X_test is not None:
            self.y_pred_test = np.random.uniform(-1.5, 1.5, size=self.y_test.shape)
        else:
            self.y_pred_test = None

    def afficher(self):
        plot_predictions(self.X_train, self.y_train, self.y_pred_train, "Train - Méthode Aléatoire", block=False)
        if self.X_test is not None and self.y_test is not None:
            plot_predictions(self.X_test, self.y_test, self.y_pred_test, "Test - Méthode Aléatoire", block=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Regression Lineaire Model Training and Evaluation')
    parser.add_argument('--E', type=int, help='Experiment number to load data for')
    parser.add_argument('--num_sonde_train', type=int, help='Number of the probes to use in the model')
    parser.add_argument('--num_sonde_test', type=int, help='Number of the probes to use in the model for testing')
    parser.add_argument('--nb_training', type=int, help='Number of training samples to use, -1 for all')
    parser.add_argument('--nb_test', type=int, help='Number of test samples to use, -1 for all')
    parser.add_argument('--methode', type=str, default='lineaire', choices=['lineaire', 'polynomiale', 'naive', 'aleatoire'], help='Method of regression to use')
    parser.add_argument('--degree', type=int, help='Degree of the polynomial for regression')
    parser.add_argument('--prediction', type=int, nargs=3, help='Prediction parameters [p, f] where p is the number of previous samples of u and f is the number of previous samples of v to use')
    parser.add_argument('--y', type=int, help='Name of the variable to predict (e.g., "u", "v")')
    args = parser.parse_args()
    args_dict = args_to_dict(args)

    # Créer et entraîner le modèle
    if args.methode == 'lineaire':
        model = ModeleRegressionLineaire(parameters = args_dict)
    elif args.methode == 'polynomiale':
        model = ModeleRegressionPolynomiale(parameters = args_dict)
    elif args.methode == 'naive':
        model = ModeleNaif(parameters = args_dict)
    elif args.methode == 'aleatoire':
        model = ModeleAleatoire(parameters = args_dict)
    else:
        raise ValueError("Méthode inconnue. Choisissez parmi 'lineaire', 'polynomiale', 'naive', 'aleatoire'.")

    model.init_parameters()

    model.remplissage_donnees()

    model.entrainer()
    print(f"R² pour l'entraînement : {model.evaluer(train=True)}")

    # Remplir les données de test
    model.remplissage_donnees(train=False)

    # Prédire les valeurs sur les données de test
    model.predire()

    print(f"R² pour le test : {model.evaluer(train=False)}")

    # Afficher les résultats
    model.afficher()