from prediction_f import ModelePrediction, plot_predictions
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class ModeleRegressionLineaire(ModelePrediction):
    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        super().__init__('regression_lineaire', X_train, y_train, X_test, y_test)

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
        plot_predictions(self.X_train, self.y_train, self.y_pred_train, "Regression Lineaire - Train")
        if self.X_test is not None and self.y_test is not None:
            plot_predictions(self.X_test, self.y_test, self.y_pred_test, "Regression Lineaire - Test")

class ModeleRegressionPolynomiale(ModelePrediction):
    def __init__(self, X_train, y_train, degree, X_test=None, y_test=None):
        super().__init__('regression_polynomiale', X_train, y_train, X_test, y_test)
        self.degree = degree
        self.X_poly = None
        
    def entrainer(self):
        poly = PolynomialFeatures(degree=self.degree)
        self.X_poly = poly.fit_transform(self.X_train)
        self.model = LinearRegression().fit(self.X_poly, self.y_train)
        self.y_pred_train = self.model.predict(self.X_poly)

    def predire(self):
        if self.X_test is not None:
            self.y_pred_test = self.model.predict(self.X_poly)
        else:
            self.y_pred_test = None
            
    def afficher(self):
        plot_predictions(self.X_train, self.y_train, self.y_pred_train, f"Regression Polynomiale (degré {self.degree}) - Train")
        if self.X_test is not None and self.y_test is not None:
            plot_predictions(self.X_test, self.y_test, self.y_pred_test, f"Regression Polynomiale (degré {self.degree}) - Test")

class ModeleNaif(ModelePrediction):
    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        super().__init__('naif', X_train, y_train, X_test, y_test)

    def entrainer(self):
        self.y_pred_train = self.X_train.flatten()

    def predire(self):
        if self.X_test is not None:
            self.y_pred_test = self.X_test.flatten()
        else:
            self.y_pred_test = None

    def afficher(self):
        plot_predictions(self.X_train, self.y_train, self.y_pred_train, "Méthode Naïve - Train")
        if self.X_test is not None and self.y_test is not None:
            plot_predictions(self.X_test, self.y_test, self.y_pred_test, "Méthode Naïve - Test")

class ModeleAleatoire(ModelePrediction):
    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        super().__init__('aleatoire', X_train, y_train, X_test, y_test)

    def entrainer(self):
        self.y_pred_train = np.random.uniform(-1.5, 1.5, size=self.X_train.shape)

    def predire(self):
        if self.X_test is not None:
            self.y_pred_test = np.random.rand(*self.X_test.shape)
        else:
            self.y_pred_test = None

    def afficher(self):
        plot_predictions(self.X_train, self.y_train, self.y_pred_train, "Méthode Aléatoire - Train")
        if self.X_test is not None and self.y_test is not None:
            plot_predictions(self.X_test, self.y_test, self.y_pred_test, "Méthode Aléatoire - Test")
   
if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from reed_data import lire_fichier_U
    from dir import DATA_DIRECTORY
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Regression Lineaire Model Training and Evaluation')
    parser.add_argument('--e', type=int, default=145, help='Experiment number to load data for')
    parser.add_argument('--num_sonde', type=int, default=0, help='Number of the probe to use in the model')
    parser.add_argument('--methode', type=str, choices=['lineaire', 'polynomiale', 'naive', 'aleatoire'], default='lineaire', help='Method of regression to use')
    parser.add_argument('--degree', type=int, default=2, help='Degree of the polynomial for regression')
    args = parser.parse_args()
    
    #lire les données
    times, sondes = lire_fichier_U(os.path.join(DATA_DIRECTORY, f'E_{args.e}', 'U'))
    if not sondes:
        print("Aucune donnée de sonde trouvée.")
        exit(1)
        
    # Préparer les données pour l'entraînement
    pos, col = sondes[args.num_sonde]
    X = np.array([v[0] for v in col]).reshape(-1, 1)  # première composante
    y = np.array([v[1] for v in col])                 # deuxième composante
    
    # Créer et entraîner le modèle
    if args.methode == 'lineaire':
        model = ModeleRegressionLineaire(X, y)
    elif args.methode == 'polynomiale':
        model = ModeleRegressionPolynomiale(X, y, degree=args.degree)
    elif args.methode == 'naive':
        model = ModeleNaif(X, y)
    elif args.methode == 'aleatoire':
        model = ModeleAleatoire(X, y)
    else:
        raise ValueError("Méthode inconnue. Choisissez parmi 'lineaire', 'polynomiale', 'naive', 'aleatoire'.")

    model.entrainer()
    print(f"R² pour l'entraînement : {model.R2entrainement()}")
    
    # Afficher les résultats
    model.afficher()