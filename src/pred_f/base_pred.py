from prediction_f import ModelePrediction, plot_predictions
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class ModeleRegressionLineaire(ModelePrediction):
    def __init__(self, prediction, X_train=None, y_train=None, X_test=None, y_test=None):
        super().__init__('regression_lineaire', prediction, X_train, y_train, X_test, y_test)

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
    def __init__(self, degree, prediction, X_train=None, y_train=None, X_test=None, y_test=None):
        super().__init__('regression_polynomiale', prediction, X_train, y_train, X_test, y_test)
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
    def __init__(self, prediction, X_train=None, y_train=None, X_test=None, y_test=None):
        super().__init__('naif', prediction, X_train, y_train, X_test, y_test)

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
    def __init__(self, prediction, X_train=None, y_train=None, X_test=None, y_test=None):
        super().__init__('aleatoire', prediction, X_train, y_train, X_test, y_test)

    def entrainer(self):
        self.y_pred_train = np.random.uniform(-1.5, 1.5, size=self.y_train.shape)

    def predire(self):
        if self.X_test is not None:
            self.y_pred_test = np.random.uniform(-1.5, 1.5, size=self.y_test.shape)
        else:
            self.y_pred_test = None

    def afficher(self):
        plot_predictions(self.X_train, self.y_train, self.y_pred_train, "Méthode Aléatoire - Train")
        if self.X_test is not None and self.y_test is not None:
            plot_predictions(self.X_test, self.y_test, self.y_pred_test, "Méthode Aléatoire - Test")
   
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Regression Lineaire Model Training and Evaluation')
    parser.add_argument('--e', type=int, default=145, help='Experiment number to load data for')
    parser.add_argument('--num_sonde', type=int, default=0, help='Number of the probe to use in the model')
    parser.add_argument('--methode', type=str, choices=['lineaire', 'polynomiale', 'naive', 'aleatoire'], default='lineaire', help='Method of regression to use')
    parser.add_argument('--nb_training', type=int, default=-1, help='Number of training samples to use, -1 for all')
    parser.add_argument('--degree', type=int, default=2, help='Degree of the polynomial for regression')
    parser.add_argument('--prediction', type=int, nargs=2, default=[0, 0], help='Prediction parameters [p, f] where p is the number of previous samples of u and f is the number of previous samples of v to use')
    args = parser.parse_args()
    
    # Créer et entraîner le modèle
    if args.methode == 'lineaire':
        model = ModeleRegressionLineaire(prediction=args.prediction)
    elif args.methode == 'polynomiale':
        model = ModeleRegressionPolynomiale(degree=args.degree, prediction=args.prediction)
    elif args.methode == 'naive':
        model = ModeleNaif(prediction=args.prediction)
    elif args.methode == 'aleatoire':
        model = ModeleAleatoire(prediction=args.prediction)
    else:
        raise ValueError("Méthode inconnue. Choisissez parmi 'lineaire', 'polynomiale', 'naive', 'aleatoire'.")


    model.remplissage_donnees(e=args.e, num_sonde=args.num_sonde, nb_training=args.nb_training)
    
    model.entrainer()
    print(f"R² pour l'entraînement : {model.R2entrainement()}")
    
    # Afficher les résultats
    model.afficher()