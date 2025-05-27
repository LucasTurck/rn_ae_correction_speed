import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class ModelePrediction:
    def __init__(self, nom_modele, X_train, y_train, X_test=None, y_test=None):
        self.nom_modele = nom_modele
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.y_pred_train = None
        self.y_pred_test = None
    
    def entrainer(self):
        raise NotImplementedError

    def predire(self, X):
        raise NotImplementedError
    
    def R2entrainement(self):
        if self.y_pred_train is not None:
            return r2_score(self.y_train, self.y_pred_train)
        return None

    def evaluer(self):
        if self.X_test is not None and self.y_test is not None:
            y_pred = self.predire(self.X_test)
            return r2_score(self.y_test, y_pred)
        return None


    def afficher(self):
        raise NotImplementedError


def plot_predictions(X, y, y_pred, title = "Prédictions vs Réel", block=True):

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Données réelles', alpha=0.5)
    plt.scatter(X, y_pred, color='red', label='Prédictions', alpha=0.5)
    plt.xlabel("Première composante (X)")
    plt.ylabel("Deuxième composante (Y)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show(block=block)
        
def scatter_plot(X, y, sonde_num):
    # Affichage d'un nuage de points pour la sonde choisie
    plt.figure(figsize=(6, 6))
    plt.scatter(X, y, alpha=0.5, label="Données réelles")
    plt.xlabel("Première composante (X)")
    plt.ylabel("Deuxième composante (Y)")
    plt.title(f"Sonde {sonde_num} : relation X/Y")
    plt.legend()
    plt.grid(True)
    plt.show()
