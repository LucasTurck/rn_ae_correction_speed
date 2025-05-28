from sklearn.ensemble import RandomForestRegressor
from prediction_f import ModelePrediction, plot_predictions

class ModeleRandomForest(ModelePrediction):
    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        super().__init__('foret_aleatoire', X_train, y_train, X_test, y_test)

    def entrainer(self):
        self.model = RandomForestRegressor()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred_train = self.model.predict(self.X_train)

    def predire(self):
        if self.X_test is not None:
            self.y_pred_test = self.model.predict(self.X_test)
        else:
            self.y_pred_test = None
         
            
    def afficher(self):
        plot_predictions(self.X_train, self.y_train, self.y_pred_train, "Random Forest - Train")
        if self.X_test is not None and self.y_test is not None:
            plot_predictions(self.X_test, self.y_test, self.y_pred_test, "Random Forest - Test")


if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from reed_data import lire_fichier_U
    from dir import DATA_DIRECTORY
    import numpy as np


    parser = argparse.ArgumentParser(description='Random Forest Model Training and Evaluation')
    parser.add_argument('--E', type=int, default=145, help='Experiment number to load data for')
    parser.add_argument('--num_sonde', type=int, default=0, help='Number of the probes to use in the model')
    args = parser.parse_args()

    
    # Créer et entraîner le modèle
    model_rf = ModeleRandomForest()
    
    model_rf.remplissage_donnees(E=args.E, num_sonde=args.num_sonde)
    
    model_rf.entrainer()
    
    print(f"R² pour l'entraînement : {model_rf.R2entrainement()}")
    
    # Afficher les résultats
    model_rf.afficher()