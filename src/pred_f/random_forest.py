from sklearn.ensemble import RandomForestRegressor
from prediction_f import ModelePrediction, plot_predictions, args_to_dict

class ModeleRandomForest(ModelePrediction):
    def __init__(self, parameters=None):
        super().__init__('foret_aleatoire', parameters)
        
    def init_parameters(self):
        super().init_parameters()

    def entrainer(self):
        self.model = RandomForestRegressor(verbose=1)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred_train = self.model.predict(self.X_train)

    def predire(self):
        if self.X_test is not None:
            self.y_pred_test = self.model.predict(self.X_test)
        else:
            self.y_pred_test = None
         
            
    def afficher(self):
        plot_predictions(self.X_train, self.y_train, self.y_pred_train, "Train - Random Forest", block=False)
        if self.X_test is not None and self.y_test is not None:
            plot_predictions(self.X_test, self.y_test, self.y_pred_test, "Test - Random Forest", block=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Random Forest Model Training and Evaluation')
    parser.add_argument('--E', type=int, help='Experiment number to load data for')
    parser.add_argument('--num_sonde_train', type=int, help='Number of the probes to use in the model')
    parser.add_argument('--num_sonde_test', type=int, help='Number of the probes to use in the model for testing')
    parser.add_argument('--nb_training', type=int, help='Number of training samples to use, -1 for all')
    parser.add_argument('--nb_test', type=int, help='Number of test samples to use, -1 for all')
    args = parser.parse_args()
    args_dict = args_to_dict(args)

    
    # Créer et entraîner le modèle
    model_rf = ModeleRandomForest(parameters=args_dict)

    model_rf.remplissage_donnees()

    model_rf.entrainer()
    
    print(f"R² pour l'entraînement : {model_rf.R2entrainement()}")
    
    # Remplir les données de test
    model_rf.remplissage_donnees(train=False)

    # Prédire les valeurs sur les données de test
    model_rf.predire()
    
    print(f"R² pour le test : {model_rf.evaluer()}")

    # Afficher les résultats
    model_rf.afficher()