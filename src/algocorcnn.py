from pred_f.cnn import CNN
from dir import DATA_DIRECTORY, MOD_PERSO_DIRECTORY, MOD_DIRECTORY
from reed_data import lire_fichier_U
from simu_X_probes import calcul_U_effective, calculate_speed_vector_using_U_eff
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

class AlgoCorCNN:
    def __init__(self):
        self.cnn = CNN()
        self.k = 0.1
        self.phi = 45
        self.U_mean = 0.0
        self.original_speed = None
        self.simulated_speed = []
        self.corrected_speed = []
        self.b_coord = None
        self.U_2_eff_1 = None
        self.U_2_eff_2 = None

    def load_model(self, model_path=None):
        if model_path is None:
            raise ValueError("Le chemin du modèle doit être spécifié.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le modèle n'existe pas à l'emplacement : {model_path}")
        self.cnn.load_model(model_path)
        self.y_col = self.cnn.parameters['y_col']
        if self.y_col == 1:
            self.b_coord = 2
        elif self.y_col == 2:
            self.b_coord = 1
        else:
            raise ValueError("y_col doit être 1 ou 2 pour indiquer la position de v et w dans le vecteur de vitesse.")
        self.cnn.parameters['nb_test'] = -1
        print(f"Modèle chargé depuis {model_path}")
        
    def set_original_speed(self):
        if self.original_speed is not None:
            return
        self.cnn.load_data()
        self.original_speed = np.array(self.cnn.data[self.cnn.parameters['num_sonde_test']])

    def calculate_U_eff(self):
        if self.original_speed is None:
            self.set_original_speed()
        
        self.U_2_eff_1, self.U_2_eff_2 = calcul_U_effective(self.original_speed, self.k, self.phi, self.U_mean, self.b_coord)
        return self.U_2_eff_1, self.U_2_eff_2
    
    def calculate_speed_vector_from_U_eff(self):
        if self.U_2_eff_1 is None or self.U_2_eff_2 is None:
            self.calculate_U_eff()
        if len(self.corrected_speed) > 0:
            u, v, w = calculate_speed_vector_using_U_eff(self.U_2_eff_1, self.U_2_eff_2, self.k, self.phi, self.U_mean, self.b_coord,
                                                         self.corrected_speed[-1][:,self.y_col])
            self.simulated_speed.append(np.array(list(zip(u, v, w))))
        else:            
            u, v, w = calculate_speed_vector_using_U_eff(self.U_2_eff_1, self.U_2_eff_2, self.k, self.phi, self.U_mean, self.b_coord)
            self.simulated_speed.append(np.array(list(zip(u, v, w))))
        return self.simulated_speed[-1]
    
    def predict_speed(self):
        if self.original_speed is None:
            self.set_original_speed()
        # if not self.cnn.model.is_model_loaded():
        #     raise RuntimeError("Le modèle n'est pas chargé. Veuillez charger le modèle avant de prédire la vitesse.")
        self.cnn.data[self.cnn.parameters['num_sonde_test']] = copy.deepcopy(self.simulated_speed[-1])
        self.cnn.create_data(train = False)
        # Vérification des entrées
        if self.cnn.parameters['timesteps_after'] == 0:
            X = self.cnn.X_test_before
            if X is None:
                raise ValueError("X_test_before est None après create_data. Vérifiez la préparation des données.")
        else:
            X = [self.cnn.X_test_before, self.cnn.X_test_after]
            if self.cnn.X_test_before is None or self.cnn.X_test_after is None:
                raise ValueError("X_test_before ou X_test_after est None après create_data. Vérifiez la préparation des données.")
        self.Y_pred = self.cnn.model.predict(X)
        self.Y_pred = self.Y_pred.reshape(-1,1)
        Y_pred = np.zeros(len(self.original_speed[:,0]))
        Y_pred[self.cnn.parameters['timesteps_before']:-self.cnn.parameters['timesteps_after']-1] = self.Y_pred[:,0]
        if self.y_col == 1:
            self.corrected_speed.append(np.column_stack((self.simulated_speed[-1][:, 0], Y_pred[:], self.simulated_speed[-1][:, 2])))
        elif self.y_col == 2:
            self.corrected_speed.append(np.column_stack((self.simulated_speed[-1][:, 0], self.simulated_speed[-1][:, 1], Y_pred[:])))

    def step_correction(self):
        self.calculate_speed_vector_from_U_eff()
        self.predict_speed()
    
    
    def plot_results(self, n = 5000):
        """Affiche les résultats de la correction."""
        
        times = np.arange(len(self.original_speed))  # Générer un axe temporel simple
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        # plt.scatter(times[:n], self.original_speed[:n, 0], label='u original (m/s)', color='orange', alpha=0.5, s=10, linewidths=0.5)
        plt.plot(times[:n], self.original_speed[:n, 0], label='u original (m/s)', color='orange', alpha=0.5, linewidth=1)
        # plt.scatter(times[:n], self.simulated_speed[:n, 0], label='u simulé (m/s)', color='blue', alpha=0.5, s=10)
        for i in range(len(self.corrected_speed)-1):
            plt.scatter(times[:n], self.corrected_speed[i][:n, 0], label=f'u corrigé {i+1} (m/s)', alpha=0.5, s=i+5)
        plt.plot(times[:n], self.corrected_speed[-1][:n, 0], color='blue', alpha=0.5, linewidth=1)
        plt.legend()
        plt.title('Composante u')
        plt.xlabel('Temps')
        plt.ylabel('Vitesse (m/s)')
        plt.grid()
        plt.subplot(1, 3, 2)
        # plt.scatter(times[:n], self.original_speed[:n, 1], label='v original (m/s)', color='orange', alpha=0.5, s=10)
        plt.plot(times[:n], self.original_speed[:n, 1], label='v original (m/s)', color='orange', alpha=0.5, linewidth=1)
        # plt.scatter(times[:n], self.simulated_speed[:n, 1], label='v simulé (m/s)', color='blue', alpha=0.5, s=10)
        for i in range(len(self.corrected_speed)-1):
            plt.scatter(times[:n], self.corrected_speed[i][:n, 1], label=f'v corrigé {i+1} (m/s)', alpha=0.5, s=i+5)
        plt.plot(times[:n], self.corrected_speed[-1][:n, 1], label=f'v corrigé {len(self.corrected_speed)} (m/s)', color='blue', alpha=0.5, linewidth=1)
        # plt.legend()
        plt.title('Composante v')
        plt.xlabel('Temps')
        plt.ylabel('Vitesse (m/s)')
        plt.grid()
        plt.tight_layout()
        plt.subplot(1, 3, 3)
        # plt.scatter(times[:n], self.original_speed[:n, 2], label='w original (m/s)', color='orange', alpha=0.5, s=10)
        plt.plot(times[:n], self.original_speed[:n, 2], label='w original (m/s)', color='orange', alpha=0.5, linewidth=1)
        # plt.scatter(times[:n], self.simulated_speed[:n, 2], label='w simulé (m/s)', color='blue', alpha=0.5, s=10)
        for i in range(len(self.corrected_speed)-1):
            plt.scatter(times[:n], self.corrected_speed[i][:n, 2], label=f'w corrigé {i+1} (m/s)', alpha=0.5, s=i+5)
        plt.plot(times[:n], self.corrected_speed[-1][:n, 2], label=f'w corrigé {len(self.corrected_speed)} (m/s)', color='blue', alpha=0.5, linewidth=1)
        # plt.legend()
        plt.title('Composante w')
        plt.xlabel('Temps')
        plt.ylabel('Vitesse (m/s)')
        plt.grid()
        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    algo = AlgoCorCNN()
    model_path = os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250627_145806_b53de2fd')
    algo.load_model(model_path)
    
    algo.set_original_speed()
    algo.calculate_U_eff()
    
    for i in range(10):
        algo.step_correction()
        print(f"Iteration {i+1}: Vitesse corrigée calculée.")
        try:
            r2 = r2_score(algo.original_speed, algo.corrected_speed[-1])
            print(f"R² score: {r2}")
        except Exception as e:
            print(f"Erreur lors du calcul du R² score: {e}")
    algo.plot_results()
    print("Affichage des résultats terminé.")