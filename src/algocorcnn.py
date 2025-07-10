from pred_f.cnn import CNN
from dir import DATA_DIRECTORY, MOD_PERSO_DIRECTORY, MOD_DIRECTORY
from reed_data import lire_fichier_U
from simu_X_probes import calcul_U_effective, calculate_speed_vector_using_U_eff
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy.signal import butter, filtfilt

class AlgoCorCNN:
    def __init__(self):
        self.cnn = CNN()
        self.cnn_start = CNN()
        self.k = 0.2
        self.phi = 45
        self.U_mean = 0.0
        self.original_speed = None
        self.simulated_speed = []
        self.corrected_speed = []
        self.b_coord = None
        self.U_2_eff_1 = None
        self.U_2_eff_2 = None
        self.correction_factor = 0.1
        self.r2_score = []

    def load_model(self, model_path=None, start = False):
        if model_path is None:
            raise ValueError("Le chemin du modèle doit être spécifié.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le modèle n'existe pas à l'emplacement : {model_path}")
        if start:
            self.cnn_start.load_model(model_path)
        else:
            self.cnn.load_model(model_path)
        self.y_col = self.cnn_start.parameters['y_col']
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
        self.cnn_start.load_data()
        self.cnn.load_data()
        self.original_speed = np.array(self.cnn_start.data[self.cnn_start.parameters['num_sonde_test']])

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
            if self.y_col == 1:
                self.simulated_speed.append(np.array(list(zip(u, self.corrected_speed[-1][:, 1], w))))
            elif self.y_col == 2:
                self.simulated_speed.append(np.array(list(zip(u, v, self.corrected_speed[-1][:, 2]))))
        else:            
            u, v, w = calculate_speed_vector_using_U_eff(self.U_2_eff_1, self.U_2_eff_2, self.k, self.phi, self.U_mean, self.b_coord)
            self.simulated_speed.append(np.array(list(zip(u, v, w))))
        return self.simulated_speed[-1]
    
    def predict_speed(self, start = False, correction_factor = 0.1):
        if self.original_speed is None:
            self.set_original_speed()
        # if not self.cnn.model.is_model_loaded():
        #     raise RuntimeError("Le modèle n'est pas chargé. Veuillez charger le modèle avant de prédire la vitesse.")
        if start:
            cnn = self.cnn_start
        else:
            cnn = self.cnn
            
        cnn.data[cnn.parameters['num_sonde_test']] = copy.deepcopy(self.simulated_speed[-1])
        cnn.create_data(train = False)
        cnn.add_power(train = False)

        # Vérification des entrées
        if cnn.parameters['timesteps_after'] == 0:
            X = cnn.X_test_before
            if X is None:
                raise ValueError("X_test_before est None après create_data. Vérifiez la préparation des données.")
        else:
            X = [cnn.X_test_before, cnn.X_test_after]
            if cnn.X_test_before is None or cnn.X_test_after is None:
                raise ValueError("X_test_before ou X_test_after est None après create_data. Vérifiez la préparation des données.")
        self.Y_pred = cnn.model.predict(X)
        self.Y_pred = self.Y_pred.reshape(-1,1)
        Y_pred = copy.deepcopy(cnn.data[cnn.parameters['num_sonde_test']][:, self.y_col])
        # Y_pred = np.zeros(len(self.original_speed[:,0]))
        Y_pred[cnn.parameters['timesteps_before']-1:-cnn.parameters['timesteps_after']-2] = self.Y_pred[:,0]
        if start:
            Y_pred = filtre_passe_bas(Y_pred, fs = 1/0.0006, fc=50, order = 4)
        else:
            Y_pred = filtre_passe_bas(Y_pred, fs = 1/0.0006, fc=500, order = 4)
        
        # print(f"Y_pred: {Y_pred[:5]}")  # Afficher les 5 premières valeurs de Y_pred pour débogage
        
        if self.y_col == 1:
            self.corrected_speed.append(np.column_stack((self.simulated_speed[-1][:, 0], Y_pred[:], self.simulated_speed[-1][:, 2])))
        elif self.y_col == 2:
            self.corrected_speed.append(np.column_stack((self.simulated_speed[-1][:, 0], self.simulated_speed[-1][:, 1], Y_pred[:])))

        if len(self.corrected_speed) > 1:
            self.corrected_speed[-1] = correction_factor * self.corrected_speed[-1] + (1 - correction_factor) * self.corrected_speed[-2]

    def step_correction(self, start = False, correction_factor = 0.1):
        self.predict_speed(start=start, correction_factor=correction_factor)
        self.calculate_speed_vector_from_U_eff()
    
    def compute_r2_score(self):
        if self.original_speed is None or self.corrected_speed is None:
            raise ValueError("Les vitesses originales et corrigées doivent être définies pour calculer le score R².")
        r2 = r2_score(self.original_speed, self.corrected_speed[-1])
        r2_u = r2_score(self.original_speed[:, 0], self.corrected_speed[-1][:, 0])
        r2_v = r2_score(self.original_speed[:, 1], self.corrected_speed[-1][:, 1])
        r2_w = r2_score(self.original_speed[:, 2], self.corrected_speed[-1][:, 2])
        self.r2_score.append([r2_u, r2_v, r2_w, r2])
        return r2

    def plot_r2_scores(self):
        """Affiche les scores R² au fil des itérations."""
        plt.figure(figsize=(10, 5))
        self.r2_score = np.array(self.r2_score)
        plt.plot(self.r2_score[:, 0], marker='o', linestyle='-', color='blue', label='R² u')
        plt.plot(self.r2_score[:, 1], marker='o', linestyle='-', color='green', label='R² v')
        plt.plot(self.r2_score[:, 2], marker='o', linestyle='-', color='red', label='R² w')
        plt.plot(self.r2_score[:, 3], marker='o', linestyle='-', color='purple', label='R² total')
        plt.title('Scores R² au fil des itérations')
        plt.legend()
        plt.xlabel('Itération')
        plt.ylabel('R² Score')
        plt.grid()
        # plt.show(block=False)
        plt.show(block=True)

    def plot_results(self, n_steps, k_steps, l_steps, n = 5000):
        """Affiche les résultats de la correction."""
        
        times = np.arange(len(self.original_speed))  # Générer un axe temporel simple
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        # plt.scatter(times[:n], self.original_speed[:n, 0], label='u original (m/s)', color='orange', alpha=0.5, s=10, linewidths=0.5)
        plt.plot(times[:n], self.original_speed[:n, 0], label='u original (m/s)', color='orange', alpha=0.5, linewidth=1)
        plt.scatter(times[:n], self.simulated_speed[0][:n, 0], label='u simulé (m/s)', color='blue', alpha=0.5, s=10)
        for i in range(n_steps):
            plt.scatter(times[:n], self.corrected_speed[i*(k_steps+l_steps)][:n, 0], label=f'u corrigé {i+1} (m/s)', alpha=0.5, s=i+5)
        plt.plot(times[:n], self.corrected_speed[-1][:n, 0], color='blue', alpha=0.5, linewidth=1)
        # plt.legend()
        plt.title('Composante u')
        plt.xlabel('Temps')
        plt.ylabel('Vitesse (m/s)')
        plt.grid()
        plt.subplot(1, 3, 2)
        # plt.scatter(times[:n], self.original_speed[:n, 1], label='v original (m/s)', color='orange', alpha=0.5, s=10)
        plt.plot(times[:n], self.original_speed[:n, 1], label='v original (m/s)', color='orange', alpha=0.5, linewidth=1)
        # plt.scatter(times[:n], self.simulated_speed[0][:n, 1], label='v simulé (m/s)', color='blue', alpha=0.5, s=10)
        for i in range(n_steps):
            plt.scatter(times[:n], self.corrected_speed[i*(k_steps+l_steps)][:n, 1], label=f'v corrigé {i+1} (m/s)', alpha=0.5, s=i+5)
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
        # plt.scatter(times[:n], self.simulated_speed[0][:n, 2], label='w simulé (m/s)', color='blue', alpha=0.5, s=10)
        for i in range(n_steps):
            plt.scatter(times[:n], self.corrected_speed[i*(k_steps+l_steps)][:n, 2], label=f'w corrigé {i+1} (m/s)', alpha=0.5, s=i+5)
        plt.plot(times[:n], self.corrected_speed[-1][:n, 2], label=f'w corrigé {len(self.corrected_speed)} (m/s)', color='blue', alpha=0.5, linewidth=1)
        # plt.legend()
        plt.title('Composante w')
        plt.xlabel('Temps')
        plt.ylabel('Vitesse (m/s)')
        plt.grid()
        plt.tight_layout()
        plt.show()


def filtre_passe_bas(signal, fs, fc, order=4):
    
    w = fc / (fs / 2)  # Normalisation de la fréquence de coupure
    b, a = butter(order, w, btype='low')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# if __name__ == "__main__":
#     algo = AlgoCorCNN()
#     model_path_start = os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250702_172926_9a1a3296')
#     # model_path_start = os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250708_160938_bd68652b')
#     algo.load_model(model_path_start, start=True)
#     # model_path = os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250707_175213_754c67cc')
#     # model_path = os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250708_180136_fbcbb8cb')
#     model_path = os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250709_114355_f87f582d')
#     algo.load_model(model_path, start=False)

#     algo.set_original_speed()
#     algo.calculate_U_eff()
    
#     algo.calculate_speed_vector_from_U_eff()
    
#     print(f"R2 score initial: {r2_score(algo.original_speed, algo.simulated_speed[0])}")
    
#     algo.predict_speed(start=True, correction_factor=1)
#     algo.simulated_speed.append(copy.deepcopy(algo.corrected_speed[-1]))
#     algo.compute_r2_score()
#     try:
#         r2 = algo.compute_r2_score()
#         print(f"R² score après la première correction: {r2}")
#     except Exception as e:
#         print(f"Erreur lors du calcul du R² score: {e}")
        
#     for i in range(30):
#         algo.predict_speed(start=False, correction_factor=1)
#         algo.simulated_speed.append(copy.deepcopy(algo.corrected_speed[-1]))
#         print(f"Étape {i+1} de la prédiction de vitesse terminée.")
#         try:
#             r2 = algo.compute_r2_score()
#             print(f"R² score après la prédiction {i+1}: {r2}")
#         except Exception as e:
#             print(f"Erreur lors du calcul du R² score: {e}")
#     print("Prédiction de vitesse terminée. Début de la correction.")
    
#     for i in range(30):
#         print(f"Étape de correction {i+1} en cours...")
#         algo.step_correction(start=False)
#         # print(f"simulated_speed: {algo.simulated_speed[-1][:5]}")
#         # print(f"corrected_speed: {algo.corrected_speed[-1][:5]}")
#         # print(f"Iteration {i+1}: Vitesse corrigée calculée.")
#         try:
#             r2 = algo.compute_r2_score()
#             print(f"R² score: {r2}")
#         except Exception as e:
#             print(f"Erreur lors du calcul du R² score: {e}")
    
#     print(f"R² scores: {algo.r2_score}")
#     algo.plot_r2_scores()
#     # algo.plot_results()
#     print("Affichage des résultats terminé.")

if __name__ == "__main__":
    algo = AlgoCorCNN()
    model_path_start = os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250702_172926_9a1a3296')
    algo.load_model(model_path_start, start=True)
    # model_path = os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250709_114355_f87f582d')
    model_path = os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250707_175213_754c67cc')
    algo.load_model(model_path, start=False)
    
    algo.set_original_speed()
    algo.calculate_U_eff()
    
    algo.calculate_speed_vector_from_U_eff()
    
    print(f"R2 score initial: {r2_score(algo.original_speed, algo.simulated_speed[0])}")
    
    algo.predict_speed(start=True, correction_factor=1)
    algo.simulated_speed.append(copy.deepcopy(algo.corrected_speed[-1]))
    algo.compute_r2_score()
    print(f"R² score après la première correction: {algo.r2_score[-1]}")
    
    n_steps = 2
    k_steps = 5
    l_steps = 3
    for i in range(n_steps):
        for j in range(k_steps):
            algo.predict_speed(start=False, correction_factor=1)
            algo.simulated_speed.append(copy.deepcopy(algo.corrected_speed[-1]))
            print(f"Étape {i+1}.{j+1} de la prédiction de vitesse terminée.")
            r2 = algo.compute_r2_score()
            print(f"R² score après la prédiction {i+1}.{j+1}: {r2}")
        for j in range(l_steps):
            algo.step_correction(start=False, correction_factor = algo.correction_factor)
            print(f"Étape {i+1}.{j+1} de la correction de vitesse terminée.")
            r2 = algo.compute_r2_score()
            print(f"R² score après la correction {i+1}.{j+1}: {r2}")

    algo.plot_r2_scores()
    algo.plot_results(n_steps, k_steps, l_steps)