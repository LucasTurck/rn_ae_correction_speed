from pred_f.reseau_neurones import ModeleReseauNeurones
from dir import DATA_DIRECTORY, MOD_PERSO_DIRECTORY, MOD_DIRECTORY
from reed_data import lire_fichier_U
from simu_X_probes import calcul_U_effective, calculate_speed_vector_using_U_eff
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

class AlgoCorrection:
    def __init__(self):
        self.model = ModeleReseauNeurones()
        self.original_speed = None
        self.simulated_speed = []
        self.corrected_speed = []
        self.correction_factor = 0.1  # Facteur de correction pour la vitesse

    def charger_model(self, run_path):
        """Charge le modèle de réseau de neurones."""
        self.model.charger_apprentissage(run_path)
        print(f"paramètres du modèle chargés : {self.model.parameters}")
    
    def set_original_speed(self):
        times, sondes = lire_fichier_U(os.path.join(DATA_DIRECTORY, f"E_{self.model.parameters['E']}", 'U'))
        self.original_speed = np.array(copy.deepcopy(sondes[self.model.parameters['num_sonde_test']][1]))
        self.original_speed = np.array(list(zip(self.original_speed[:, 0], self.original_speed[:, 1], self.original_speed[:, 2])))

    def set_simulated_speed(self):
        if self.original_speed is None:
            raise ValueError("La vitesse originale doit être définie avant de définir la vitesse simulée.")
        
        k = self.model.parameters['k']
        phi = self.model.parameters['phi']
        U_mean = self.model.parameters['U_mean']
        y = self.model.parameters['y']
        if y == 1:
            b_coord = 2
        elif y == 2:
            b_coord = 1
        else:
            raise ValueError("y doit être 1 ou 2 pour indiquer la position de v et w dans le vecteur de vitesse.")

        U_2_eff_1, U_2_eff_2 = calcul_U_effective(self.original_speed, k, phi, U_mean, b_coord=b_coord)
        u, v, w = calculate_speed_vector_using_U_eff(U_2_eff_1, U_2_eff_2, k, phi, U_mean, b_coord=b_coord)
        self.simulated_speed.append(np.array(list(zip(u, v, w))))
        if not self.simulated_speed:
            raise ValueError("La vitesse simulée est vide. Vérifiez les paramètres et les données d'entrée.")

    def add_noise_in_simulated_speed(self, noise_level=0.1):
        """Ajoute du bruit à la vitesse simulée."""
        if not self.simulated_speed:
            raise ValueError("La vitesse simulée doit être définie avant d'ajouter du bruit.")
        
        # noise = np.random.normal(0, noise_level, self.simulated_speed[-1].shape)
        # self.simulated_speed[-1] += noise
        # self.simulated_speed[-1]*=(1- noise_level)
        print(f"Bruit ajouté à la vitesse simulée : {self.simulated_speed[-1][:5]}...")

    def correct_speed2(self):
        y = self.model.parameters['y']
        if self.original_speed is None or not self.simulated_speed:
            raise ValueError("Les vitesses originales et simulées doivent être définies avant la correction.")
        if len(self.corrected_speed) == 0:
            self.corrected_speed.append(copy.deepcopy(self.original_speed))
            # self.corrected_speed.append(copy.deepcopy(self.simulated_speed[0]))
            self.corrected_speed[-1][:,y] += np.random.normal(0, 0.1, len(self.corrected_speed[-1][:,y]))  # Ajouter un bruit initial à la vitesse corrigée
        data = self.corrected_speed[-1] if len(self.corrected_speed) > 0 else self.simulated_speed[0]

        X = []
        p_start, f_start, h_start = self.model.parameters['prediction_start']
        p_end, f_end, h_end = self.model.parameters['prediction_end']
        timesteps = self.model.parameters['timesteps']
        n_start = max(p_start + (p_end - p_start) * timesteps,
                      f_start + (f_end - f_start) * timesteps,
                      h_start + (h_end - h_start) * timesteps)
        for i in range(n_start, len(data)):
            feature_vector = []
            for j in range(timesteps):
                timesteps_vector = []
                for k in range(p_start, p_end):
                    timesteps_vector.append(data[i - j - k][0])  # u
                for k in range(f_start, f_end):
                    timesteps_vector.append(data[i - j - k][1])  # v
                for k in range(h_start, h_end):
                    timesteps_vector.append(data[i - j - k][2])  # w
                feature_vector.append(timesteps_vector)
            X.append(feature_vector[::-1])  # Inverser l'ordre des features
        
        # print(f"y: {y}")
        
        # Utiliser data_array pour l'indexation multidimensionnelle

        c = np.zeros(len(data))
        c[:n_start] = self.original_speed[:n_start, y]
        predictions = self.model.model.predict(np.array(X)).flatten()
        print(f"max_idx: {n_start}, c shape: {c.shape}, data shape: {data.shape}, predictions shape: {predictions.shape}")
        c[n_start:] = predictions
        c = np.array(c)
        
        print(f"c : {c[:5]}...")
        
        if len(c) != len(data):
            print(f"Longueur des données corrigées: {c.shape}, Longueur des données originales: {data.shape}")
            raise ValueError("La longueur des données corrigées ne correspond pas à la longueur des données originales.")
        
        if y == 1:
            self.simulated_speed.append(np.column_stack((data[:,0], c[:], data[:,2])))
        elif y == 2:
            self.simulated_speed.append(np.column_stack((data[:,0], data[:,1], c[:])))
        else:
            raise ValueError("y doit être 1 ou 2 pour indiquer la position de v et w dans le vecteur de vitesse.")
        
        print(f"Vitesse simulée : {self.simulated_speed[-1][:5]}...")  # Afficher les 5 premières valeurs
        
        k = self.model.parameters['k']
        phi = self.model.parameters['phi']
        U_mean = self.model.parameters['U_mean']
        if y == 1:
            b_coord = 2
        elif y == 2:
            b_coord = 1
        
        U_2_eff_1, U_2_eff_2 = calcul_U_effective(self.simulated_speed[-1], k, phi, U_mean, b_coord=b_coord)
        u, v, w = calculate_speed_vector_using_U_eff(U_2_eff_1, U_2_eff_2, k, phi, U_mean, b_coord=b_coord, c=c)
        print(f"u, v, w après correction : {u[:5]}, {v[:5]}, {w[:5]}...")  # Afficher les 5 premières valeurs
        if y == 1:
            self.corrected_speed.append(np.array(list(zip(u, c, w))))
        elif y == 2:
            self.corrected_speed.append(np.array(list(zip(u, v, c))))
        if len(self.corrected_speed) > 1:
            self.corrected_speed[-1] = self.correction_factor * self.corrected_speed[-1]  + (1 - self.correction_factor) * self.corrected_speed[-2]
        # else:
        #     self.corrected_speed[-1] = self.correction_factor * self.corrected_speed[-1] + (1 - self.correction_factor) * self.simulated_speed[0]
        print(f"Vitesse corrigée après correction : {self.corrected_speed[-1][:5]}...")  # Afficher les 5 premières valeurs
    
    def correct_speed(self):
        """Corrige la vitesse simulée en utilisant le modèle de réseau de neurones."""
        
        y = self.model.parameters['y']
        if self.original_speed is None or not self.simulated_speed:
            raise ValueError("Les vitesses originales et simulées doivent être définies avant la correction.")
        if len(self.corrected_speed) == 0:
            # self.corrected_speed.append(copy.deepcopy(self.original_speed[0]))
            self.corrected_speed.append(copy.deepcopy(self.simulated_speed[0]))
            self.corrected_speed[-1][:,y] = self.original_speed[:,y] + np.random.normal(0, 0.01, len(self.corrected_speed[-1][:,y]))
            # self.corrected_speed[-1][:,0] = self.original_speed[:,0]  
            # self.corrected_speed[-1][:,2] = self.original_speed[:,2]  # Initialiser les autres composantes
            # self.corrected_speed[-1][:,1] = self.original_speed[:,1]  # Initialiser la composante v ou w selon y
        data = self.corrected_speed[-1] if len(self.corrected_speed) > 0 else self.simulated_speed[0]
        
        k = self.model.parameters['k']
        phi = self.model.parameters['phi']
        U_mean = self.model.parameters['U_mean']
        if y == 1:
            b_coord = 2
        elif y == 2:
            b_coord = 1
            
        U_2_eff_1, U_2_eff_2 = calcul_U_effective(data, k, phi, U_mean, b_coord=b_coord)
        u, v, w = calculate_speed_vector_using_U_eff(U_2_eff_1, U_2_eff_2, k, phi, U_mean, b_coord=b_coord)
        
        if y == 1:
            v = data[:, 1]  # v est la deuxième composante
        elif y == 2:
            w = data[:, 2]
        
        X = []
        p_start, f_start, h_start = self.model.parameters['prediction_start']
        p_end, f_end, h_end = self.model.parameters['prediction_end']
        timesteps = self.model.parameters['timesteps']
        n_start = max(p_start + (p_end - p_start) * timesteps,
                      f_start + (f_end - f_start) * timesteps,
                      h_start + (h_end - h_start) * timesteps)
        for i in range(n_start, len(u)):
            feature_vector = []
            for j in range(timesteps):
                timesteps_vector = []
                for l in range(p_start, p_end):
                    timesteps_vector.append(u[i - j - l])
                for l in range(f_start, f_end):
                    timesteps_vector.append(v[i - j - l])
                for l in range(h_start, h_end):
                    timesteps_vector.append(w[i - j - l])
                feature_vector.append(timesteps_vector)
            X.append(feature_vector[::-1])  # Inverser l'ordre des features
        
        c = np.zeros(len(u))
        c[:n_start] = self.original_speed[:n_start, y]
        # c[:n_start] = data[:n_start, y]  # Utiliser la vitesse originale ou simulée pour les premières valeurs
        predictions = self.model.model.predict(np.array(X)).flatten()
        print(f"max_idx: {n_start}, c shape: {c.shape}, data shape: {data.shape}, predictions shape: {predictions.shape}")
        c[n_start:] = predictions
        c = np.array(c)
        
        print(f"c : {c[:5]}...")
        
        if len(c) != len(data):
            print(f"Longueur des données corrigées: {c.shape}, Longueur des données originales: {data.shape}")
            raise ValueError("La longueur des données corrigées ne correspond pas à la longueur des données originales.")
        
        u, v, w = calculate_speed_vector_using_U_eff(U_2_eff_1, U_2_eff_2, k, phi, U_mean, b_coord=b_coord, c=c)
        print(f"u, v, w après correction : {u[:5]}, {v[:5]}, {w[:5]}...")
        
        if y == 1:
            self.corrected_speed.append(np.column_stack((u, c, w)))
        elif y == 2:
            self.corrected_speed.append(np.column_stack((u, v, c)))
        else:
            raise ValueError("y doit être 1 ou 2 pour indiquer la position de v et w dans le vecteur de vitesse.")
        if len(self.corrected_speed) > 1:
            self.corrected_speed[-1] = self.correction_factor * self.corrected_speed[-1]  + (1 - self.correction_factor) * self.corrected_speed[-2]
        print(f"Vitesse corrigée : {self.corrected_speed[-1][:5]}...")  # Afficher les 5 premières valeurs

        

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
    algo = AlgoCorrection()
    # algo.charger_model(os.path.join(MOD_PERSO_DIRECTORY, 'lstm_deep_dense', 'run_20250611_171132_e5f7cebe'))
    # algo.charger_model(os.path.join(MOD_PERSO_DIRECTORY, 'lstm_deep_dense', 'run_20250611_172014_256d71f6'))
    # algo.charger_model(os.path.join(MOD_PERSO_DIRECTORY, 'lstm_deep_dense', 'run_20250612_102729_2a9e54f8'))
    # algo.charger_model(os.path.join(MOD_DIRECTORY, 'lstm_deep_dense', 'run_20250611_193227_f50f7962'))
    # algo.charger_model(os.path.join(MOD_DIRECTORY, 'deep_lstm_bn_dropout', 'run_20250611_195237_f4146bff'))
    # algo.charger_model(os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250616_154321_bf051922'))
    # algo.charger_model(os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250616_162556_fcf864a8'))
    # algo.charger_model(os.path.join(MOD_DIRECTORY,'cnn_lstm','run_20250617_100619_7e5a9a9c'))
    # algo.charger_model(os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250617_141735_54a73b26'))
    # algo.charger_model(os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250617_144539_06fae9f7'))
    algo.charger_model(os.path.join(MOD_PERSO_DIRECTORY, 'cnn_lstm', 'run_20250617_150506_d6e15fd7'))


    n = 200
    
    algo.set_original_speed()
    algo.set_simulated_speed()
    algo.add_noise_in_simulated_speed(noise_level=0.1)  # Ajouter du bruit à la vitesse simulée
    
    for i in range(10):  # Effectuer n itérations de correction
        algo.correct_speed()
        # print(f"Correction {i+1} effectuée : {algo.corrected_speed[-1][ : 5]}...")  # Afficher les 5 premières valeurs de la dernière correction
        try:
            print(f"R2 pour la correction {i+1} : {r2_score(algo.original_speed[:n, algo.model.parameters['y']], algo.corrected_speed[-1][:n, algo.model.parameters['y']])}")
        except Exception as e:
            print(f"Erreur lors du calcul de R2 pour la correction {i+1} : {e}")
    algo.plot_results(n=n)  # Afficher les résultats de la correction
