from reed_data import lire_fichier_U
from dir import DATA_DIRECTORY
import os
import numpy as np
import matplotlib.pyplot as plt




def calcul_U_effective(speed_vector, k, phi, U_mean, b_coord = 1):
    """
    speed_vector : array-like de forme (..., 3)
    Retourne deux tableaux U_eff_1, U_eff_2 de même forme que speed_vector[..., 0]
    """
    speed_vector = np.asarray(speed_vector)
    a = speed_vector[..., 0]
    if b_coord == 1: 
        b = speed_vector[..., 1]
        c = speed_vector[..., 2]
    elif b_coord == 2:
        b = speed_vector[..., 2]
        c = speed_vector[..., 1]
    else:
        raise ValueError("b_coord doit être 1 ou 2 pour indiquer la position de v et w dans le vecteur de vitesse.")
    phi_rad = np.deg2rad(phi)  # phi en radians pour np.cos/np.sin

    U_2_eff_1 = (U_mean + a) ** 2 + b**2 + c**2 - (1 - k**2) * ((U_mean + a) * np.cos(phi_rad) - b * np.sin(phi_rad))**2

    U_2_eff_2 = (U_mean + a) ** 2 + b**2 + c**2 - (1 - k**2) * ((U_mean + a) * np.cos(-phi_rad) - b * np.sin(-phi_rad))**2

    return U_2_eff_1, U_2_eff_2

def calculate_speed_vector_using_U_eff(U_2_eff_1, U_2_eff_2, k, phi, U_mean, b_coord = 1, c = 0):
    """
    U_eff_1, U_eff_2 : array-like de même forme
    Retourne u, v, w sous forme de tableaux numpy
    """
    U_2_eff_1 = np.asarray(U_2_eff_1) - c**2
    U_2_eff_2 = np.asarray(U_2_eff_2) - c**2
    phi_rad = np.deg2rad(phi)

    K = (1 - k**2)
    c_1 = U_2_eff_1 / (1 - K * np.cos(phi_rad)**2)
    c_2 = U_2_eff_2 / (1 - K * np.cos(phi_rad)**2)
    aa = (1 - K * np.sin(phi_rad)**2) / (1 - K * np.cos(phi_rad)**2)
    bb = - K * np.sin(2 * phi_rad) / (1 - K * np.cos(phi_rad)**2)
    A = (np.sqrt(aa) * (c_2 - c_1)/bb + (c_1 + c_2)/2)/4
    B = (np.sqrt(aa) * (c_1 - c_2)/bb + (c_1 + c_2)/2)/4

    # print("A min/max:", np.min(A), np.max(A))
    # print("B min/max:", np.min(B), np.max(B))
    # print("a min/max:", np.min(aa), np.max(aa))
    # print("b min/max:", np.min(bb), np.max(bb))
    # print("c_1 min/max:", np.min(c_1), np.max(c_1))
    # print("c_2 min/max:", np.min(c_2), np.max(c_2))

    a = np.sqrt(A) + np.sqrt(B) - U_mean
    b = (np.sqrt(A) - np.sqrt(B))/np.sqrt(aa)
    c = np.zeros_like(a)
    if b_coord == 1:
        return a, b, c
    elif b_coord == 2:
        return a, c, b
    else:
        raise ValueError("b_coord doit être 1 ou 2 pour indiquer la position de v et w dans le vecteur de vitesse.")

if __name__ == "__main__":
    # k = 0.00001     # k denotes the tangential sensitivity coefficient
    k = 0.1
    phi = 45       # Angle of the probe in degrees to the mean flow direction
    U_mean = 3.5  # Mean flow speed in m/s
    b_coord = 2  # Indique la position de v et w dans le vecteur de vitesse (1 pour v en 2ème composante, 2 pour w en 2ème composante)

    times, sondes = lire_fichier_U(os.path.join(DATA_DIRECTORY, 'E_125', 'U'))
    
    speeds = sondes[0][1]  # On prend les vitesses de la première sonde
    
    n = 100  # Nombre de points à afficher
    times = np.array(times[:n])
    speeds = np.array(speeds[:n])

    U_eff_1, U_eff_2 = calcul_U_effective(speeds, k, phi, U_mean, b_coord=b_coord)
    # u, v, w = calculate_speed_vector_using_U_eff(U_eff_1, U_eff_2, k, phi, U_mean, w = speeds[:, 2])  # On utilise la composante w de speeds pour w
    u, v, w = calculate_speed_vector_using_U_eff(U_eff_1, U_eff_2, k, phi, U_mean, b_coord=b_coord)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.scatter(times, u, label='u probes (m/s)', color='r', alpha=0.5, s=10)
    plt.scatter(times, speeds[:, 0], label='u original (m/s)', color='orange', alpha=0.5, s=10)
    plt.legend()
    plt.title('Composante u')
    plt.xlabel('Temps (s)')
    plt.ylabel('Vitesse (m/s)')
    plt.grid()
    plt.subplot(1, 3, 2)
    if b_coord == 1:
        plt.scatter(times, v, label='v probes (m/s)', color='g', alpha=0.5, s=10)
        plt.scatter(times, speeds[:, 1], label='v original (m/s)', color='orange', alpha=0.5, s=10)
        plt.legend()
        plt.title('Composante v')
    elif b_coord == 2:
        plt.scatter(times, w, label='w probes (m/s)', color='g', alpha=0.5, s=10)
        plt.scatter(times, speeds[:, 2], label='w original (m/s)', color='orange', alpha=0.5, s=10)
        plt.legend()
        plt.title('Composante w')
    plt.xlabel('Temps (s)')
    plt.ylabel('Vitesse (m/s)')
    plt.grid()
    plt.subplot(1, 3, 3)
    if b_coord == 1:
        plt.scatter(u, v, label='u,v probes', color='b', alpha=0.5, s=10)
        plt.scatter(speeds[:, 0], speeds[:, 1], label='u,v original', color='orange', alpha=0.5, s=10)
        plt.legend()
        plt.title('Composante u,v')
        plt.xlabel('u (m/s)')
        plt.ylabel('v (m/s)')
    elif b_coord == 2:
        plt.scatter(u, w, label='u,w probes', color='b', alpha=0.5, s=10)
        plt.scatter(speeds[:, 0], speeds[:, 2], label='u,w original', color='orange', alpha=0.5, s=10)
        plt.legend()
        plt.title('Composante u,w')
        plt.xlabel('u (m/s)')
        plt.ylabel('w (m/s)')
    plt.grid()
    plt.tight_layout()
    plt.show()