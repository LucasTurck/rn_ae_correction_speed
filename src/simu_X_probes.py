from reed_data import lire_fichier_U
from dir import DATA_DIRECTORY
import os
import numpy as np
import matplotlib.pyplot as plt




def calcul_U_effective(speed_vector, k, phi, U_mean):
    """
    speed_vector : array-like de forme (..., 3)
    Retourne deux tableaux U_eff_1, U_eff_2 de même forme que speed_vector[..., 0]
    """
    speed_vector = np.asarray(speed_vector)
    u = speed_vector[..., 0]
    v = speed_vector[..., 1]
    w = speed_vector[..., 2]
    phi_rad = np.deg2rad(phi)  # phi en radians pour np.cos/np.sin

    U_2_eff_1 = (U_mean + u) ** 2 + v**2 + w**2 - (1 - k**2) * ((U_mean + u) * np.cos(phi_rad) - v * np.sin(phi_rad))**2
    # U_eff_1 = u ** 2 + v**2 + w**2 - (1 - k**2) * ((U_mean + u) * np.cos(phi_rad) - v * np.sin(phi_rad))**2

    U_2_eff_2 = (U_mean + u) ** 2 + v**2 + w**2 - (1 - k**2) * ((U_mean + u) * np.cos(-phi_rad) - v * np.sin(-phi_rad))**2
    # U_eff_2 = u ** 2 + v**2 + w**2 - (1 - k**2) * ((U_mean + u) * np.cos(-phi_rad) - v * np.sin(-phi_rad))**2

    return U_2_eff_1, U_2_eff_2

def calculate_speed_vector_using_U_eff(U_2_eff_1, U_2_eff_2, k, phi, U_mean, w = 0):
    """
    U_eff_1, U_eff_2 : array-like de même forme
    Retourne u, v, w sous forme de tableaux numpy
    """
    U_2_eff_1 = np.asarray(U_2_eff_1) - w**2
    U_2_eff_2 = np.asarray(U_2_eff_2) - w**2
    phi_rad = np.deg2rad(phi)

    K = (1 - k**2)
    c_1 = U_2_eff_1 / (1 - K * np.cos(phi_rad)**2)
    c_2 = U_2_eff_2 / (1 - K * np.cos(phi_rad)**2)
    a = (1 - K * np.sin(phi_rad)**2) / (1 - K * np.cos(phi_rad)**2)
    b = - K * np.sin(2 * phi_rad) / (1 - K * np.cos(phi_rad)**2)
    A = (np.sqrt(a) * (c_2 - c_1)/b + (c_1 + c_2)/2)/4
    B = (np.sqrt(a) * (c_1 - c_2)/b + (c_1 + c_2)/2)/4
    
    print("A min/max:", np.min(A), np.max(A))
    print("B min/max:", np.min(B), np.max(B))
    print("a min/max:", np.min(a), np.max(a))
    print("b min/max:", np.min(b), np.max(b))
    print("c_1 min/max:", np.min(c_1), np.max(c_1))
    print("c_2 min/max:", np.min(c_2), np.max(c_2))

    u = np.sqrt(A) + np.sqrt(B) - U_mean
    v = (np.sqrt(A) - np.sqrt(B))/np.sqrt(a)
    w = np.zeros_like(u)
    return u, v, w

if __name__ == "__main__":
    # k = 0.00001     # k denotes the tangential sensitivity coefficient
    k = 0.1
    phi = 45       # Angle of the probe in degrees to the mean flow direction
    U_mean = 3.5  # Mean flow speed in m/s
    
    times, sondes = lire_fichier_U(os.path.join(DATA_DIRECTORY, 'E_125', 'U'))
    
    speeds = sondes[1][1]  # On prend les vitesses de la première sonde
    
    n = 100  # Nombre de points à afficher
    times = np.array(times[:n])
    speeds = np.array(speeds[:n])
    
    U_eff_1, U_eff_2 = calcul_U_effective(speeds, k, phi, U_mean)
    # u, v, w = calculate_speed_vector_using_U_eff(U_eff_1, U_eff_2, k, phi, U_mean, w = speeds[:, 2])  # On utilise la composante w de speeds pour w
    u, v, w = calculate_speed_vector_using_U_eff(U_eff_1, U_eff_2, k, phi, U_mean)

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
    plt.scatter(times, v, label='v probes (m/s)', color='g', alpha=0.5, s=10)
    plt.scatter(times, speeds[:, 1], label='v original (m/s)', color='orange', alpha=0.5, s=10)
    plt.legend()
    plt.title('Composante v')
    plt.xlabel('Temps (s)')
    plt.ylabel('Vitesse (m/s)')
    plt.grid()
    plt.subplot(1, 3, 3)
    plt.scatter(u, v, label='u,v probes', color='b', alpha=0.5, s=10)
    plt.scatter(speeds[:, 0], speeds[:, 1], label='u,v original', color='orange', alpha=0.5, s=10)
    plt.legend()
    plt.title('Composante u,v')
    plt.xlabel('u (m/s)')
    plt.ylabel('v (m/s)')
    plt.grid()
    plt.tight_layout()
    plt.show()