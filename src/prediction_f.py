import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from reed_data import lire_fichier_U
from dir import DATA_DIRECTORY
import os
import argparse

def pred_regression_lineaire(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def pred_regression_polynomiale(X, y, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model

def pred_foret_aleatoire(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

def pred_reseau_neurones(X, y):
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(10, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)
    return model

def pred_methode_naive(X):
    # Méthode naïve : Y = X
    return X.flatten()

def evaluer_modele(methode, model, X, y):
    if methode == 'naive':
        y_pred = pred_methode_naive(X)
    if methode == 'lineaire' or methode == 'polynomiale':
        y_pred = model.predict(X)
    elif methode == 'foret_aleatoire':
        y_pred = model.predict(X)
    elif methode == 'reseau_neurones':
        y_pred = model.predict(X).flatten()
    elif methode == 'scatter':
        # Pour le scatter plot, on ne calcule pas de prédictions
        return None
    else:
        raise ValueError("Modèle inconnu")

    return r2_score(y, y_pred)

def main(methode):
    # Paramètres
    e = 145
    sonde_num = 0  # Change le numéro de la sonde ici

    # Lecture des données
    filepath = os.path.join(DATA_DIRECTORY, f'E_{e}', 'U')
    times, sondes = lire_fichier_U(filepath)

    # Extraction des données pour la sonde choisie
    pos, col = sondes[sonde_num]
    X = np.array([v[0] for v in col]).reshape(-1, 1)  # première composante
    y = np.array([v[1] for v in col])                 # deuxième composante


    if methode == 'lineaire':
        model = pred_regression_lineaire(X, y)
    elif methode == 'polynomiale':
        degree = 2 # Change le degré du polynôme ici
        model = pred_regression_polynomiale(X, y, degree)
    elif methode == 'foret_aleatoire':
        model = pred_foret_aleatoire(X, y)
    elif methode == 'reseau_neurones':
        model = pred_reseau_neurones(X, y)
    elif methode == 'naive':
        model = pred_methode_naive(X)
    elif methode == 'scatter':
        scatter_plot(X, y, sonde_num)
        return
    else:
        raise ValueError("Méthode inconnue. Choisissez parmi 'lineaire', 'polynomiale', 'foret_aleatoire', 'reseau_neurones', 'naive'.")
    
    # Évaluation du modèle
    r2 = evaluer_modele(methode, model, X, y)
    print(f"R² pour la méthode {methode} :", r2)
    # Affichage des prédictions
    plot_predictions(X, y, model, sonde_num)
    
def plot_predictions(X, y, model, sonde_num):
    if isinstance(model, LinearRegression):
        y_pred = model.predict(X)
    elif isinstance(model, PolynomialFeatures):
        X_poly = model.transform(X)
        y_pred = model.predict(X_poly)
    elif isinstance(model, RandomForestRegressor):
        y_pred = model.predict(X)
    elif isinstance(model, keras.Sequential):
        y_pred = model.predict(X).flatten()
    else:
        raise ValueError("Modèle inconnu")

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Données réelles', alpha=0.5)
    plt.scatter(X, y_pred, color='red', label='Prédictions', alpha=0.5)
    plt.xlabel("Première composante (X)")
    plt.ylabel("Deuxième composante (Y)")
    plt.title(f"Sonde {sonde_num} : Prédictions vs Réel")
    plt.legend()
    plt.grid(True)
    plt.show()
        
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

if __name__ == "__main__":
    # Paramètres
    parser = argparse.ArgumentParser()
    parser.add_argument('--methode', type=str, default='null', choices=['lineaire', 'polynomiale', 'foret_aleatoire', 'reseau_neurones', 'naive', 'scatter'], help='Méthode de prédiction à utiliser')
    args = parser.parse_args()
    if args.methode != 'null':
        main(args.methode)
