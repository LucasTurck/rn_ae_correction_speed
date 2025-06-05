from dir import RDN_DIRECTORY, MOD_PERSO_DIRECTORY
from pred_f.reseau_neurones import ModeleReseauNeurones
import ui.affichage
import tkinter as tk
from tkinter import ttk
import os
import json
import gc
import threading

class UICompileModel(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        dir_parameters = os.path.join(RDN_DIRECTORY, "last_config.json")
        if not os.path.exists(dir_parameters):
            dir_parameters = os.path.join(RDN_DIRECTORY, "default_config.json")
        if not os.path.exists(dir_parameters):
            raise FileNotFoundError(f"Le fichier de configuration {dir_parameters} n'existe pas.")
        with open(dir_parameters, 'r') as file:
            self.parameters = json.load(file)

        self.model_RdN = ModeleReseauNeurones(parameters=self.parameters)
        
        self.result_window = tk.Toplevel(self)
        self.result_window.title("Résultats de la compilation")
        self.result_window.geometry("400x300")

        ttk.Button(self.result_window, text="Fermer", command=self.UI_destroy).pack(pady=10)

        threading.Thread(target=self.entrainement, daemon=True).start()

    def UI_destroy(self):
        self.model_RdN.clear()
        del self.model_RdN
        gc.collect()  # Collecte les objets inutilisés
        self.destroy()
        
    def entrainement(self):
        self.model_RdN.remplissage_donnees()
        self.model_RdN.create_reseau()
        self.model_RdN.entrainer()
        print(f"R² pour l'entraînement : {self.model_RdN.evaluer(train=True)}")
        self.model_RdN.sauvegarder_apprentissage(MOD_PERSO_DIRECTORY)
        # Optionnel : afficher un message à la fin
        # tk.messagebox.showinfo("Info", "Entraînement terminé !")
        print(f"adresse de model_RdN : {self.model_RdN}")
        # Bouton pour afficher l'historique d'entraînement
        ttk.Button(self.result_window, text="Afficher l'historique d'entraînement", command=lambda: ui.afficher_historique(self.model_RdN, self.result_window)).pack()

        # Bouton pour afficher les résultats de l'entraînement
        ttk.Button(self.result_window, text="Afficher les résultats de l'entraînement", command=lambda: ui.afficher_resultats_train(self.model_RdN, self.result_window)).pack(pady=10)

        # Bouton pour aficher les résultats du réseau en changeant les données de test
        ttk.Button(self.result_window, text="Afficher les résultats du test", command=lambda: ui.afficher_resultats_test(self.model_RdN, self.result_window)).pack(pady=10)
