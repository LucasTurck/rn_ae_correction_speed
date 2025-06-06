from dir import MOD_PERSO_DIRECTORY
from pred_f.reseau_neurones import ModeleReseauNeurones
from ui.affichage import affichage_resultats_train, affichage_resultats_test, affichage_historique
import tkinter as tk
from tkinter import ttk
import gc
import threading
import copy

class UITest(ttk.Frame):
    def __init__(self, parent, controller, parameters=None, model=None):
        super().__init__(parent)
        self.controller = controller
        if model is not None:
            self.model_RdN = model.copy_model()
        else:
            new_parameters = copy.deepcopy(parameters) if parameters else {}
            self.model_RdN = ModeleReseauNeurones(parameters=new_parameters)
            self.model_RdN.init_parameters()
        
        self.result_window = tk.Toplevel(self)
        self.result_window.title(f"Résultats - {self.model_RdN.parameters['architecture']}")
        self.result_window.minsize(width=200, height=1)

        ttk.Button(self.result_window, text="Fermer", command=self.UI_destroy).pack(pady=10)
        

    def compile_model(self):
        self.thread = threading.Thread(target=self.entrainement, daemon=True)
        self.thread.start()

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
        self.result_window.after(0, self.afficher_resultats)

    def afficher_resultats(self):

        # Bouton pour afficher l'historique d'entraînement
        ttk.Button(self.result_window, text="Afficher l'historique d'entraînement", command=lambda: affichage_historique(self.model_RdN, self.result_window)).pack(pady=10)
        
        # Bouton pour afficher les résultats de l'entraînement
        ttk.Button(self.result_window, text="Afficher les résultats de l'entraînement", command=lambda: affichage_resultats_train(self.model_RdN, self.result_window)).pack(pady=10)

        # Bouton pour aficher les résultats du réseau en changeant les données de test
        ttk.Button(self.result_window, text="Afficher les résultats du test", command=lambda: affichage_resultats_test(self.model_RdN, self.result_window)).pack(pady=10)
