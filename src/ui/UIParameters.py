from dir import RDN_DIRECTORY
from pred_f.reseau_neurones import NAMES_LOSSES
import tkinter as tk
from tkinter import ttk
import os
import json
from ui.UITest import UITest

class UIParameters(ttk.Frame):
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
        self.archi_var = tk.StringVar(value=self.parameters.get('architecture', 'dense_simple'))
        self.E_var = tk.StringVar(value=self.parameters.get('E', 125))
        self.sonde_train_var = tk.StringVar(value=self.parameters.get('num_sonde_train', 0))
        self.nb_train_var = tk.StringVar(value=self.parameters.get('nb_training', 1000))
        self.sonde_test_var = tk.StringVar(value=self.parameters.get('num_sonde_test', 0))
        self.nb_test_var = tk.StringVar(value=self.parameters.get('nb_test', 1000))
        self.recursive_u_var = tk.StringVar(value=self.parameters.get('prediction', [0, 0, 0])[0])
        self.recursive_v_var = tk.StringVar(value=self.parameters.get('prediction', [0, 0, 0])[1])
        self.recursive_w_var = tk.StringVar(value=self.parameters.get('prediction', [0, 0, 0])[2])
        self.epochs_var = tk.StringVar(value=self.parameters.get('epochs', 10))
        self.batch_size_var = tk.StringVar(value=self.parameters.get('batch_size', 32))
        self.y_var = tk.StringVar(value=self.parameters.get('y', 1))
        self.erreur_var = tk.StringVar(value=self.parameters.get('loss', 'mae'))  # Variable pour l'erreur à afficher dans l'historique
        self.create_window()
        
    def create_window(self):
        parameters_window = tk.Toplevel(self)
        parameters_window.title("Choix des paramètres pour compiler un nouveau réseau de neurones")
        parameters_window.geometry("300x500")

        # Création des widgets pour les paramètres
        self.create_widgets(parameters_window)

    def create_widgets(self, parent):
        # Ajouter des widgets pour les paramètres

        # Bouton pour fermer la fenêtre
        ttk.Button(parent, text="Fermer", command=parent.destroy).pack(pady=10)

        ## Architecture :
        dir_architectures = os.path.join(RDN_DIRECTORY, "architectures.json")
        if not os.path.exists(dir_architectures):
            raise FileNotFoundError(f"Le fichier de configuration {dir_architectures} n'existe pas.")
        with open(dir_architectures, 'r') as file:
            architectures = json.load(file)
        names_archi = [arch['name'] for arch in architectures]
        ttk.Label(parent, text="Architecture :").pack()
        ttk.Combobox(parent, textvariable=self.archi_var, values=names_archi, state="readonly").pack()

        ## Erreur :
        ttk.Label(parent, text="Erreur :").pack()
        names_loss = NAMES_LOSSES
        ttk.Combobox(parent, values=names_loss, textvariable=self.erreur_var, state="readonly").pack()

        ## data :
        ### E :
        ttk.Label(parent, text="E :").pack()
        ttk.Entry(parent, textvariable=self.E_var).pack()

        ### sonde entrainement :
        ttk.Label(parent, text="Sonde d'entraînement :").pack()
        ttk.Entry(parent, textvariable=self.sonde_train_var).pack()

        ### nombres de données d'entraînement :
        ttk.Label(parent, text="Nombre de données d'entraînement :").pack()
        ttk.Entry(parent, textvariable=self.nb_train_var).pack()

        ## Prédiction :
        ttk.Label(parent, text="Prédiction :").pack()
        ttk.Entry(parent, textvariable=self.recursive_u_var).pack()
        ttk.Entry(parent, textvariable=self.recursive_v_var).pack()
        ttk.Entry(parent, textvariable=self.recursive_w_var).pack()

        ## y :
        ttk.Label(parent, text="y :").pack()
        ttk.Entry(parent, textvariable=self.y_var).pack()

        ## paramètres d'apprentissage : 
        ### epochs :
        ttk.Label(parent, text="Nombre d'epochs :").pack()
        ttk.Entry(parent, textvariable=self.epochs_var).pack()

        ### batch_size :
        ttk.Label(parent, text="Taille du batch :").pack()
        ttk.Entry(parent, textvariable=self.batch_size_var).pack()

        # Bouton pour compiler le reseau
        ttk.Button(parent, text="Compiler le réseau", command=lambda: self.compile_model()).pack()

    def save_parameters(self):
        # Enregistrer les paramètres dans le fichier JSON
        self.parameters['architecture'] = self.archi_var.get()
        self.parameters['E'] = int(self.E_var.get())
        self.parameters['num_sonde_train'] = int(self.sonde_train_var.get())
        self.parameters['nb_training'] = int(self.nb_train_var.get())
        self.parameters['num_sonde_test'] = int(self.sonde_test_var.get())
        self.parameters['nb_test'] = int(self.nb_test_var.get())
        self.parameters['prediction'] = [
            int(self.recursive_u_var.get()),
            int(self.recursive_v_var.get()),
            int(self.recursive_w_var.get())
        ]
        self.parameters['y'] = int(self.y_var.get())
        self.parameters['epochs'] = int(self.epochs_var.get())
        self.parameters['batch_size'] = int(self.batch_size_var.get())
        self.parameters['loss'] = self.erreur_var.get()  # Enregistrer l'erreur sélectionnée pour l'historique

        # Enregistrer dans le fichier de configuration
        dir_parameters = os.path.join(RDN_DIRECTORY, "last_config.json")
        try:
            with open(dir_parameters, 'w') as file:
                json.dump(self.parameters, file, indent=4)
        except Exception as e:
            tk.messagebox.showerror("Erreur", f"Impossible d'enregistrer les paramètres : {e}")
            
    def compile_model(self):
        self.save_parameters()
        test = UITest(self, self.controller, parameters=self.parameters)
        test.compile_model()