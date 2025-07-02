from dir import RDN_DIRECTORY, MOD_PERSO_DIRECTORY, MOD_DIRECTORY
from pred_f.cnn import CNN

import os
from tkinter import ttk
import tkinter as tk
import json
import matplotlib.pyplot as plt
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class UICnn(ttk.Frame):
    def __init__(self, parent, controller):
        """Initialise l'interface utilisateur pour les paramètres du CNN."""
        super().__init__(parent)
        self.controller = controller
        
        self.parameters_window = tk.Toplevel(self)
        self.parameters_window.title("CNN")
        self.parameters_window.minsize(width=400, height=1)
        
        # Bouton pour ouvrir la fenêtre des paramètres du CNN
        ttk.Button(self.parameters_window, text="Paramètres du CNN", command=lambda: self.open_parameters_window()).pack(pady=10)
        
        # Bouton pour charger un modèle existant
        ttk.Button(self.parameters_window, text="Charger un modèle existant", command=lambda: self.open_load_model_window()).pack(pady=10)

        # Bouton pour fermer la fenêtre
        ttk.Button(self.parameters_window, text="Fermer", command=self.parameters_window.destroy).pack(pady=10)

    def open_parameters_window(self):
        """Ouvre la fenêtre des paramètres du CNN."""
        self.parameters_window = UIParametersCnn(self, self.controller)
        self.parameters_window.pack(fill=tk.BOTH, expand=True)

    def open_load_model_window(self):
        """Ouvre la fenêtre de chargement d'un modèle existant."""
        
        self.load_model_window = UILoadModel(self, self.controller)
        self.load_model_window.pack(fill=tk.BOTH, expand=True)

class  UIParametersCnn(ttk.Frame):
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
        
        self.archi_var = tk.StringVar(value=self.parameters.get('architecture', 'cnn_lstm'))
        self.E_var = tk.StringVar(value=self.parameters.get('E', 125))
        self.sonde_train_var = tk.StringVar(value=self.parameters.get('num_sonde_train', 0))
        self.nb_train_var = tk.StringVar(value=self.parameters.get('nb_training', -1))
        
        self.sonde_test_var = tk.StringVar(value=self.parameters.get('num_sonde_test', 1))
        self.nb_test_var = tk.StringVar(value=self.parameters.get('nb_test', 100))
        
        self.puissance_var = tk.StringVar(value=self.parameters.get('puisance', [1, 2, 3]))
        self.timesteps_before_var = tk.StringVar(value=self.parameters.get('timesteps_before', 10))
        self.timesteps_after_var = tk.StringVar(value=self.parameters.get('timesteps_after', 10))
        self.epochs_var = tk.StringVar(value=self.parameters.get('epochs', 10))
        self.batch_size_var = tk.StringVar(value=self.parameters.get('batch_size', 32))
        self.y_col_var = tk.StringVar(value=self.parameters.get('y_col', 1))
        self.erreur_var = tk.StringVar(value=self.parameters.get('loss', 'mae'))
        self.prediction_u_var = tk.StringVar(value=self.parameters.get('prediction', [1, 0, 0])[0])
        self.prediction_v_var = tk.StringVar(value=self.parameters.get('prediction', [1, 0, 0])[1])
        self.prediction_w_var = tk.StringVar(value=self.parameters.get('prediction', [1, 0, 0])[2])
        
        self.create_window()
        
    def create_window(self):
        parameters_window = tk.Toplevel(self)
        parameters_window.title("Choix des paramètres pour compiler un nouveau réseau de neurones")
        parameters_window.minsize(width=300, height=1)
        
        # Création des widgets pour les paramètres
        self.create_widgets(parameters_window)
        
    def create_widgets(self, parent):
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
        self.archi_combobox = ttk.Combobox(parent, textvariable=self.archi_var, values=names_archi)
        self.archi_combobox.pack()

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

        ## Timesteps :
        ttk.Label(parent, text="Timesteps avant :").pack()
        ttk.Entry(parent, textvariable=self.timesteps_before_var).pack()
        ttk.Label(parent, text="Timesteps après :").pack()
        ttk.Entry(parent, textvariable=self.timesteps_after_var).pack()
        
        ## y_col :
        ttk.Label(parent, text='y_col : ').pack()
        ttk.Entry(parent, textvariable=self.y_col_var).pack()
        
        ## Prediction :
        ttk.Label(parent, text="Prédiction :").pack()
        frame_pred = ttk.Frame(parent)
        frame_pred.pack(pady=5)  # Centré par défaut

        ttk.Entry(frame_pred, textvariable=self.prediction_u_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Entry(frame_pred, textvariable=self.prediction_v_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Entry(frame_pred, textvariable=self.prediction_w_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ## puissances :
        ttk.Label(parent, text="Puissances (séparées par des virgules) :").pack()
        ttk.Entry(parent, textvariable=self.puissance_var).pack()
        
        ## epochs :
        ttk.Label(parent, text="Epochs :").pack()
        ttk.Entry(parent, textvariable=self.epochs_var).pack()
        
        # ## batch_size :
        ttk.Label(parent, text="Taille du batch :").pack()
        ttk.Entry(parent, textvariable=self.batch_size_var).pack()

        # Bouton pour compiler le reseau
        ttk.Button(parent, text="Compiler le réseau", command=lambda: self.compile_model()).pack(side=tk.BOTTOM, pady=10)

    def save_parameters(self):
        """Sauvegarde les paramètres dans le fichier de configuration."""
        self.parameters['architecture'] = self.archi_var.get()
        self.parameters['E'] = int(self.E_var.get())
        self.parameters['num_sonde_train'] = int(self.sonde_train_var.get())
        self.parameters['nb_training'] = int(self.nb_train_var.get())
        self.parameters['num_sonde_test'] = int(self.sonde_test_var.get())
        self.parameters['nb_test'] = int(self.nb_test_var.get())
        self.parameters['timesteps_before'] = int(self.timesteps_before_var.get())
        self.parameters['timesteps_after'] = int(self.timesteps_after_var.get())
        self.parameters['epochs'] = int(self.epochs_var.get())
        self.parameters['batch_size'] = int(self.batch_size_var.get())
        self.parameters['y_col'] = int(self.y_col_var.get())
        self.parameters['loss'] = self.erreur_var.get()
        self.parameters['prediction'] = [
            int(self.prediction_u_var.get()),
            int(self.prediction_v_var.get()),
            int(self.prediction_w_var.get())
        ]
        self.parameters['puisance'] = [float(p) for p in self.puissance_var.get().split(',')]
        sum = self.parameters['prediction'][0] + self.parameters['prediction'][1] + self.parameters['prediction'][2]
        self.parameters['input_shape_before'] = [self.parameters['timesteps_before'], sum*len(self.parameters['puisance'])]
        self.parameters['input_shape_after'] = [self.parameters['timesteps_after'], sum*len(self.parameters['puisance'])]

        try:
            with open(os.path.join(RDN_DIRECTORY, "last_config.json"), 'w') as file:
                json.dump(self.parameters, file, indent=4)
            print("Paramètres sauvegardés avec succès.")
        except Exception as e:
            tk.messagebox.showerror("Erreur", f"Impossible de sauvegarder les paramètres : {e}")
            
    def compile_model(self):
        """Compile le modèle avec les paramètres spécifiés."""
        self.save_parameters()
        uitest = UItestCnn(self, self.controller)
        uitest.compile_model(parameters=self.parameters)
        
class UILoadModel(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.path_mod_var = tk.StringVar()
        self.architecture_var = tk.StringVar()
        self.run_var = tk.StringVar()
        self.run_display_list = []
        self.run_display_to_dir = {}
        
        
        self.create_window()
        
    def create_window(self):
        self.load_model_window = tk.Toplevel(self)
        self.load_model_window.title("Charger un modèle existant")
        self.load_model_window.minsize(width=300, height=1)

        # Bouton pour fermer la fenêtre
        ttk.Button(self.load_model_window, text="Fermer", command=self.load_model_window.destroy).pack(pady=10)

        # menu déroulant pour sélectionner dossier ou sont stockées les architectures
        ttk.Label(self.load_model_window, text="Sélectionner le dossier des modèles :").pack()
        self.path_combo = ttk.Combobox(self.load_model_window, textvariable=self.path_mod_var)
        self.path_combo.pack()
        self.path_combo['values'] = [MOD_PERSO_DIRECTORY, MOD_DIRECTORY]
        self.path_combo.current(0)  # Sélectionne le premier dossier par défaut
        self.path_combo.bind("<<ComboboxSelected>>", self.update_architectures)
        self.path_combo.config(width=70)  # Modifier la largeur du Combobox (en nombre de caractères)
        
        # menu déroulant pour sélectionner l'architecture
        ttk.Label(self.load_model_window, text="Sélectionner l'architecture :").pack()
        self.archi_combo = ttk.Combobox(self.load_model_window, textvariable=self.architecture_var, state="readonly")
        self.archi_combo.pack()
        self.archi_combo.bind("<<ComboboxSelected>>", self.update_runs)
        self.archi_combo.config(width=40)  # Modifier la largeur du Combobox (en nombre de caractères)

        # menu déroulant pour sélectionner le run
        ttk.Label(self.load_model_window, text="Run entrainé :").pack()
        self.run_combo = ttk.Combobox(self.load_model_window, textvariable=self.run_var, state="readonly")
        self.run_combo.pack()
        self.run_combo.config(width=40)  # Modifier la largeur du Combobox (en nombre de caractères)
        self.populate_architectures()
        
        # Bouton pour charger le modèle sélectionné
        ttk.Button(self.load_model_window, text="Charger le modèle", command=self.load_model).pack(pady=10)
        
    def populate_architectures(self):
        architectures = [d for d in os.listdir(self.path_mod_var.get()) if os.path.isdir(os.path.join(self.path_mod_var.get(), d))]
        
        self.archi_combo['values'] = architectures
        if architectures:
            self.architecture_var.set(architectures[-1])
            self.run_combo.set('')
            self.update_runs()
        else:
            self.architecture_var.set("")
            self.run_combo['values'] = []
            self.run_var.set("")
            self.run_display_list = []
            self.run_display_to_dir = {}
            
    def update_architectures(self, event=None):
        """Met à jour la liste des architectures disponibles."""
        self.populate_architectures()
        
    def update_runs(self, event=None):
        """Met à jour la liste des runs disponibles."""
        architectures = self.architecture_var.get()
        runs_dir = os.path.join(self.path_mod_var.get(), architectures)
        self.run_display_list = []
        self.run_display_to_dir = {}
        if os.path.exists(runs_dir):
            runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            for run in runs:
                config_path = os.path.join(runs_dir, run, "assets", "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as file:
                            config = json.load(file)
                            display_name = f"{run} - {config.get('architecture', 'unknown')}"
                            self.run_display_list.append(display_name)
                            self.run_display_to_dir[display_name] = os.path.join(runs_dir, run)
                    except json.JSONDecodeError:
                        print(f"Erreur de décodage JSON pour le fichier {config_path}")
            self.run_combo['values'] = self.run_display_list
            if self.run_display_list:
                self.run_var.set(self.run_display_list[0])
            else:
                self.run_var.set("")
        else:
            self.run_combo['values'] = []
            self.run_var.set("")
            
    def load_model(self):
        """Charge le modèle sélectionné."""
        architecture = self.architecture_var.get()
        run = self.run_display_to_dir.get(self.run_var.get(), None)

        if not architecture or not run:
            tk.messagebox.showerror("Erreur", "Veuillez sélectionner une architecture et un run.")
            return
        
        path = os.path.join(self.path_mod_var.get(), architecture, run)
        if not os.path.exists(path):
            tk.messagebox.showerror("Erreur", f"Le chemin {path} n'existe pas.")
            return
        
        uitest = UItestCnn(self, self.controller)
        uitest.load_model(path=path)
        
        
class UItestCnn(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        self.result_window = tk.Toplevel(self)
        self.result_window.minsize(width=200, height=1)

        ttk.Button(self.result_window, text="Fermer", command=self.result_window.destroy).pack(pady=10)

    def compile_model(self, parameters=None):
        if parameters is None:
            tk.messagebox.showerror("Erreur", "Aucun modèle sélectionné.")
            return
        
        self.model_RdN = CNN(parameters=parameters)
        self.model_RdN.init_parameters()
        self.result_window.title(f"Résultats - {self.model_RdN.parameters['architecture']}")
        self.thread = threading.Thread(target=self.entrainement, daemon=True)
        self.thread.start()

    def entrainement(self):
        self.model_RdN.load_data()
        self.model_RdN.create_reseau()
        self.model_RdN.create_data(train=True)
        self.model_RdN.add_power(train=True)
        self.model_RdN.create_data(train=False)
        self.model_RdN.add_power(train=False)
        self.model_RdN.train()
        self.model_RdN.predict(train=True)
        print(f"R² pour l'entraînement : {self.model_RdN.r2_score(train=True)}")
        self.model_RdN.predict(train=False)
        print(f"R² pour le test : {self.model_RdN.r2_score(train=False)}")
        self.result_window.after(0, self.afficher_resultats)
        
        # bouton pour sauvegarder le modèle
        ttk.Button(self.result_window, text="Sauvegarder le modèle", command=lambda: self.model_RdN.save_model(MOD_PERSO_DIRECTORY)).pack(pady=10)
    
    def load_model(self, path):
        self.model_RdN = CNN()
        self.model_RdN.load_model(path)
        self.result_window.title(f"Résultats - {self.model_RdN.parameters['architecture']}")
        self.model_RdN.load_data()
        self.model_RdN.create_data(train=True)
        self.model_RdN.add_power(train=True)
        self.model_RdN.create_data(train=False)
        self.model_RdN.add_power(train=False)
        self.model_RdN.predict(train=True)
        print(f"R² pour l'entraînement : {self.model_RdN.r2_score(train=True)}")
        self.model_RdN.predict(train=False)
        print(f"R² pour le test : {self.model_RdN.r2_score(train=False)}")
        self.afficher_resultats()

    def afficher_resultats(self):

        fig, self.ax = plt.subplots(1, 3, figsize=(12, 6))
        self.model_RdN.affichage_history('mae', axis=self.ax[0])
        self.model_RdN.affichage_prediction(train=True, axis=self.ax[1])
        self.model_RdN.affichage_prediction(train=False, axis=self.ax[2])

        plt.tight_layout()
        plt.show()
