from dir import RDN_DIRECTORY, MOD_DIRECTORY, MOD_PERSO_DIRECTORY
# import reseau_neurones module correctly (adjust the import as needed for your project structure)
from pred_f.reseau_neurones import *
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import json
import os
      

class UIparams(ttk.Frame):
    """Classe pour l'interface utilisateur des paramètres de l'architecture du réseau de neurones."""
    def __init__(self, root, controller):
        super().__init__(root)
        self.controller = controller
        dir_params = os.path.join(RDN_DIRECTORY, "last_config.json")
        with open(dir_params, 'r') as file:
            self.params = json.load(file)
        self.archi_var = tk.StringVar(value=self.params.get('architecture', 'dense_simple'))
        self.E_var = tk.StringVar(value=self.params.get('E', 125))
        self.sonde_train_var = tk.StringVar(value=self.params.get('num_sonde_train', 0))
        self.nb_train_var = tk.StringVar(value=self.params.get('nb_training', 1000))
        self.sonde_test_var = tk.StringVar(value=self.params.get('num_sonde_test', 0))
        self.nb_test_var = tk.StringVar(value=self.params.get('nb_test', 1000))
        self.recursive_u_var = tk.StringVar(value=self.params.get('prediction', [0, 0, 0])[0])
        self.recursive_v_var = tk.StringVar(value=self.params.get('prediction', [0, 0, 0])[1])
        self.recursive_w_var = tk.StringVar(value=self.params.get('prediction', [0, 0, 0])[2])
        self.epochs_var = tk.StringVar(value=self.params.get('epochs', 10))
        self.batch_size_var = tk.StringVar(value=self.params.get('batch_size', 32))
        self.y_var = tk.StringVar(value=self.params.get('y', 1))
        self.create_widgets()

    def create_widgets(self):
        # Ajouter des widgets pour les paramètres

        ## Architecture :
        dir_architectures = os.path.join(RDN_DIRECTORY, "architectures.json")
        with open(dir_architectures, 'r') as file:
            architectures = json.load(file)
        names_archi = [arch['name'] for arch in architectures]
        ttk.Label(self, text="Architecture :").pack()
        ttk.Combobox(self, textvariable=self.archi_var, values=names_archi, state="readonly").pack()

        ## data :
        ### E :
        ttk.Label(self, text="E :").pack()
        ttk.Entry(self, textvariable=self.E_var).pack()

        ### sonde entrainement :
        ttk.Label(self, text="Sonde d'entraînement :").pack()
        ttk.Entry(self, textvariable=self.sonde_train_var).pack()

        ### nombres de données d'entraînement :
        ttk.Label(self, text="Nombre de données d'entraînement :").pack()
        ttk.Entry(self, textvariable=self.nb_train_var).pack()
        
        ### sonde test :
        ttk.Label(self, text="Sonde de test :").pack()
        ttk.Entry(self, textvariable=self.sonde_test_var).pack()

        ### nombres de données de test :
        ttk.Label(self, text="Nombre de données de test :").pack()
        ttk.Entry(self, textvariable=self.nb_test_var).pack()

        ## Prédiction :
        ttk.Label(self, text="Prédiction :").pack()
        ttk.Entry(self, textvariable=self.recursive_u_var).pack()
        ttk.Entry(self, textvariable=self.recursive_v_var).pack()
        ttk.Entry(self, textvariable=self.recursive_w_var).pack()
        
        ## y :
        ttk.Label(self, text="y :").pack()
        ttk.Entry(self, textvariable=self.y_var).pack()

        ## paramètres d'apprentissage : 
        ### epochs :
        ttk.Label(self, text="Nombre d'epochs :").pack()
        ttk.Entry(self, textvariable=self.epochs_var).pack()
        
        ### batch_size :
        ttk.Label(self, text="Taille du batch :").pack()
        ttk.Entry(self, textvariable=self.batch_size_var).pack()

        ## Bouton pour enregistrer les paramètres
        ttk.Button(self, text="Enregistrer les paramètres", command=self.save_params).pack()
        
        # Bouton pour compiler le reseau
        ttk.Button(self, text="Compiler le réseau", command=lambda: self.controller.compile_model()).pack()

        # Bouton pour aller à l'interface de sélection du modèle
        ttk.Button(self, text="Sélectionner un modèle", command=lambda: self.controller.show_frame("UIModelSelector")).pack()

    def save_params(self):
        # Enregistrer les paramètres dans le fichier JSON
        self.params['architecture'] = self.archi_var.get()
        self.params['E'] = int(self.E_var.get())
        self.params['num_sonde_train'] = int(self.sonde_train_var.get())
        self.params['nb_training'] = int(self.nb_train_var.get())
        self.params['num_sonde_test'] = int(self.sonde_test_var.get())
        self.params['nb_test'] = int(self.nb_test_var.get())
        self.params['prediction'] = [
            int(self.recursive_u_var.get()),
            int(self.recursive_v_var.get()),
            int(self.recursive_w_var.get())
        ]
        self.params['y'] = int(self.y_var.get())
        self.params['epochs'] = int(self.epochs_var.get())
        self.params['batch_size'] = int(self.batch_size_var.get())

        # Enregistrer dans le fichier de configuration
        dir_params = os.path.join(RDN_DIRECTORY, "last_config.json")
        with open(dir_params, 'w') as file:
            json.dump(self.params, file, indent=4)
        
        # print("Paramètres enregistrés :", self.params)
        
        # Mettre à jour le modèle avec les nouveaux paramètres
        
        self.controller.set_params(params=self.params)
    
class UINetwork(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        ttk.Label(self, text="Ici tu peux créer, entraîner et afficher le réseau").pack(pady=10)
        ttk.Button(self, text="Retour aux paramètres", command=lambda: controller.show_frame("UIparams")).pack(pady=10)

class UIModelSelector(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.archi_var = tk.StringVar()
        self.run_var = tk.StringVar()

        # Premier menu déroulant : architectures
        ttk.Label(self, text="Architecture :").pack()
        self.archi_combo = ttk.Combobox(self, textvariable=self.archi_var, state="readonly")
        self.archi_combo.pack()
        self.archi_combo.bind("<<ComboboxSelected>>", self.update_runs)

        # Second menu déroulant : runs
        ttk.Label(self, text="Run entraîné :").pack()
        self.run_combo = ttk.Combobox(self, textvariable=self.run_var, state="readonly", width=80)
        self.run_combo.pack()

        self.populate_architectures()
        
        # Bonton de selection : 
        ttk.Button(self, text="Sélectionner le modèle", command=self.select_model).pack(pady=10)
        
        # Bouton pour revenir aux paramètres
        ttk.Button(self, text="Retour aux paramètres", command=lambda: controller.show_frame("UIparams")).pack(pady=10)    

    def populate_architectures(self):
        # Liste les dossiers dans MOD_DIRECTORY
        archis = [d for d in os.listdir(MOD_DIRECTORY) if os.path.isdir(os.path.join(MOD_DIRECTORY, d))]
        self.archi_combo['values'] = archis
        if archis:
            self.archi_var.set(archis[0])
            self.update_runs()

    def update_runs(self, event=None):
        archi = self.archi_var.get()
        runs_dir = os.path.join(MOD_DIRECTORY, archi)
        run_display_list = []
        self.run_display_to_dir = {}
        if os.path.isdir(runs_dir):
            runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            for run in runs:
                config_path = os.path.join(runs_dir, run, "config.json")
                if os.path.isfile(config_path):
                    try:
                        with open(config_path, "r") as f:
                            config = json.load(f)
                        # Construire un texte résumé (adapte selon tes besoins)
                        display = f"{run} | recursive= {config.get('prediction','?')}, epochs={config.get('epochs','?')}, batch={config.get('batch_size','?')}"
                    except Exception as e:
                        display = f"{run} | erreur lecture config"
                else:
                    display = f"{run} | pas de config"
                run_display_list.append(display)
                self.run_display_to_dir[display] = run
            self.run_combo['values'] = run_display_list
            if run_display_list:
                self.run_var.set(run_display_list[0])
            else:
                self.run_var.set('')
        else:
            self.run_combo['values'] = []
            self.run_var.set('')
        
    def select_model(self):
        # recupérer les paramètres du modèle sélectionné
        archi = self.archi_var.get()
        run = self.run_display_to_dir.get(self.run_var.get(), None)
        if run:
            config_path = os.path.join(MOD_DIRECTORY, archi, run, "config.json")
            if os.path.isfile(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                self.controller.set_params(config)
            else:
                print("Erreur : Fichier de configuration introuvable.")
        else:
            print("Erreur : Aucune exécution sélectionnée.")

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interface Réseau de Neurones")
        self.geometry("500x600")
        self.frames = {}
        
        self.model_Rdn = None

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        for F in (UIparams, UINetwork, UIModelSelector):
            frame = F(container, self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("UIparams")
        self.bind("<Escape>", lambda e: self.destroy())

    def show_frame(self, name):
        frame = self.frames[name]
        frame.tkraise()
        
    def set_params(self, params):
        self.model_Rdn.set_parameters(params)
    
    def compile_model(self, params):
        self.model_Rdn = ModeleReseauNeurones(params=params)
        self.model_Rdn.remplissage_donnees()
        self.model_Rdn.create_reseau()
        self.model_Rdn.entrainer()
        self.model_Rdn.R2entrainement()
        
        self.model_Rdn.remplissage_donnees(train=False)
        self.model_Rdn.predire()
        self.model_Rdn.evaluer()
        
        self.model_Rdn.sauvegarder_apprentissage(MOD_PERSO_DIRECTORY)
        
        
        
        
    
    

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
