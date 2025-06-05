from dir import MOD_DIRECTORY
from pred_f.reseau_neurones import ModeleReseauNeurones
import tkinter as tk
from tkinter import ttk
import os
import json
from ui.UITest import UITest


class UIModelSelector(ttk.Frame):
    """
    Classe pour le sélecteur de modèle.
    """
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.model_RdN = ModeleReseauNeurones()
        
        self.architecture_var = tk.StringVar()
        self.run_var = tk.StringVar()
        self.sonde_test_var = tk.IntVar()
        self.nb_test_var = tk.IntVar()
        self.run_display_to_dir = {}
        self.run_display_list = []
        
        self.create_window()
        
    def create_window(self):
        self.selection_window = tk.Toplevel(self)
        self.selection_window.title("Sélection du modèle")
        self.selection_window.geometry("300x200")
        
        # Menu déroulant : architecture
        ttk.Label(self.selection_window, text="Architecture :").pack()
        self.archi_combo = ttk.Combobox(self.selection_window, textvariable=self.architecture_var, state="readonly")
        self.archi_combo.pack()
        self.archi_combo.bind("<<ComboboxSelected>>", self.update_runs)
        
        # Menu déroulant : runs
        ttk.Label(self.selection_window, text="Run entrainé :").pack()
        self.run_combo = ttk.Combobox(self.selection_window, textvariable=self.run_var, state="readonly")
        self.run_combo.pack()
        self.run_combo.config(width=40)  # Modifier la largeur du Combobox (en nombre de caractères)
        
        self.populate_architectures()
        
        # Bouton pour charger le modèle sélectionné
        ttk.Button(self.selection_window, text="Selectionner le modèle", command=self.load_model).pack(pady=10)


    def populate_architectures(self):
        architectures = [d for d in os.listdir(MOD_DIRECTORY) if os.path.isdir(os.path.join(MOD_DIRECTORY, d))]
        self.archi_combo['values'] = architectures
        if architectures:
            self.architecture_var.set(architectures[0])
            self.update_runs()
        else:
            self.architecture_var.set("")
            self.run_combo['values'] = []
            self.run_var.set("")

    def update_runs(self, event=None):
        architectures = self.architecture_var.get()
        runs_dir = os.path.join(MOD_DIRECTORY, architectures)
        self.run_display_list = []
        self.run_display_to_dir = {}
        if os.path.exists(runs_dir):
            runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            for run in runs:
                config_path = os.path.join(runs_dir, run, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as file:
                            config = json.load(file)
                        display = f"recursive= {config.get('prediction','?')}, epochs={config.get('epochs','?')}, batch={config.get('batch_size','?')}"
                    except Exception as e:
                        display = f"{run} | Erreur de chargement config: {str(e)}"
                else:
                    display = f"{run} | config.json manquant"
                self.run_display_list.append(display)
                self.run_display_to_dir[display] = run
            self.run_combo['values'] = self.run_display_list
            if self.run_display_list:
                self.run_var.set(self.run_display_list[0])
            else:
                self.run_var.set("")
        else:
            self.run_combo['values'] = []
            self.run_var.set("")    
    
    def load_model(self):
        
        architecture = self.architecture_var.get()
        run = self.run_display_to_dir.get(self.run_var.get(), None)
        # if not architecture or not run:
        #     tk.messagebox.showerror("Erreur", "Veuillez sélectionner une architecture et un run.")
        #     return
        # config_path = os.path.join(MOD_DIRECTORY, architecture, run, "config.json")
        # if not os.path.exists(config_path):
        #     tk.messagebox.showerror("Erreur", f"Le fichier de configuration {config_path} n'existe pas.")
        #     return
        # with open(config_path, 'r') as file:
        #     parameters = json.load(file)
        # self.model_RdN.load_parameters(parameters)
        path = os.path.join(MOD_DIRECTORY, architecture, run)
        if not os.path.exists(path):
            tk.messagebox.showerror("Erreur", f"Le modèle {path} n'existe pas.")
            return
        self.model_RdN.charger_apprentissage(path)

        # Créer une instance de UITest avec les paramètres chargés
        test = UITest(self.selection_window, self.controller, model=self.model_RdN)
        test.afficher_resultats()