from dir import MOD_PERSO_DIRECTORY
from AlgoCorrection import AlgoCorrection
import tkinter as tk
from tkinter import ttk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

class UIAlgoCorrection(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.algo_correction = AlgoCorrection()
        
        self.architecture_var = tk.StringVar()
        self.run_var = tk.StringVar()
        self.sonde_test_var = tk.IntVar()
        self.nb_test_var = tk.IntVar()
        self.run_display_to_dir = {}
        self.run_display_list = []
        
        self.algo_window = tk.Toplevel(self)
        self.algo_window.title("Résultats de l'algorithme de correction")
        self.algo_window.minsize(width=200, height=1)

        ttk.Button(self.algo_window, text="Fermer", command=self.UI_destroy).pack(pady=10)

        # Menu déroulant : architecture
        ttk.Label(self.algo_window, text="Architecture :").pack()
        self.archi_combo = ttk.Combobox(self.algo_window, textvariable=self.architecture_var, state="readonly")
        self.archi_combo.pack()
        self.archi_combo.bind("<<ComboboxSelected>>", self.update_runs)
        
        # Menu déroulant : runs
        ttk.Label(self.algo_window, text="Run entrainé :").pack()
        self.run_combo = ttk.Combobox(self.algo_window, textvariable=self.run_var, state="readonly")
        self.run_combo.pack()
        self.run_combo.config(width=40)  # Modifier la largeur du Combobox (en nombre de caractères)
        
        self.populate_architectures()
        
        # Bouton pour charger le modèle sélectionné
        ttk.Button(self.algo_window, text="Selectionner le modèle", command=self.load_model).pack(pady=10)

        self.result_window = tk.Toplevel(self.algo_window)
        self.result_window.title("Résultats de l'algorithme de correction")
        self.result_window.geometry("400x300")
        self.result_window.withdraw()  # Cacher la fenêtre de résultats au début
        
        
        # Frame pour les contrôles
        self.control_frame = tk.Frame(self.result_window)
        self.control_frame.pack(fill="x", padx=10, pady=5)

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(10, 4))

        # Canvas pour afficher la figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.result_window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Bouton pour fermer la fenêtre
        tk.Button(self.control_frame, text="Fermer", command=self.result_window.destroy).pack(side="right")


    def UI_destroy(self):
        self.algo_correction.clear()
        del self.algo_correction
        self.destroy()
        
    def populate_architectures(self):
        architectures = [d for d in os.listdir(MOD_PERSO_DIRECTORY) if os.path.isdir(os.path.join(MOD_PERSO_DIRECTORY, d))]
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
        runs_dir = os.path.join(MOD_PERSO_DIRECTORY, architectures)
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
        if not architecture or not run:
            tk.messagebox.showerror("Erreur", "Veuillez sélectionner une architecture et un run.")
            return
        path = os.path.join(MOD_PERSO_DIRECTORY, architecture, run)
        if not os.path.exists(path):
            tk.messagebox.showerror("Erreur", f"Le modèle {path} n'existe pas.")
            return

        self.algo_correction.charger_model(path)
        self.algo_correction.set_original_speed()
        self.algo_correction.set_simulated_speed()

        ttk.Button(self.control_frame, text="Corriger la vitesse", command=self.correct_and_display_results).pack(pady=10)
        self.result_window.deiconify()  # Afficher la fenêtre de résultats

    def correct_and_display_results(self):
        self.algo_correction.correct_speed()
        # Afficher les résultats dans la fenêtre de résultats
        self.update_results_display()

    def update_results_display(self):
        
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        self.ax1.plot(self.algo_correction.original_speed[:self.nb_test_var.get(), 0], label='u original (m/s)', color='orange', alpha=0.5, linewidth=1)
        for i in range(len(self.algo_correction.corrected_speed)):
            self.ax1.scatter(range(self.nb_test_var.get()), self.algo_correction.corrected_speed[i][:self.nb_test_var.get(), 0], label=f'u corrigé {i+1} (m/s)', alpha=0.5, s=i+5)
            
        self.ax1.set_title('Composante u')
        self.ax1.set_xlabel('Temps')
        self.ax1.set_ylabel('Vitesse (m/s)')
        self.ax1.grid()
        self.ax1.legend()
        
        self.ax2.plot(self.algo_correction.original_speed[:self.nb_test_var.get(), 1], label='v original (m/s)', color='orange', alpha=0.5, linewidth=1)
        for i in range(len(self.algo_correction.corrected_speed)):
            self.ax2.scatter(range(self.nb_test_var.get()), self.algo_correction.corrected_speed[i][:self.nb_test_var.get(), 1], label=f'v corrigé {i+1} (m/s)', alpha=0.5, s=i+5)
            
        self.ax2.set_title('Composante v')
        self.ax2.set_xlabel('Temps')
        self.ax2.set_ylabel('Vitesse (m/s)')
        self.ax2.grid()
        self.ax2.legend()
        
        self.ax3.plot(self.algo_correction.original_speed[:self.nb_test_var.get(), 2], label='w original (m/s)', color='orange', alpha=0.5, linewidth=1)
        for i in range(len(self.algo_correction.corrected_speed)):
            self.ax3.scatter(range(self.nb_test_var.get()), self.algo_correction.corrected_speed[i][:self.nb_test_var.get(), 2], label=f'w corrigé {i+1} (m/s)', alpha=0.5, s=i+5)

        self.ax3.set_title('Composante w')
        self.ax3.set_xlabel('Temps')
        self.ax3.set_ylabel('Vitesse (m/s)')
        self.ax3.grid()
        self.ax3.legend()
        
        self.canvas.draw()  # Redessiner la figure avec les nouvelles données
        self.result_window.update_idletasks()