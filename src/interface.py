from ui.UIModelSelector import UIModelSelector
from ui.UIParameters import UIParameters
from ui.UIAlgoCorrection import UIAlgoCorrection
import tkinter as tk
from tkinter import ttk
import os

def open_ui_parameters(parent=None, controller=None):
    """Ouvre l'interface des paramètres du réseau de neurones."""
    global ui_parameters
    ui_parameters = UIParameters(parent, controller)

def open_ui_model_selector(parent=None, controller=None):
    """Ouvre l'interface de sélection du modèle."""
    global ui_model_selector
    ui_model_selector = UIModelSelector(parent, controller)

def open_ui_algo_correction(parent=None, controller=None):
    """Ouvre l'interface de correction de l'algorithme."""
    global ui_algo_correction
    ui_algo_correction = UIAlgoCorrection(parent, controller)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(os.cpu_count())
    os.environ["TF_NUM_INTEROP_THREADS"] = str(os.cpu_count())
    # app = MainApp()
    # app.mainloop()
    
    main_window = tk.Tk()
    main_window.title("Interface Réseau de Neurones")
    # bouton pour ouvrir l'interface UIparameters
    ttk.Button(main_window, text="Paramètres du réseau de neurones", command=lambda: open_ui_parameters(main_window, None)).pack(pady=10)
    # bouton pour ouvrir l'interface UIModelSelector
    ttk.Button(main_window, text="Sélectionner un modèle", command=lambda: open_ui_model_selector(main_window, None)).pack(pady=10)
    # bouton pour ouvrir l'interface UIAlgoCorrection
    ttk.Button(main_window, text="Corriger l'algorithme", command=lambda: open_ui_algo_correction(main_window, None)).pack(pady=10)

    def destroy_ui():
        """Détruit les interfaces UIParameters et UIModelSelector."""
        global ui_parameters, ui_model_selector, ui_algo_correction
        try:
            if ui_parameters:
                ui_parameters.UI_destroy()
            if ui_model_selector:
                ui_model_selector.destroy()
            if ui_algo_correction:
                ui_algo_correction.destroy()
        except Exception as e:
            # print(f"Erreur lors de la destruction des interfaces : {e}")
            pass
        finally:
            ui_parameters = None
            ui_model_selector = None
            ui_algo_correction = None
            main_window.destroy()

    # Bouton pour quitter l'application
    ttk.Button(main_window, text="Quitter", command=destroy_ui).pack(pady=10)

    # Créer une instance de MainApp
    main_window.mainloop()