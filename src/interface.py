from dir import RDN_DIRECTORY, MOD_DIRECTORY, MOD_PERSO_DIRECTORY
# import reseau_neurones module correctly (adjust the import as needed for your project structure)
from pred_f.reseau_neurones import *
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import json
import os
import threading
      

class UIparams(ttk.Frame):
    """Classe pour l'interface utilisateur des paramètres de l'architecture du réseau de neurones."""
    def __init__(self, root, controller):
        super().__init__(root)
        self.controller = controller
        
        dir_params = os.path.join(RDN_DIRECTORY, "last_config.json")
        if not os.path.exists(dir_params):
            dir_params = os.path.join(RDN_DIRECTORY, "default_config.json")
        if not os.path.exists(dir_params):
            raise FileNotFoundError(f"Le fichier de configuration {dir_params} n'existe pas.")
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
        self.erreur_var = tk.StringVar(value=self.params.get('loss', 'mae'))  # Variable pour l'erreur à afficher dans l'historique
        self.create_widgets()

    def create_widgets(self):
        # Ajouter des widgets pour les paramètres

        ## Architecture :
        dir_architectures = os.path.join(RDN_DIRECTORY, "architectures.json")
        if not os.path.exists(dir_architectures):
            raise FileNotFoundError(f"Le fichier de configuration {dir_architectures} n'existe pas.")
        with open(dir_architectures, 'r') as file:
            architectures = json.load(file)
        names_archi = [arch['name'] for arch in architectures]
        ttk.Label(self, text="Architecture :").pack()
        ttk.Combobox(self, textvariable=self.archi_var, values=names_archi, state="readonly").pack()
        
        ## Erreur :
        ttk.Label(self, text="Erreur :").pack()
        names_loss = NAMES_METRICS
        ttk.Combobox(self, values=names_loss, textvariable=self.erreur_var, state="readonly").pack()

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
        ttk.Button(self, text="Compiler le réseau", command=lambda: self.fenetre_compile()).pack()
        
        # Bouton pour afficher l'historique d'entraînement
        # ttk.Button(self, text="Afficher l'historique d'entraînement", command=lambda: self.controller.afficher_historique()).pack()

        # # Bouton pour afficher les résultats de l'entraînement
        # ttk.Button(self, text="Afficher les résultats de l'entraînement", command=lambda: self.controller.afficher_resultats_train()).pack(pady=10)

        # # Bouton pour aficher les résultats du réseau en changeant les données de test
        # ttk.Button(self, text="Afficher les résultats du test", command=lambda: self.controller.afficher_resultats_test()).pack(pady=10)

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
        self.params['loss'] = self.erreur_var.get()  # Enregistrer l'erreur sélectionnée pour l'historique

        # Enregistrer dans le fichier de configuration
        dir_params = os.path.join(RDN_DIRECTORY, "last_config.json")
        with open(dir_params, 'w') as file:
            json.dump(self.params, file, indent=4)
        
        # print("Paramètres enregistrés :", self.params)
        
        # Mettre à jour le modèle avec les nouveaux paramètres
        
        self.controller.set_params(params=self.params)
    
    def fenetre_compile(self):
        self.save_params()  # Enregistrer les paramètres avant de compiler
        UICompileModel(self, self.controller)
        
    # def run_prediction(self):

    #     self.params['num_sonde_test'] = int(self.sonde_test_var.get())
    #     self.params['nb_test'] = int(self.nb_test_var.get())
        
    #     self.controller.set_params(self.params)
    #     self.controller.remplissage_donnees(train=False)
    #     self.controller.predire()

    #     print(f"R² pour le test : {self.controller.evaluer(train=False)}")

    #     self.controller.afficher_resultats()

class UICompileModel(ttk.Frame):
    """Classe pour l'interface utilisateur de compilation du modèle."""
    def __init__(self, parent, controller):
        """Initialise l'interface utilisateur de compilation du modèle."""
        super().__init__(parent)
        self.controller = controller
        self.model_RdN = ModeleReseauNeurones(parameters=parent.params)
        
        # Afficher les résultats dans une nouvelle fenêtre
        results_window = tk.Toplevel(self)
        results_window.title("Résultats")
        
        
        def destroy_window():
            self.model_RdN.clear()
            del self.model_RdN
            # Fermer la fenêtre de résultats
            results_window.destroy()
            
        
        # Bouton pour fermer cette fenêtre
        ttk.Button(results_window, text="Fermer", command=destroy_window).pack(pady=10)

        # Lancer l'entraînement dans un thread pour ne pas bloquer l'UI
        def entrainement():
            # self.model_RdN.set_parameters(parameters=params)
            self.model_RdN.remplissage_donnees()
            self.model_RdN.create_reseau()
            self.model_RdN.entrainer()
            print(f"R² pour l'entraînement : {self.model_RdN.evaluer(train=True)}")
            self.model_RdN.sauvegarder_apprentissage(MOD_PERSO_DIRECTORY)
            # Optionnel : afficher un message à la fin
            # tk.messagebox.showinfo("Info", "Entraînement terminé !")
            print(f"adresse de model_RdN : {self.model_RdN}")
            # Bouton pour afficher l'historique d'entraînement
            ttk.Button(results_window, text="Afficher l'historique d'entraînement", command=lambda: self.controller.afficher_historique(self.model_RdN, results_window)).pack()

            # Bouton pour afficher les résultats de l'entraînement
            ttk.Button(results_window, text="Afficher les résultats de l'entraînement", command=lambda: self.controller.afficher_resultats_train(self.model_RdN, results_window)).pack(pady=10)

            # Bouton pour aficher les résultats du réseau en changeant les données de test
            ttk.Button(results_window, text="Afficher les résultats du test", command=lambda: self.controller.afficher_resultats_test(self.model_RdN, results_window)).pack(pady=10)

        threading.Thread(target=entrainement, daemon=True).start()

        
        
class UIModelSelector(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        # Bouton pour revenir aux paramètres
        ttk.Button(self, text="Retour aux paramètres", command=lambda: controller.show_frame("UIparams")).pack(pady=10) 
        self.controller = controller
        self.archi_var = tk.StringVar()
        self.run_var = tk.StringVar()
        self.params = None
        self.sonde_test_var = None
        self.nb_test_var = None

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
        
        # Bouton pour afficher l'historique d'entraînement
        ttk.Button(self, text="Afficher l'historique d'entraînement", command=lambda: self.controller.afficher_historique()).pack(pady=10)
        
        # Bouton pour afficher les résultats du test
        ttk.Button(self, text="Afficher les résultats du test", command=lambda: self.controller.afficher_resultats_test()).pack(pady=10)

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
                self.controller.remplissage_donnees(train=True)
                self.controller.charger_apprentissage(os.path.join(MOD_DIRECTORY, archi, run))
                
                # self.params = config
                # self.sonde_test_var = tk.StringVar(value=self.params.get('num_sonde_test', 0))
                # self.nb_test_var = tk.StringVar(value=self.params.get('nb_test', 1000))

                # ttk.Label(self, text="Numéro de la sonde de test :").pack()
                # ttk.Entry(self, textvariable=self.sonde_test_var).pack()

                # ttk.Label(self, text="Nombre de tests :").pack()
                # ttk.Entry(self, textvariable=self.nb_test_var).pack()
                
                # ttk.Button(self, text="Lancer la prédiction", command=self.run_prediction).pack(pady=10)
                

            else:
                print("Erreur : Fichier de configuration introuvable.")
        else:
            print("Erreur : Aucune exécution sélectionnée.")
        
    def run_prediction(self):

        self.params['num_sonde_test'] = int(self.sonde_test_var.get())
        self.params['nb_test'] = int(self.nb_test_var.get())
        
        self.controller.set_params(self.params)
        self.controller.remplissage_donnees(train=False)
        self.controller.predire()

        print(f"R² pour le test : {self.controller.evaluer(train=False)}")

        self.controller.afficher_resultats()
        
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interface Réseau de Neurones")
        self.geometry("500x700")
        self.frames = {}
        self.erreur_var = tk.StringVar(value='loss')  # Variable pour l'erreur à afficher dans l'historique
        
        self.model_Rdn = ModeleReseauNeurones()

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        for F in (UIparams, UIModelSelector):
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
    
    def remplissage_donnees(self, train=True):
        self.model_Rdn.remplissage_donnees(train=train)
        
    def charger_apprentissage(self, directory):
        self.model_Rdn.charger_apprentissage(directory)
    
    def predire(self):
        self.model_Rdn.predire()
    
    def evaluer(self, train=True):
        return self.model_Rdn.evaluer(train=train)
    
    def compile_model(self, params):
        
        
        # Afficher les résultats dans une nouvelle fenêtre
        new_window = tk.Toplevel(self)
        new_window.title("Résultats")

        self.model_Rdn.set_parameters(parameters=params)
        self.model_Rdn.remplissage_donnees()
        self.model_Rdn.create_reseau()
        self.model_Rdn.entrainer()
        print(f"R² pour l'entraînement : {self.model_Rdn.evaluer(train=True)}")
        # self.model_Rdn.R2entrainement()
        
        # self.model_Rdn.remplissage_donnees(train=False)
        # self.model_Rdn.predire()
        # print(f"R² pour le test : {self.model_Rdn.evaluer()}")
        # self.model_Rdn.evaluer()
        
        self.model_Rdn.sauvegarder_apprentissage(MOD_PERSO_DIRECTORY)
        
        # self.afficher_resultats(test=False)

    def afficher_resultats_test(self, model_Rdn=None, frame=None):
        if frame is None:
            frame = self
        if model_Rdn is None:
            model_Rdn = self.model_Rdn
        # Afficher les résultats dans une nouvelle fenêtre
        results_window = tk.Toplevel(frame)
        results_window.title("Résultats")
        results_window.geometry("600x400")
        
        # Frame pour les contrôles
        control_frame = ttk.Frame(results_window)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # Variables de tests
        sonde_var = tk.StringVar(value=model_Rdn.parameters.get('num_sonde_test', 0))
        nb_test_var = tk.StringVar(value=model_Rdn.parameters.get('nb_test', 1000))

        # Création de la figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Fonction de mise à jour du graphique
        def update_graph():
            sonde = int(sonde_var.get())
            nb_test = int(nb_test_var.get())
            params = {
                'num_sonde_test': sonde,
                'nb_test': nb_test
            }

            model_Rdn.set_parameters(params)
            model_Rdn.remplissage_donnees(train=False)
            model_Rdn.predire()
            print(f"R² pour le test : {model_Rdn.evaluer(train=False)}")

            if model_Rdn.X_test is not None and model_Rdn.y_test is not None:
                ax.clear()
                # Tracer les données de test
                ax.scatter(model_Rdn.X_test[:,0], 
                           model_Rdn.y_test[:], 
                           label='Données de test', color='blue', s=10)
                
                # Tracer les prédictions
                ax.scatter(model_Rdn.X_test[:,0], 
                           model_Rdn.y_pred_test[:], 
                           label='Prédictions', color='red', s=10)
                
                ax.set_title("Résultats du test")
                ax.set_xlabel("u")
                if model_Rdn.parameters['y'] == 1:
                    ax.set_ylabel("w")
                else:
                    ax.set_ylabel("v")
                ax.legend(title=f"R² : {model_Rdn.r2test:.5f}")
                ax.grid(True)
            else:
                ax.set_title("Aucune donnée de test disponible")
                ax.set_xlabel("u")
                ax.set_ylabel("w")
            canvas.draw()
        # Boutons de contrôle
        ttk.Button(control_frame, text="Rafraîchir", command=update_graph).pack(side="left", padx=10)
        ttk.Label(control_frame, text="Sonde :").pack(side="left")
        ttk.Entry(control_frame, textvariable=sonde_var, width=5).pack(side="left")
        ttk.Label(control_frame, text="Nombre de tests :").pack(side="left")
        ttk.Entry(control_frame, textvariable=nb_test_var, width=6).pack(side="left")
        
        # Bouton pour sauvegarder l'image
        from tkinter import filedialog
        def save_figure():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf")]
            )
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches="tight")
        
        ttk.Button(control_frame, text="Sauvegarder", command=save_figure).pack(side="right")
        ttk.Button(control_frame, text="Fermer", command=results_window.destroy).pack(side="right", padx=10)

        # Canvas pour le graphique
        canvas = FigureCanvasTkAgg(fig, master=results_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, results_window)
        toolbar.update()
        
        # Afficher le graphique initial
        update_graph()
        
        plt.close(fig)  # Fermer la figure matplotlib (pas la fenêtre tkinter)

    def afficher_resultats_train(self, model_Rdn=None, frame=None):

        if frame is None:
            frame = self

        if model_Rdn is None:
            model_Rdn = self.model_Rdn

        if model_Rdn.X_train is None or model_Rdn.y_train is None or model_Rdn.y_pred_train is None:
            tk.messagebox.showwarning("Avertissement", "Aucune donnée d'entraînement disponible.")
            return
        # Afficher les résultats dans une nouvelle fenêtre
        results_window = tk.Toplevel(frame)
        results_window.title("Résultats")
        
        fig, ax = plt.subplots()
        plot_predictions(model_Rdn.X_train, model_Rdn.y_train, model_Rdn.y_pred_train, ax=ax)

        canvas = FigureCanvasTkAgg(fig, master=results_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(results_window, text="Fermer", command=results_window.destroy).pack(pady=10)
        plt.close(fig)

    def afficher_historique(self, model_Rdn=None, frame=None):
        if frame is None:
            frame = self
        if model_Rdn is None:
            model_Rdn = self.model_Rdn
        if model_Rdn.history is None:
            tk.messagebox.showwarning("Avertissement", "Aucun historique d'entraînement disponible.")
            return
        # Créer une fenêtre de résultats avec plus d'options
        results_window = tk.Toplevel(frame)
        results_window.title("Historique d'entraînement")
        results_window.geometry("800x600")  # Taille plus grande
        
        # Frame pour les contrôles
        control_frame = ttk.Frame(results_window)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # Variable pour l'échelle
        # scale_var = tk.StringVar(value="linear")
        start_epoch_var = tk.IntVar(value=2)
        end_epoch_var = tk.IntVar(value=len(model_Rdn.history.history.get(model_Rdn.parameters['loss'], [])) - 1)

        # Création de la figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Fonction de mise à jour du graphique
        def update_graph():
            erreur = self.erreur_var.get()
            import numpy as np
            ax.clear()
            start = start_epoch_var.get()
            end = end_epoch_var.get()
            # Récupérer et tracer les données d'entraînement
            if erreur in model_Rdn.history.history:
                y_train = model_Rdn.history.history[erreur]
                y_train_safe = [v if v != 0 else 1e-8 for v in y_train[start:end]]
                ax.plot(y_train_safe, label='Entraînement', linewidth=2, markersize=3)
                
                # Marquer le point minimal
                min_idx = np.argmin(y_train_safe)
                ax.scatter(min_idx, y_train_safe[min_idx], color='red', s=100, 
                        label=f'Min: {y_train_safe[min_idx]:.5f} (epoch {min_idx})')
            
            # Récupérer et tracer les données de validation
            val_key = f'val_{erreur}'
            if val_key in model_Rdn.history.history:
                y_val = model_Rdn.history.history[val_key]
                y_val_safe = [v if v != 0 else 1e-8 for v in y_val[start:end]]
                ax.plot(y_val_safe, label='Validation', linewidth=2, marker='x', markersize=3, alpha=0.8)
                
                # Marquer le point minimal de validation
                min_val_idx = np.argmin(y_val_safe)
                ax.scatter(min_val_idx, y_val_safe[min_val_idx], color='green', s=100, 
                        label=f'Min val: {y_val_safe[min_val_idx]:.5f} (epoch {min_val_idx+start})')
    
            # Paramétrage du graphique
            ax.set_title(f"Historique d'entraînement - {erreur}", fontsize=14)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(f"{erreur.upper()}", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            # ax.set_yscale(scale_var.get())


            # Ajouter une annotation pour la valeur finale
            if erreur in model_Rdn.history.history:
                final_value = y_train_safe[-1]
                ax.annotate(f'Valeur finale: {final_value:.5f}', 
                        xy=(len(y_train_safe)-1, final_value),
                        xytext=(len(y_train_safe)*0.8, y_train_safe[start]),
                        arrowprops=dict(arrowstyle='->'))
            
            ax.legend(loc='best')
            canvas.draw()
        
        # # Boutons de contrôle
        # ttk.Label(control_frame, text="Échelle:").pack(side="left")
        # ttk.Radiobutton(control_frame, text="Linéaire", variable=scale_var, 
        #             value="linear", command=update_graph).pack(side="left")
        # ttk.Radiobutton(control_frame, text="Logarithmique", variable=scale_var, 
        #             value="log", command=update_graph).pack(side="left")
        # ttk.Radiobutton(control_frame, text="Logarithmique symétrique", variable=scale_var,
        #             value="logit", command=update_graph).pack(side="left")
        ttk.Button(control_frame, text="Rafraîchir", command=update_graph).pack(side="left", padx=10)
        ttk.Label(control_frame, text="Début:").pack(side="left")
        ttk.Entry(control_frame, textvariable=start_epoch_var, width=5).pack(side="left")
        ttk.Label(control_frame, text="Fin:").pack(side="left")
        ttk.Entry(control_frame, textvariable=end_epoch_var, width=5).pack(side="left")
        
        ttk.Radiobutton(control_frame, text="mae", variable=self.erreur_var,
                    value="mae", command=update_graph).pack(side="left")
        ttk.Radiobutton(control_frame, text="mse", variable=self.erreur_var,
                    value="mse", command=update_graph).pack(side="left")

        # Bouton pour sauvegarder l'image
        from tkinter import filedialog
        def save_figure():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf")]
            )
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches="tight")
        
        ttk.Button(control_frame, text="Sauvegarder", command=save_figure).pack(side="right")
        ttk.Button(control_frame, text="Fermer", command=results_window.destroy).pack(side="right", padx=10)
    
        # Canvas pour le graphique
        canvas = FigureCanvasTkAgg(fig, master=results_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, results_window)
        toolbar.update()
        
        # Afficher le graphique initial
        update_graph()
        
        plt.close(fig)  # Fermer la figure matplotlib (pas la fenêtre tkinter)

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(os.cpu_count())
    os.environ["TF_NUM_INTEROP_THREADS"] = str(os.cpu_count())
    app = MainApp()
    app.mainloop()
