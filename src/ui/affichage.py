import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import numpy as np

def save_figure(fig, defaultpath, defaultname):
    # Fonction pour sauvegarder la figure
    if fig is None:
        raise ValueError("La figure ne peut pas être None.")
    
    # Demande à l'utilisateur où sauvegarder la figure
    file_path = tk.filedialog.asksaveasfilename(
        initialdir=defaultpath,
        initialfile=defaultname,
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf")]
    )
    if file_path:
        fig.savefig(file_path, dpi=300, bbox_inches='tight')

def affichage_resultats_train(model = None, parent = None):
    # Fonction pour afficher les résultats de l'entraînement
    if model is None:
        raise ValueError("Le modèle doit être fourni pour afficher les résultats de l'entraînement.")
    if parent is None:
        raise ValueError("La fenêtre parent doit être fournie pour afficher les résultats de l'entraînement.")

    if model.X_train is None or model.y_train is None or model.y_pred_train is None:
        tk.messagebox.showwarning("Avertissement", "Aucune donnée d'entraînement disponible.")
        return

    # On affiche les résultat dans une nouvelle fenêtre
    result_window = tk.Toplevel(parent)
    result_window.title("Résultats de l'entraînement")
    result_window.geometry("400x300")
    
    # Frame pour les contrôles
    control_frame = tk.Frame(result_window)
    control_frame.pack(fill="x", padx=10, pady=5)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Canvas pour afficher la figure
    canvas = FigureCanvasTkAgg(fig, master=result_window)

    # Bouton pour fermer la fenêtre
    tk.Button(control_frame, text="Fermer", command=result_window.destroy).pack(side="right")

    ax.scatter(model.X_train[:, -1, 0], model.y_train, label='Données réelles', color='blue')
    ax.scatter(model.X_train[:, -1, 0], model.y_pred_train, label='Prédictions', color='red')
    ax.set_xlabel("u")
    if model.parameters['y'] == 1:
        ax.set_ylabel("w")
    elif model.parameters['y'] == 2:
        ax.set_ylabel("v")
    ax.legend(title=f"R² : {model.r2train:.5f}")
    ax.set_title(f"train - {model.parameters['architecture']}, epochs : {model.parameters['epochs']}, loss : {model.parameters['loss']}")
    ax.grid(True)
    
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Bouton pour sauvegarder la figure
    tk.Button(control_frame, text="Sauvegarder la figure", command=lambda: save_figure(fig=fig, defaultname=f"train - {model.parameters['architecture']}, epochs : {model.parameters['epochs']}, loss : {model.parameters['loss']}")).pack(side="left")
    
    
    # Toolbar
    toolbar = NavigationToolbar2Tk(canvas, result_window)
    toolbar.update()
    
    
    plt.close(fig)  # Ferme la figure pour libérer la mémoire

def affichage_resultats_test(model = None, parent = None, simulate=False):
    # Fonction pour afficher les résultats du test
    if model is None:
        raise ValueError("Le modèle doit être fourni pour afficher les résultats du test.")
    if parent is None:
        raise ValueError("La fenêtre parent doit être fournie pour afficher les résultats du test.")
    
    # Afficher les résultats dans une nouvelle fenêtre
    result_window = tk.Toplevel(parent)
    result_window.title("Résultats du test")
    result_window.geometry("600x450")
    
    # Frame pour les contrôles
    control_frame = tk.Frame(result_window)
    control_frame.pack(fill="x", padx=10, pady=5)
    
    # variables de tests
    sonde_test_var = tk.IntVar(value=model.parameters.get('num_sonde_test', 0))
    nb_test_var = tk.IntVar(value=model.parameters.get('nb_test', 100))
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Fonction de mise à jour des résultats de test
    def update_results():
        sonde_test = int(sonde_test_var.get())
        nb_test = int(nb_test_var.get())
        
        parameters = {
            'num_sonde_test': sonde_test,
            'nb_test': nb_test
        }
        
        model.set_parameters(parameters)
        model.remplissage_donnees(train=False, simulate=simulate)
        model.predire()
        print(f"R² pour le test : {model.evaluer(train=False)}")
        
        if model.X_test is None or model.y_test is None or model.y_pred_test is None:
            tk.messagebox.showwarning("Avertissement", "Aucune donnée de test disponible.")
            return
        ax.clear()
        
        # tracer les données de test
        ax.scatter(model.X_test[:, -1, 0], model.y_test, label='Données réelles', color='blue')

        # tracer les prédictions
        ax.scatter(model.X_test[:, -1, 0], model.y_pred_test, label='Prédictions', color='red')

        ax.set_xlabel("u")
        if model.parameters['y'] == 1:
            ax.set_ylabel("w")
        elif model.parameters['y'] == 2:
            ax.set_ylabel("v")
        ax.legend(title=f"R² : {model.r2test:.5f}")
        ax.set_title(f"test - {model.parameters['architecture']}, epochs : {model.parameters['epochs']}, loss : {model.parameters['loss']}")
        ax.grid(True)

        canvas.draw()

    # Boutons de contrôles
    tk.Button(control_frame, text="update", command=update_results).pack(side="left")
    tk.Label(control_frame, text="Sonde de test :").pack(side="left")
    tk.Entry(control_frame, textvariable=sonde_test_var, width=5).pack(side="left")
    tk.Label(control_frame, text="Nombre de tests :").pack(side="left")
    tk.Entry(control_frame, textvariable=nb_test_var, width=5).pack(side="left")
    

    tk.Button(control_frame, text="Sauvegarder", command=save_figure).pack(side="right")
    tk.Button(control_frame, text="Fermer", command=result_window.destroy).pack(side="right", padx=10)

    # Canvas pour afficher la figure
    canvas = FigureCanvasTkAgg(fig, master=result_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Toolbar
    toolbar = NavigationToolbar2Tk(canvas, result_window)
    toolbar.update()
    
    update_results()  # Mettre à jour les résultats de test initialement
    
    plt.close(fig)  # Ferme la figure pour libérer la mémoire

def affichage_historique(model = None, parent = None):
    # Fonction pour afficher l'historique d'entraînement
    if model is None:
        raise ValueError("Le modèle doit être fourni pour afficher l'historique d'entraînement.")
    if parent is None:
        raise ValueError("La fenêtre parent doit être fournie pour afficher l'historique d'entraînement.")
    if model.history is None:
        tk.messagebox.showwarning("Avertissement", "Aucun historique d'entraînement disponible.")
        return
    
    # Afficher l'historique dans une nouvelle fenêtre
    history_window = tk.Toplevel(parent)
    history_window.title("Historique d'entraînement")
    history_window.geometry("600x400")

    # Frame pour les contrôles
    control_frame = tk.Frame(history_window)
    control_frame.pack(fill="x", padx=10, pady=5)
    
    # Variables pour l'historique
    start_epoch_var = tk.IntVar(value=3)
    end_epoch_var = tk.IntVar(value=len(model.history.history.get(model.parameters['loss'], []))-1)
    loss_var = tk.StringVar(value=model.parameters.get('loss', 'mae'))  # Variable pour l'erreur à afficher dans l'historique
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Fonction de mise à jour de l'historique
    def update_history():
        start_epoch = int(start_epoch_var.get())
        end_epoch = int(end_epoch_var.get())
        loss = loss_var.get()
        ax.clear()
        
        if loss not in model.history.history:
            tk.messagebox.showwarning("Avertissement", f"Erreur '{loss}' non trouvée dans l'historique.")
            return
        
        # Récupérer et tracer l'historique d'entrainement
        if loss in model.history.history:
            loss_values = model.history.history[loss][start_epoch-1:end_epoch]
            ax.plot(loss_values, label=loss, linewidth=2, markersize=3)
            
            # Marquer le point minimal
            min_idx = np.argmin(loss_values)
            ax.scatter(min_idx, loss_values[min_idx], color='red', s=100,
                       label=f"Min: {loss_values[min_idx]:.4f} (Epoch {min_idx + start_epoch})")
        
        # Récupérer et tracer l'historique de validation si disponible
        val_loss = f"val_{loss}"
        if val_loss in model.history.history:
            val_loss_values = model.history.history[val_loss][start_epoch-1:end_epoch]
            ax.plot(val_loss_values, label=val_loss, linewidth=2, markersize=3)
            
            # Marquer le point minimal
            min_idx = np.argmin(val_loss_values)
            ax.scatter(min_idx, val_loss_values[min_idx], color='green', s=100,
                       label=f"Min: {val_loss_values[min_idx]:.4f} (Epoch {min_idx + start_epoch})")
            
        ax.set_xlabel("Epochs")
        ax.set_ylabel(f"{loss.capitalize()}", fontsize=12)
        ax.set_title(f"Historique d'entraînement - {model.parameters['architecture']}, epochs : {model.parameters['epochs']}, loss : {model.parameters['loss']}")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Ajouter à la legende la valeure minimale
        ax.legend(title=f"Valeur finale : {loss_values[-1]:.4f} (Epoch {end_epoch})")
        
        ax.legend(loc='upper right')
        canvas.draw()
        
    # Boutons de contrôles
    tk.Button(control_frame, text="update", command=update_history).pack(side="left")
    tk.Label(control_frame, text="Start Epoch:").pack(side="left")
    tk.Entry(control_frame, textvariable=start_epoch_var, width=5).pack(side="left")
    tk.Label(control_frame, text="End Epoch:").pack(side="left")
    tk.Entry(control_frame, textvariable=end_epoch_var, width=5).pack(side="left")
    
    tk.Radiobutton(control_frame, text="mae", variable=loss_var, value="mae", command=update_history).pack(side="left")
    tk.Radiobutton(control_frame, text="mse", variable=loss_var, value="mse", command=update_history).pack(side="left")
    
    # Bouton de sauvegarde et de fermeture
    tk.Button(control_frame, text="Sauvegarder", command=lambda: save_figure(fig=fig, defaultname=f"historique - {model.parameters['architecture']}, epochs : {model.parameters['epochs']}, loss : {model.parameters['loss']}")).pack(side="right")
    
    tk.Button(control_frame, text="Fermer", command=history_window.destroy).pack(side="right", padx=10)
    
    # Canvas pour afficher la figure
    canvas = FigureCanvasTkAgg(fig, master=history_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Toolbar
    toolbar = NavigationToolbar2Tk(canvas, history_window)
    toolbar.update()
    
    update_history()  # Mettre à jour l'historique initialement
    
    plt.close(fig)  # Ferme la figure pour libérer la mémoire