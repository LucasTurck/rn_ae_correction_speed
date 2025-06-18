from dir import DATA_DIRECTORY
import os
import re

def lire_fichier_U(filepath):
    times = []
    sondes = {}  # Dictionnaire pour stocker les sondes
    colonnes = []  # Liste de listes : colonnes[i][j] = tuple (x, y, z) pour la sonde i au temps j
    positions_sondes = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lignes = f.readlines()


    # Récupère les positions des sondes
    for ligne in lignes:
        if ligne.strip().startswith('# Probe'):
            # Ex: "# Probe 0 (0.17 0.025 0.08)"
            match = re.search(r'# Probe \d+ \(([^)]+)\)', ligne)
            if match:
                pos = tuple(float(x) for x in match.group(1).split())
                positions_sondes.append(pos)


    # Cherche la première ligne de données (ignore les lignes commençant par # ou vides)
    data_lines = [l for l in lignes if l.strip() and not l.strip().startswith('#')]
    if not data_lines:
        return [], [], []

    # Détermine le nombre de sondes (colonnes)
    premiere_ligne = data_lines[0]
    # On suppose que chaque colonne est entre parenthèses
    n_colonnes = premiere_ligne.count('(')
    colonnes = [[] for _ in range(n_colonnes)]

    for ligne in data_lines:
        # Sépare le temps du reste
        parts = ligne.strip().split('(')
        if len(parts) < 2:
            continue
        # Le temps est le premier élément avant la première parenthèse
        time_str = parts[0].strip().split()[0]
        try:
            t = float(time_str)
        except ValueError:
            continue
        times.append(t)

        # Récupère chaque triplet (x, y, z)
        matches = re.findall(r'\(([^)]+)\)', ligne)
        for i, match in enumerate(matches):
            coords = tuple(float(x) for x in match.strip().split())
            colonnes[i].append(coords)
    
    for i in range(len(positions_sondes)):
        sondes[i] = [positions_sondes[i], colonnes[i]]
        
    # On moyenne u pour chaque sonde
    for i in range(len(sondes)):
        mean_u = sum(col[0] for col in sondes[i][1]) / len(sondes[i][1])
        sondes[i][1] = [(col[0] - mean_u, col[1], col[2]) for col in sondes[i][1]]

    return times, sondes

if __name__ == "__main__":
    # Exemple d'utilisation
    e=145
    filepath = os.path.join(DATA_DIRECTORY,'E_'+str(e), 'U')
    if not os.path.exists(filepath):
        print("Le fichier n'existe pas :", filepath)
        exit(1)
    times, sondes = lire_fichier_U(filepath)        
    print("Temps :", times[:4])
    for i, (pos, col) in sondes.items():
        print(f"Sonde {i} (position):", pos)
        print(f"Sonde {i} (extrait):", col[:4])