import pandas as pd
import numpy as np

def hamming_distance(s1, s2):
    """Calculer la distance de Hamming entre deux chaînes de même longueur."""
    if len(s1) != len(s2):
        raise ValueError("Les chaînes doivent être de même longueur")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

# Éléments des méta-modèles PRINCE2 et Scrum
prince2_elements = {
    "BusinessCase": ["ID", "Title", "Description"],
    "ProjectBoard": ["ID", "Name", "Members"],
    "ProjectPlan": ["ID", "Name", "StartDate", "EndDate"],
    "StagePlan": ["ID", "Name", "StageObjective"],
    "WorkPackage": ["ID", "Name", "Tasks"],
    "EndStageReport": ["ID", "Name", "Summary"]
}

scrum_elements = {
    "ProductBacklog": ["ID", "Name", "Description"],
    "Sprint": ["ID", "Name", "Goal", "StartDate", "EndDate"],
    "ScrumTeam": ["ID", "Name", "Members"],
    "SprintBacklog": ["ID", "Name", "Tasks"],
    "Increment": ["ID", "Name", "Description", "Version"],
    "DailyScrum": ["ID", "Date", "Notes"]
}

# Convertir les éléments en chaînes de même longueur pour la comparaison
def pad_to_max_length(element_list, max_length):
    return [e.ljust(max_length, ' ') for e in element_list]

# Trouver la longueur maximale pour les éléments
max_length_prince2 = max(len(' '.join(prince2_elements[k])) for k in prince2_elements)
max_length_scrum = max(len(' '.join(scrum_elements[k])) for k in scrum_elements)
max_length = max(max_length_prince2, max_length_scrum)

# Calcul de la matrice de distance de Hamming
distance_matrix = []
for p_elem in prince2_elements:
    row = []
    for s_elem in scrum_elements:
        p_str = ' '.join(pad_to_max_length(prince2_elements[p_elem], max_length))
        s_str = ' '.join(pad_to_max_length(scrum_elements[s_elem], max_length))
        
        # Calculer la distance de Hamming
        try:
            distance = hamming_distance(p_str, s_str)
        except ValueError:
            # En cas de chaînes de longueurs différentes
            distance = np.nan
        
        row.append(distance)
    distance_matrix.append(row)

# Conversion en DataFrame pour un affichage plus lisible
distance_df_matrix = pd.DataFrame(distance_matrix, 
                                  index=prince2_elements.keys(), 
                                  columns=scrum_elements.keys())

# Normalisation pour obtenir des similarités (1 - distance normalisée)
max_distance = np.nanmax(distance_df_matrix.values)
similarity_df_matrix = 1 - (distance_df_matrix / max_distance)

# Recherche du meilleur seuil
thresholds = np.arange(0.1, 1.0, 0.1)
best_threshold = 0
best_f_measure = 0
best_precision = 0
best_recall = 0

# Correspondances réelles (mappings fournis)
real_matches = {
    ("BusinessCase", "ProductBacklog"),
    ("ProjectBoard", "ScrumTeam"),
    ("ProjectPlan", "Sprint"),
    ("StagePlan", "SprintBacklog"),
    ("WorkPackage", "Increment"),
    ("EndStageReport", "DailyScrum")
}

for threshold in thresholds:
    found_matches = set()
    for i, p_elem in enumerate(prince2_elements.keys()):
        for j, s_elem in enumerate(scrum_elements.keys()):
            if similarity_df_matrix.iloc[i, j] >= threshold:
                found_matches.add((p_elem, s_elem))
    
    true_positives = len(found_matches.intersection(real_matches))
    false_positives = len(found_matches.difference(real_matches))
    false_negatives = len(real_matches.difference(found_matches))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    if f_measure > best_f_measure:
        best_f_measure = f_measure
        best_threshold = threshold
        best_precision = precision
        best_recall = recall

# Affichage des résultats
print("Hamming Distance Matrix:")
print(distance_df_matrix)
print("\nSimilarity Matrix (normalized):")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {best_precision}")
print(f"Recall: {best_recall}")
print(f"F-Measure: {best_f_measure}")

# Affichage des correspondances réelles pour vérification
print("\nReal Matches:")
print(real_matches)
