import pandas as pd
import numpy as np
import Levenshtein as lev

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

# Convertir les éléments en chaînes pour la comparaison
def convert_to_string(element):
    return ' '.join(element)

# Calcul de la matrice de distance de Levenshtein
distance_matrix = []
for p_elem in prince2_elements:
    row = []
    for s_elem in scrum_elements:
        p_str = convert_to_string(prince2_elements[p_elem])
        s_str = convert_to_string(scrum_elements[s_elem])
        
        distance = lev.distance(p_str, s_str)
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
print("Levenshtein Distance Matrix:")
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
