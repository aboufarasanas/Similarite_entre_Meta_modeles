import pandas as pd
import numpy as np

# Fonction pour calculer la similarité de Jaccard entre deux ensembles
def jaccard_similarity(set1, set2):
    if not set1 and not set2:  # Si les deux ensembles sont vides
        return 1.0
    elif not set1 or not set2:  # Si l'un des ensembles est vide
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Eléments des méta-modèles PRINCE2 et Scrum
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

# Correspondances réelles (Mappings fournies)
real_matches = {
    ("BusinessCase", "ProductBacklog"),
    ("ProjectBoard", "ScrumTeam"),
    ("ProjectPlan", "Sprint"),
    ("StagePlan", "SprintBacklog"),
    ("WorkPackage", "Increment"),
    ("EndStageReport", "DailyScrum")
}

# Calcul de la matrice de similarité de Jaccard
similarity_matrix = []
for p_elem in prince2_elements:
    row = []
    for s_elem in scrum_elements:
        similarity = jaccard_similarity(set(prince2_elements[p_elem]), set(scrum_elements[s_elem]))
        row.append(similarity)
    similarity_matrix.append(row)

# Conversion en DataFrame pour un affichage plus lisible
similarity_df_matrix = pd.DataFrame(similarity_matrix, 
                                    index=prince2_elements.keys(), 
                                    columns=scrum_elements.keys())

# Recherche du meilleur seuil
thresholds = np.arange(0.1, 1.0, 0.1)
best_threshold = 0
best_f_measure = 0
best_precision = 0
best_recall = 0

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
print("Jaccard Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {best_precision}")
print(f"Recall: {best_recall}")
print(f"F-Measure: {best_f_measure}")

# Affichage des correspondances réelles pour vérification
print("\nReal Matches:")
print(real_matches)
