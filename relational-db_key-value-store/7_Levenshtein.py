import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance

# Définir les descriptions textuelles pour chaque élément de méta-modèle
relational_database_elements = {
    "SQLElement": "Name",
    "SQLColumn": "Type Kind Name",
    "SQLTable": "Name"
}

key_value_store_elements = {
    "KeystoreElement": "Name",
    "KeyValue": "Key Value Name",
    "Entity": "Name"
}

# Fonction pour calculer la similarité de Levenshtein
def levenshtein_similarity(str1, str2):
    lev_dist = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0  # Si les deux chaînes sont vides
    return 1.0 - lev_dist / max_len

# Initialiser une matrice de similarité vide
similarity_matrix = np.zeros((len(relational_database_elements), len(key_value_store_elements)))

# Remplir la matrice de similarité
for i, (rel_key, rel_desc) in enumerate(relational_database_elements.items()):
    for j, (kv_key, kv_desc) in enumerate(key_value_store_elements.items()):
        similarity_matrix[i, j] = levenshtein_similarity(rel_desc, kv_desc)

# Créer une DataFrame pour une meilleure visualisation
relational_keys = list(relational_database_elements.keys())
key_value_keys = list(key_value_store_elements.keys())
similarity_df_matrix = pd.DataFrame(similarity_matrix, index=relational_keys, columns=key_value_keys)

# Fonction pour calculer les métriques de performance
def calculate_metrics(predicted, real):
    tp = len(predicted & real)
    fp = len(predicted - real)
    fn = len(real - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure

# Vraies correspondances (à ajuster pour votre cas)
real_matches = {
    ("SQLElement", "KeystoreElement"),
    ("SQLColumn", "KeyValue"),
    ("SQLTable", "Entity")
}

# Trouver le meilleur seuil dynamiquement
best_threshold = 0
best_f_measure = 0

for threshold in np.arange(0.1, 1.0, 0.01):
    predicted_matches = set(
        (relational_keys[i], key_value_keys[j]) 
        for i in range(len(relational_keys)) 
        for j in range(len(key_value_keys)) 
        if similarity_matrix[i, j] >= threshold
    )
    _, _, f_measure = calculate_metrics(predicted_matches, real_matches)
    if f_measure > best_f_measure:
        best_f_measure = f_measure
        best_threshold = threshold

# Utiliser le meilleur seuil pour obtenir les correspondances prédites finales
predicted_matches = set(
    (relational_keys[i], key_value_keys[j]) 
    for i in range(len(relational_keys)) 
    for j in range(len(key_value_keys)) 
    if similarity_matrix[i, j] >= best_threshold
)

# Calculer les métriques finales
precision, recall, f_measure = calculate_metrics(predicted_matches, real_matches)

# Afficher les résultats
print("Levenshtein Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Measure: {f_measure}")

# Afficher les correspondances réelles pour vérification
print("\nReal Matches:")
print(real_matches)
