import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

# Définir les éléments des méta-modèles
relational_database_elements = {
    "SQLElement": {"Name"},
    "SQLColumn": {"Type", "Kind", "Name"},
    "SQLTable": {"Name"}
}

key_value_store_elements = {
    "KeystoreElement": {"Name"},
    "KeyValue": {"Key", "Value", "Name"},
    "Entity": {"Name"}
}

# Correspondances réelles (hypothétiques)
real_matches = {
    ("SQLElement", "KeystoreElement"),
    ("SQLColumn", "KeyValue"),
    ("SQLTable", "Entity")
}

# Applatir les attributs pour chaque élément pour la comparaison
relational_database_flatten = {k: v for k, v in relational_database_elements.items()}
key_value_store_flatten = {k: v for k, v in key_value_store_elements.items()}

# Obtenir toutes les étiquettes d'attributs uniques
all_labels = list(set.union(*relational_database_flatten.values(), *key_value_store_flatten.values()))

# Encoder les attributs comme vecteurs binaires
mlb = MultiLabelBinarizer(classes=all_labels)
relational_database_vectors = mlb.fit_transform(relational_database_flatten.values())
key_value_store_vectors = mlb.transform(key_value_store_flatten.values())

# Calculer la similarité de Jaccard
similarity_matrix = np.zeros((len(relational_database_vectors), len(key_value_store_vectors)))

for i, rd_vec in enumerate(relational_database_vectors):
    for j, kv_vec in enumerate(key_value_store_vectors):
        similarity_matrix[i, j] = jaccard_score(rd_vec, kv_vec)

# Créer une DataFrame pour une meilleure visualisation
relational_keys = list(relational_database_flatten.keys())
key_value_keys = list(key_value_store_flatten.keys())
similarity_df_matrix = pd.DataFrame(similarity_matrix, index=relational_keys, columns=key_value_keys)

# Fonction pour calculer la précision, le rappel et la F-mesure
def calculate_metrics(predicted, real):
    tp = len(predicted & real)
    fp = len(predicted - real)
    fn = len(real - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure

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
print("Jaccard Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Measure: {f_measure}")

# Afficher les correspondances réelles pour vérification
print("\nReal Matches:")
print(real_matches)
