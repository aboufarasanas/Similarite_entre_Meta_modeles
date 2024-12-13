from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Définir les éléments des méta-modèles
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

# Convertir les éléments en chaînes de texte
def elements_to_text(elements):
    return {key: ' '.join(value.split()) for key, value in elements.items()}

relational_database_texts = elements_to_text(relational_database_elements)
key_value_store_texts = elements_to_text(key_value_store_elements)

# Vectoriser les textes
vectorizer = TfidfVectorizer()
all_texts = list(relational_database_texts.values()) + list(key_value_store_texts.values())
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Calculer la similarité cosinus entre les éléments des deux méta-modèles
num_relational = len(relational_database_elements)
num_key_value = len(key_value_store_elements)
similarity_matrix = cosine_similarity(tfidf_matrix[:num_relational], tfidf_matrix[num_relational:])

# Créer une DataFrame pour une meilleure visualisation
relational_keys = list(relational_database_elements.keys())
key_value_keys = list(key_value_store_elements.keys())
similarity_df_matrix = pd.DataFrame(similarity_matrix, index=relational_keys, columns=key_value_keys)

# Fonction de Similarité de Monge-Elkan
def monge_elkan_similarity(similarity_matrix):
    # Moyenne des meilleures similarités pour chaque élément
    max_similarities = similarity_matrix.max(axis=1)
    # Moyenne des meilleures similarités pour chaque élément dans le second ensemble
    avg_max_similarities = similarity_matrix.max(axis=0).mean()
    return avg_max_similarities

# Calculer la similarité Monge-Elkan
monge_elkan_sim = monge_elkan_similarity(similarity_matrix)

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

for threshold in np.arange(0, 1.0, 0.01):
    predicted_matches = set(
        (relational_keys[i], key_value_keys[j]) 
        for i in range(num_relational) 
        for j in range(num_key_value) 
        if similarity_matrix[i, j] >= threshold
    )
    _, _, f_measure = calculate_metrics(predicted_matches, real_matches)
    if f_measure > best_f_measure:
        best_f_measure = f_measure
        best_threshold = threshold

# Utiliser le meilleur seuil pour obtenir les correspondances prédites finales
predicted_matches = set(
    (relational_keys[i], key_value_keys[j]) 
    for i in range(num_relational) 
    for j in range(num_key_value) 
    if similarity_matrix[i, j] >= best_threshold
)

# Calculer les métriques finales
precision, recall, f_measure = calculate_metrics(predicted_matches, real_matches)

# Afficher les résultats
print("Monge-Elkan Similarity:")
print(f"Similarity: {monge_elkan_sim}")

print("\nSimilarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Measure: {f_measure}")

# Afficher les correspondances réelles pour vérification
print("\nReal Matches:")
print(real_matches)
