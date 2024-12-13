from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

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

# Créer une liste de toutes les descriptions combinées pour TF-IDF
all_elements = list(relational_database_elements.values()) + list(key_value_store_elements.values())

# Appliquer TF-IDF pour convertir les descriptions en une matrice de caractéristiques
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_elements)

# Appliquer l'Analyse Sémantique Latente (LSA) en utilisant TruncatedSVD
n_components = 2  # Nombre de dimensions réduites
svd = TruncatedSVD(n_components=n_components)
lsa_matrix = svd.fit_transform(tfidf_matrix)

# Séparer la matrice LSA en deux matrices distinctes pour chaque méta-modèle
relational_lsa_matrix = lsa_matrix[:len(relational_database_elements)]
key_value_lsa_matrix = lsa_matrix[len(relational_database_elements):]

# Calculer la similarité cosinus entre les éléments des deux méta-modèles
similarity_matrix = cosine_similarity(relational_lsa_matrix, key_value_lsa_matrix)

# Créer une DataFrame pour une meilleure visualisation
relational_keys = list(relational_database_elements.keys())
key_value_keys = list(key_value_store_elements.keys())
similarity_df_matrix = pd.DataFrame(similarity_matrix, index=relational_keys, columns=key_value_keys)

# Définir le seuil optimal (exemple fixe, ajuster si nécessaire)
best_threshold = 0.5

# Calculer les métriques de performance
pred_matches = set()

for i, rel_key in enumerate(relational_keys):
    for j, kv_key in enumerate(key_value_keys):
        if similarity_df_matrix.iloc[i, j] >= best_threshold:
            pred_matches.add((rel_key, kv_key))

# Exemple de vraies correspondances (à ajuster pour votre cas)
real_matches = {
    ("SQLElement", "KeystoreElement"),
    ("SQLColumn", "KeyValue"),
    ("SQLTable", "Entity")
}

true_positives = len(pred_matches & real_matches)
precision = true_positives / len(pred_matches) if pred_matches else 0
recall = true_positives / len(real_matches) if real_matches else 0
f_measure = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

# Afficher les résultats
print("LSA Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Measure: {f_measure}")

# Afficher les correspondances réelles pour vérification
print("\nReal Matches:")
print(real_matches)
