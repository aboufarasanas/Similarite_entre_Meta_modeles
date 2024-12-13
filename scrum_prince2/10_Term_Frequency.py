from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Définir les éléments des méta-modèles avec leurs attributs et relations sous forme de texte
prince2_elements = {
    "BusinessCase": "ID Title Description",
    "ProjectBoard": "ID Name Members",
    "ProjectPlan": "ID Name StartDate EndDate",
    "StagePlan": "ID Name StageObjective",
    "WorkPackage": "ID Name Tasks",
    "EndStageReport": "ID Name Summary"
}

scrum_elements = {
    "ProductBacklog": "ID Name Description",
    "Sprint": "ID Name Goal StartDate EndDate",
    "ScrumTeam": "ID Name Members",
    "SprintBacklog": "ID Name Tasks",
    "Increment": "ID Name Description Version",
    "DailyScrum": "ID Date Notes"
}

# Combiner les textes des deux méta-modèles
all_elements = list(prince2_elements.values()) + list(scrum_elements.values())

# Utiliser TfidfVectorizer pour créer une représentation TF-IDF
vectorizer = TfidfVectorizer(use_idf=False, norm=None)  # Utilisation de TF uniquement
X = vectorizer.fit_transform(all_elements)

# Calculer la similarité cosine entre les vecteurs de TF
# Les premiers éléments sont de PRINCE2 et les derniers sont de Scrum
similarity_matrix = cosine_similarity(X[:len(prince2_elements)], X[len(prince2_elements):])

# Créer une DataFrame pour visualiser les résultats
prince2_keys = list(prince2_elements.keys())
scrum_keys = list(scrum_elements.keys())
similarity_df_matrix = pd.DataFrame(similarity_matrix, index=prince2_keys, columns=scrum_keys)

# Fonction pour calculer les métriques de performance
def calculate_metrics(predicted, real):
    tp = len(predicted & real)
    fp = len(predicted - real)
    fn = len(real - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure

# Correspondances réelles (exemple hypothétique)
real_matches = {
    ("BusinessCase", "ProductBacklog"),
    ("ProjectBoard", "ScrumTeam"),
    ("ProjectPlan", "Sprint"),
    ("StagePlan", "SprintBacklog"),
    ("WorkPackage", "SprintBacklog"),
    ("EndStageReport", "Increment")
}

# Trouver le meilleur seuil dynamiquement
best_threshold = 0
best_f_measure = 0

for threshold in np.arange(0, 1.0, 0.01):
    predicted_matches = set(
        (prince2_keys[i], scrum_keys[j]) 
        for i in range(len(prince2_keys)) 
        for j in range(len(scrum_keys)) 
        if similarity_matrix[i, j] >= threshold
    )
    _, _, f_measure = calculate_metrics(predicted_matches, real_matches)
    if f_measure > best_f_measure:
        best_f_measure = f_measure
        best_threshold = threshold

# Utiliser le meilleur seuil pour obtenir les correspondances prédites finales
predicted_matches = set(
    (prince2_keys[i], scrum_keys[j]) 
    for i in range(len(prince2_keys)) 
    for j in range(len(scrum_keys)) 
    if similarity_matrix[i, j] >= best_threshold
)

# Calculer les métriques finales
precision, recall, f_measure = calculate_metrics(predicted_matches, real_matches)

# Afficher les résultats
print("Term Frequency (TF) Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Measure: {f_measure}")

# Afficher les correspondances réelles pour vérification
print("\nReal Matches:")
print(real_matches)
