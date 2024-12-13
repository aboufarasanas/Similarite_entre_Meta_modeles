from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Préparer les descriptions des éléments des méta-modèles
uml_elements = {
    "UMLModelElement": "Name",
    "UMLAttribute": "isFinal isStatic Type",
    "UMLClassifier": "isFinal isPublic",
    "UMLOperation": "isFinal isStatic returnType",
    "UMLParameter": "isFinal parameterType",
    "UMLClass": "",
    "UMLInterface": ""
}

csharp_elements = {
    "CSharpElement": "Name",
    "CSharpField": "isFinal isStatic Type",
    "CSharpClass": "isFinal isPublic",
    "CSharpMethod": "isFinal isStatic returnType",
    "CSharpParameter": "isFinal parameterType",
    "CSharpInterface": "isFinal isPublic"
}

# Combiner les descriptions des deux méta-modèles
uml_descriptions = list(uml_elements.values())
csharp_descriptions = list(csharp_elements.values())

# Utiliser CountVectorizer pour le modèle Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(uml_descriptions + csharp_descriptions)

# Calculer la similarité cosinus entre les vecteurs BoW
similarity_matrix = cosine_similarity(X[:len(uml_elements)], X[len(uml_elements):])

# Créer une DataFrame pour visualiser les résultats
uml_keys = list(uml_elements.keys())
csharp_keys = list(csharp_elements.keys())
similarity_df_matrix = pd.DataFrame(similarity_matrix, index=uml_keys, columns=csharp_keys)

# Fonction pour calculer les métriques de performance
def calculate_metrics(predicted, real):
    tp = len(predicted & real)
    fp = len(predicted - real)
    fn = len(real - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure

# Correspondances réelles
real_matches = {
    ("UMLModelElement", "CSharpElement"),
    ("UMLAttribute", "CSharpField"),
    ("UMLClassifier", "CSharpClass"),
    ("UMLOperation", "CSharpMethod"),
    ("UMLParameter", "CSharpParameter"),
    ("UMLInterface", "CSharpInterface")
}

# Trouver le meilleur seuil dynamiquement
best_threshold = 0
best_f_measure = 0

for threshold in np.arange(0, 1.0, 0.01):
    predicted_matches = set(
        (uml_keys[i], csharp_keys[j]) 
        for i in range(len(uml_keys)) 
        for j in range(len(csharp_keys)) 
        if similarity_matrix[i, j] >= threshold
    )
    _, _, f_measure = calculate_metrics(predicted_matches, real_matches)
    if f_measure > best_f_measure:
        best_f_measure = f_measure
        best_threshold = threshold

# Utiliser le meilleur seuil pour obtenir les correspondances prédites finales
predicted_matches = set(
    (uml_keys[i], csharp_keys[j]) 
    for i in range(len(uml_keys)) 
    for j in range(len(csharp_keys)) 
    if similarity_matrix[i, j] >= best_threshold
)

# Calculer les métriques finales
precision, recall, f_measure = calculate_metrics(predicted_matches, real_matches)

# Afficher les résultats
print("Bag of Words Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Measure: {f_measure}")

# Afficher les correspondances réelles pour vérification
print("\nReal Matches:")
print(real_matches)
