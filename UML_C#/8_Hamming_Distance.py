import pandas as pd
import numpy as np

# Fonction pour calculer la distance de Hamming
def hamming_distance(s1, s2):
    # Assurer que les chaînes ont la même longueur
    if len(s1) != len(s2):
        raise ValueError("Les chaînes doivent avoir la même longueur pour calculer la distance de Hamming.")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

# Éléments des méta-modèles UML et CSharp
uml_elements = {
    "UMLModelElement": ["Name"],
    "UMLAttribute": ["isFinal", "isStatic", "Type"],
    "UMLClassifier": ["isFinal", "isPublic"],
    "UMLOperation": ["isFinal", "isStatic", "returnType"],
    "UMLParameter": ["isFinal", "parameterType"],
    "UMLClass": [],
    "UMLInterface": []
}

csharp_elements = {
    "CSharpElement": ["Name"],
    "CSharpField": ["isFinal", "isStatic", "Type"],
    "CSharpClass": ["isFinal", "isPublic"],
    "CSharpMethod": ["isFinal", "isStatic", "returnType"],
    "CSharpParameter": ["isFinal", "parameterType"],
    "CSharpInterface": ["isFinal", "isPublic"]
}

# Convertir les éléments en chaînes pour la comparaison (remplir avec des espaces pour égaliser la longueur)
def convert_to_fixed_length_string(element, length):
    return ' '.join(element).ljust(length)

# Trouver la longueur maximale des chaînes pour la comparaison
max_length = max(max(len(' '.join(e)) for e in uml_elements.values()),
                  max(len(' '.join(e)) for e in csharp_elements.values()))

# Calcul de la matrice de distance de Hamming
distance_matrix = []
for u_elem in uml_elements:
    row = []
    for c_elem in csharp_elements:
        u_str = convert_to_fixed_length_string(uml_elements[u_elem], max_length)
        c_str = convert_to_fixed_length_string(csharp_elements[c_elem], max_length)
        
        try:
            distance = hamming_distance(u_str, c_str)
        except ValueError:
            distance = np.nan  # Utiliser NaN pour les longueurs différentes
        row.append(distance)
    distance_matrix.append(row)

# Conversion en DataFrame pour un affichage plus lisible
distance_df_matrix = pd.DataFrame(distance_matrix, 
                                  index=uml_elements.keys(), 
                                  columns=csharp_elements.keys())

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
    ("UMLModelElement", "CSharpElement"),
    ("UMLAttribute", "CSharpField"),
    ("UMLClassifier", "CSharpClass"),
    ("UMLOperation", "CSharpMethod"),
    ("UMLParameter", "CSharpParameter"),
    ("UMLInterface", "CSharpInterface")
}

for threshold in thresholds:
    found_matches = set()
    for i, u_elem in enumerate(uml_elements.keys()):
        for j, c_elem in enumerate(csharp_elements.keys()):
            if similarity_df_matrix.iloc[i, j] >= threshold:
                found_matches.add((u_elem, c_elem))
    
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
