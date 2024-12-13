import numpy as np
import pandas as pd
from pyjarowinkler import distance

# Define UML and C# metamodels
uml_classes = {
    "UMLModelElement": ["Name"],
    "UMLAttribute": ["isFinal", "isStatic", "Type"],
    "UMLClassifier": ["isFinal", "isPublic"],
    "UMLOperation": ["isFinal", "isStatic", "returnType"],
    "UMLParameter": ["isFinal", "parameterType"],
    "UMLClass": [],
    "UMLInterface": []
}

csharp_classes = {
    "CSharpElement": ["Name"],
    "CSharpField": ["isFinal", "isStatic", "Type"],
    "CSharpClass": ["isFinal", "isPublic"],
    "CSharpMethod": ["isFinal", "isStatic", "returnType"],
    "CSharpParameter": ["isFinal", "parameterType"],
    "CSharpInterface": ["isFinal", "isPublic"]
}

# Convert attribute lists to strings for comparison
def convert_to_text(class_dict):
    return {k: ' '.join(v) if v else '' for k, v in class_dict.items()}

uml_texts = convert_to_text(uml_classes)
csharp_texts = convert_to_text(csharp_classes)

# Function to compute Jaro-Winkler similarity
def jaro_winkler_similarity(s1, s2):
    # Ensure neither string is None
    if s1 is None:
        s1 = ""
    if s2 is None:
        s2 = ""
    try:
        return distance.get_jaro_distance(s1, s2, scaling=0.1)
    except Exception as e:
        print(f"Error calculating similarity for ({s1}, {s2}): {e}")
        return 0.0  # Return a default value in case of error

# Initialize similarity matrix
n_uml = len(uml_texts)
n_csharp = len(csharp_texts)
similarity_matrix = np.zeros((n_uml, n_csharp))

# Compute Jaro-Winkler similarity matrix
for i, (uml_key, uml_text) in enumerate(uml_texts.items()):
    for j, (csharp_key, csharp_text) in enumerate(csharp_texts.items()):
        similarity_matrix[i, j] = jaro_winkler_similarity(uml_text, csharp_text)

# Create DataFrame for better visualization
uml_keys = list(uml_texts.keys())
csharp_keys = list(csharp_texts.keys())
similarity_df_matrix = pd.DataFrame(similarity_matrix, index=uml_keys, columns=csharp_keys)

# Define range of thresholds to test
thresholds = np.arange(0.1, 1.0, 0.1)

# Real correspondences (update these based on your actual matches)
real_matches = {
    ("UMLModelElement", "CSharpElement"),
    ("UMLAttribute", "CSharpField"),
    ("UMLClassifier", "CSharpClass"),
    ("UMLOperation", "CSharpMethod"),
    ("UMLParameter", "CSharpParameter"),
    ("UMLClass", "CSharpClass"),
    ("UMLInterface", "CSharpInterface")
}

# Function to calculate metrics for a given threshold
def calculate_metrics(threshold):
    predicted_matches = set(
        (uml_keys[i], csharp_keys[j])
        for i in range(len(uml_keys))
        for j in range(len(csharp_keys))
        if similarity_matrix[i, j] >= threshold
    )
    
    tp = len(predicted_matches & real_matches)
    fp = len(predicted_matches - real_matches)
    fn = len(real_matches - predicted_matches)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure

# Find the best threshold
best_threshold = None
best_precision = best_recall = best_f_measure = 0

for threshold in thresholds:
    precision, recall, f_measure = calculate_metrics(threshold)
    if f_measure > best_f_measure:
        best_f_measure = f_measure
        best_precision = precision
        best_recall = recall
        best_threshold = threshold

# Output results
print("Jaro-Winkler Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold:.1f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall: {best_recall:.4f}")
print(f"F-Measure: {best_f_measure:.4f}")
