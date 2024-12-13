import numpy as np
import pandas as pd

# Define the UML and C# metamodels
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

# Flatten the attributes into sets for comparison
uml_elements = {k: set(v) for k, v in uml_classes.items()}
csharp_elements = {k: set(v) for k, v in csharp_classes.items()}

# Real correspondences
real_matches = {
    ("UMLModelElement", "CSharpElement"),
    ("UMLAttribute", "CSharpField"),
    ("UMLClassifier", "CSharpClass"),
    ("UMLOperation", "CSharpMethod"),
    ("UMLParameter", "CSharpParameter"),
    ("UMLInterface", "CSharpInterface")
}

# Function to calculate Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

# Create a similarity matrix
uml_keys = list(uml_elements.keys())
csharp_keys = list(csharp_elements.keys())
similarity_matrix = np.zeros((len(uml_keys), len(csharp_keys)))

for i, uml_key in enumerate(uml_keys):
    for j, csharp_key in enumerate(csharp_keys):
        similarity_matrix[i, j] = jaccard_similarity(uml_elements[uml_key], csharp_elements[csharp_key])

# Create DataFrame for better visualization
similarity_df_matrix = pd.DataFrame(similarity_matrix, index=uml_keys, columns=csharp_keys)

# Function to calculate precision, recall, and F-measure
def calculate_metrics(predicted, real):
    tp = len(predicted & real)
    fp = len(predicted - real)
    fn = len(real - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure

# Find the best threshold dynamically
best_threshold = 0
best_f_measure = 0
best_precision = 0
best_recall = 0
best_matches = set()

for threshold in np.arange(0.1, 1.01, 0.01):
    predicted_matches = set(
        (uml_keys[i], csharp_keys[j]) 
        for i in range(len(uml_keys)) 
        for j in range(len(csharp_keys)) 
        if similarity_matrix[i, j] >= threshold
    )
    
    precision, recall, f_measure = calculate_metrics(predicted_matches, real_matches)
    
    if f_measure > best_f_measure:
        best_f_measure = f_measure
        best_precision = precision
        best_recall = recall
        best_threshold = threshold
        best_matches = predicted_matches

# Output results
print("Jaccard Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold:.2f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall: {best_recall:.4f}")
print(f"F-Measure: {best_f_measure:.4f}")
print(f"\nPredicted Matches at Best Threshold:")
print(best_matches)
print("\nReal Matches:")
print(real_matches)
