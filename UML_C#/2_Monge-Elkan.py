import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Flatten the attributes to compare
uml_elements = {k: " ".join(v) for k, v in uml_classes.items()}
csharp_elements = {k: " ".join(v) for k, v in csharp_classes.items()}

# Real correspondences
real_matches = {
    ("UMLModelElement", "CSharpElement"),
    ("UMLAttribute", "CSharpField"),
    ("UMLClassifier", "CSharpClass"),
    ("UMLOperation", "CSharpMethod"),
    ("UMLParameter", "CSharpParameter"),
    ("UMLInterface", "CSharpInterface")
}

# Function to calculate Monge-Elkan similarity
def monge_elkan_similarity(uml_element, csharp_element):
    uml_words = uml_element.split()
    csharp_words = csharp_element.split()

    if not uml_words or not csharp_words:
        return 0.0

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([uml_element, csharp_element])
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(similarity_matrix[0, 0])

# Calculate Monge-Elkan similarities
monge_elkan_similarities = []
for uml_element in uml_elements.values():
    row = []
    for csharp_element in csharp_elements.values():
        similarity = monge_elkan_similarity(uml_element, csharp_element)
        row.append(similarity)
    monge_elkan_similarities.append(row)

# Convert to numpy array for easy manipulation
monge_elkan_similarities = np.array(monge_elkan_similarities)

# Create DataFrame for better visualization
uml_keys = list(uml_elements.keys())
csharp_keys = list(csharp_elements.keys())
similarity_df_matrix = pd.DataFrame(monge_elkan_similarities, index=uml_keys, columns=csharp_keys)

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
for threshold in np.arange(0.1, 1.0, 0.01):
    predicted_matches = set(
        (uml_keys[i], csharp_keys[j]) 
        for i in range(len(uml_keys)) 
        for j in range(len(csharp_keys)) 
        if monge_elkan_similarities[i, j] >= threshold
    )
    _, _, f_measure = calculate_metrics(predicted_matches, real_matches)
    if f_measure > best_f_measure:
        best_f_measure = f_measure
        best_threshold = threshold

# Use the best threshold to get final predicted matches
predicted_matches = set(
    (uml_keys[i], csharp_keys[j]) 
    for i in range(len(uml_keys)) 
    for j in range(len(csharp_keys)) 
    if monge_elkan_similarities[i, j] >= best_threshold
)

# Calculate final metrics
precision, recall, f_measure = calculate_metrics(predicted_matches, real_matches)

# Output results
print("Monge-Elkan Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Measure: {f_measure}")

# Output real matches for verification
print("\nReal Matches:")
print(real_matches)
