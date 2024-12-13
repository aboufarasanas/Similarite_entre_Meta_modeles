import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

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

# Convert attribute lists to strings for LSA
def convert_to_text(class_dict):
    return {k: ' '.join(v) if v else '' for k, v in class_dict.items()}

uml_texts = convert_to_text(uml_classes)
csharp_texts = convert_to_text(csharp_classes)

# Combine texts for vectorization
texts = list(uml_texts.values()) + list(csharp_texts.values())

# Create TF-IDF vectorizer and LSA model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
lsa = TruncatedSVD(n_components=2)  # Number of components can be adjusted
X_lsa = lsa.fit_transform(X)

# Compute cosine similarity matrix
n_uml = len(uml_texts)
n_csharp = len(csharp_texts)

# Extract LSA vectors for UML and C#
lsa_uml = X_lsa[:n_uml]
lsa_csharp = X_lsa[n_uml:]

# Calculate cosine similarity
similarity_matrix = cosine_similarity(lsa_uml, lsa_csharp)

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
print("LSA Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold:.1f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall: {best_recall:.4f}")
print(f"F-Measure: {best_f_measure:.4f}")
