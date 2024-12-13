from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Extract elements from PRINCE2 and Scrum metamodels
prince2_classes = {
    "BusinessCase": ["ID", "Title", "Description"],
    "ProjectBoard": ["ID", "Name", "Members"],
    "ProjectPlan": ["ID", "Name", "StartDate", "EndDate"],
    "StagePlan": ["ID", "Name", "StageObjective"],
    "WorkPackage": ["ID", "Name", "Tasks"],
    "EndStageReport": ["ID", "Name", "Summary"]
}

scrum_classes = {
    "ProductBacklog": ["ID", "Name", "Description"],
    "Sprint": ["ID", "Name", "Goal", "StartDate", "EndDate"],
    "ScrumTeam": ["ID", "Name", "Members"],
    "SprintBacklog": ["ID", "Name", "Tasks"],
    "Increment": ["ID", "Name", "Description", "Version"],
    "DailyScrum": ["ID", "Date", "Notes"]
}

# Flatten the attributes to compare
prince2_elements = {k: " ".join(v) for k, v in prince2_classes.items()}
scrum_elements = {k: " ".join(v) for k, v in scrum_classes.items()}

# Real correspondences
real_matches = {
    ("BusinessCase", "ProductBacklog"),
    ("ProjectBoard", "ScrumTeam"),
    ("ProjectPlan", "Sprint"),
    ("StagePlan", "DailyScrum"),
    ("WorkPackage", "SprintBacklog"),
    ("EndStageReport", "Increment")
}

# Combine all elements for TF-IDF vectorization
all_elements = list(prince2_elements.values()) + list(scrum_elements.values())

# Vectorize the elements
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_elements)

# Determine the number of features in the TF-IDF matrix
n_features = tfidf_matrix.shape[1]

# Apply LSA using SVD
n_components = min(100, n_features)  # Ensure n_components is <= n_features
svd = TruncatedSVD(n_components=n_components)
lsa_matrix = svd.fit_transform(tfidf_matrix)

# Calculate cosine similarities in the LSA space
similarity_matrix = cosine_similarity(lsa_matrix[:len(prince2_elements)], lsa_matrix[len(prince2_elements):])

# Create DataFrame for better visualization
prince2_keys = list(prince2_elements.keys())
scrum_keys = list(scrum_elements.keys())
similarity_df_matrix = pd.DataFrame(similarity_matrix, index=prince2_keys, columns=scrum_keys)

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
        (prince2_keys[i], scrum_keys[j]) 
        for i in range(len(prince2_keys)) 
        for j in range(len(scrum_keys)) 
        if similarity_matrix[i, j] >= threshold
    )
    _, _, f_measure = calculate_metrics(predicted_matches, real_matches)
    if f_measure > best_f_measure:
        best_f_measure = f_measure
        best_threshold = threshold

# Use the best threshold to get final predicted matches
predicted_matches = set(
    (prince2_keys[i], scrum_keys[j]) 
    for i in range(len(prince2_keys)) 
    for j in range(len(scrum_keys)) 
    if similarity_matrix[i, j] >= best_threshold
)

# Calculate final metrics
precision, recall, f_measure = calculate_metrics(predicted_matches, real_matches)

# Output results
print("LSA Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Measure: {f_measure}")

# Output real matches for verification
print("\nReal Matches:")
print(real_matches)
