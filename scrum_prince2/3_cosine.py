from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# PRINCE2 and Scrum elements
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

# Flatten attributes into strings
prince2_elements = {k: " ".join(v) for k, v in prince2_classes.items()}
scrum_elements = {k: " ".join(v) for k, v in scrum_classes.items()}

# Real matches provided
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

# Compute TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_elements)

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix[:len(prince2_elements)], tfidf_matrix[len(prince2_elements):])

# Create DataFrame for similarity matrix
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

# Find the best threshold
best_threshold = 0
best_f_measure = 0
best_precision = 0
best_recall = 0
best_matches = set()

for threshold in np.arange(0.1, 1.0, 0.01):
    predicted_matches = set(
        (prince2_keys[i], scrum_keys[j])
        for i in range(len(prince2_keys))
        for j in range(len(scrum_keys))
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
print("Cosine Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold:.2f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall: {best_recall:.4f}")
print(f"F-Measure: {best_f_measure:.4f}")
print(f"\nPredicted Matches at Best Threshold:")
print(best_matches)
print("\nReal Matches:")
print(real_matches)
