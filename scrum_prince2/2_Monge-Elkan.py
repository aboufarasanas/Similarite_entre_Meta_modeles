from itertools import product
import numpy as np
import pandas as pd
import jellyfish

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

# Define Monge-Elkan similarity function
def monge_elkan_similarity(set1, set2):
    def best_match(word, words):
        return max(jellyfish.jaro_winkler_similarity(word, w) for w in words)
    
    return np.mean([best_match(word, set2) for word in set1])

# Create similarity matrix
prince2_keys = list(prince2_classes.keys())
scrum_keys = list(scrum_classes.keys())

similarity_matrix = np.zeros((len(prince2_keys), len(scrum_keys)))

for i, p_class in enumerate(prince2_keys):
    for j, s_class in enumerate(scrum_keys):
        similarity_matrix[i, j] = monge_elkan_similarity(prince2_classes[p_class], scrum_classes[s_class])

# Convert to DataFrame for visualization
similarity_df_matrix = pd.DataFrame(similarity_matrix, index=prince2_keys, columns=scrum_keys)

# Real correspondences
real_matches = {
    ("BusinessCase", "ProductBacklog"),
    ("ProjectBoard", "ScrumTeam"),
    ("ProjectPlan", "Sprint"),
    ("StagePlan", "SprintBacklog"),
    ("WorkPackage", "SprintBacklog"),
    ("EndStageReport", "Increment")
}

# Function to calculate metrics for a given threshold
def calculate_metrics(threshold):
    predicted_matches = set(
        (prince2_keys[i], scrum_keys[j])
        for i in range(len(prince2_keys))
        for j in range(len(scrum_keys))
        if similarity_matrix[i, j] >= threshold
    )
    
    tp = len(predicted_matches & real_matches)
    fp = len(predicted_matches - real_matches)
    fn = len(real_matches - predicted_matches)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure, predicted_matches

# Find the best threshold
best_threshold = None
best_precision = best_recall = best_f_measure = 0
best_matches = set()

thresholds = np.arange(0.1, 1.0, 0.1)

for threshold in thresholds:
    precision, recall, f_measure, predicted_matches = calculate_metrics(threshold)
    if f_measure > best_f_measure:
        best_f_measure = f_measure
        best_precision = precision
        best_recall = recall
        best_threshold = threshold
        best_matches = predicted_matches

# Output results
print("Monge-Elkan Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold:.1f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall: {best_recall:.4f}")
print(f"F-Measure: {best_f_measure:.4f}")
print(f"\nMatches Found at Best Threshold ({best_threshold:.1f}):")
print(best_matches)
print("\nReal Matches:")
print(real_matches)
