from pyjarowinkler import distance
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

# Calculate Jaro-Winkler similarities
def jaro_winkler_similarity(str1, str2):
    return distance.get_jaro_distance(str1, str2, winkler=True)

jaro_winkler_matrix = np.zeros((len(prince2_elements), len(scrum_elements)))
prince2_keys = list(prince2_elements.keys())
scrum_keys = list(scrum_elements.keys())

for i, p_key in enumerate(prince2_keys):
    for j, s_key in enumerate(scrum_keys):
        jaro_winkler_matrix[i, j] = jaro_winkler_similarity(prince2_elements[p_key], scrum_elements[s_key])

# Create DataFrame for better visualization
similarity_df_matrix = pd.DataFrame(jaro_winkler_matrix, index=prince2_keys, columns=scrum_keys)

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
        if jaro_winkler_matrix[i, j] >= threshold
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
    if jaro_winkler_matrix[i, j] >= best_threshold
)

# Calculate final metrics
precision, recall, f_measure = calculate_metrics(predicted_matches, real_matches)

# Output results
print("Jaro-Winkler Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Measure: {f_measure}")

# Output real matches for verification
print("\nReal Matches:")
print(real_matches)
