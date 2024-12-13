import pandas as pd
import numpy as np

# Function to calculate Jaccard similarity between two sets
def jaccard_similarity(set1, set2):
    if not set1 and not set2:  # Both sets are empty
        return 1.0
    elif not set1 or not set2:  # One of the sets is empty
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Elements of PRINCE2 and Scrum meta-models
prince2_elements = {
    "BusinessCase": ["ID", "Title", "Description"],
    "ProjectBoard": ["ID", "Name", "Members"],
    "ProjectPlan": ["ID", "Name", "StartDate", "EndDate"],
    "StagePlan": ["ID", "Name", "StageObjective"],
    "WorkPackage": ["ID", "Name", "Tasks"],
    "EndStageReport": ["ID", "Name", "Summary"]
}

scrum_elements = {
    "ProductBacklog": ["ID", "Name", "Description"],
    "Sprint": ["ID", "Name", "Goal", "StartDate", "EndDate"],
    "ScrumTeam": ["ID", "Name", "Members"],
    "SprintBacklog": ["ID", "Name", "Tasks"],
    "Increment": ["ID", "Name", "Description", "Version"],
    "DailyScrum": ["ID", "Date", "Notes"]
}

# Real mappings provided
real_matches = {
    ("BusinessCase", "ProductBacklog"),
    ("ProjectBoard", "ScrumTeam"),
    ("ProjectPlan", "Sprint"),
    ("StagePlan", "SprintBacklog"),
    ("WorkPackage", "Increment"),
    ("EndStageReport", "DailyScrum")
}

# Compute the Jaccard similarity matrix
similarity_matrix = []
for p_elem in prince2_elements:
    row = []
    for s_elem in scrum_elements:
        similarity = jaccard_similarity(set(prince2_elements[p_elem]), set(scrum_elements[s_elem]))
        row.append(similarity)
    similarity_matrix.append(row)

# Convert to DataFrame for readability
similarity_df_matrix = pd.DataFrame(similarity_matrix, 
                                    index=prince2_elements.keys(), 
                                    columns=scrum_elements.keys())

# Find the best threshold using F-measure
thresholds = np.arange(0.1, 1.0, 0.1)
best_threshold = 0
best_f_measure = 0
best_precision = 0
best_recall = 0

for threshold in thresholds:
    found_matches = set()
    for i, p_elem in enumerate(prince2_elements.keys()):
        for j, s_elem in enumerate(scrum_elements.keys()):
            if similarity_df_matrix.iloc[i, j] >= threshold:
                found_matches.add((p_elem, s_elem))
    
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

# Display results
print("Jaccard Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {best_precision:.2f}")
print(f"Recall: {best_recall:.2f}")
print(f"F-Measure: {best_f_measure:.2f}")

# Display real matches for verification
print("\nReal Matches:")
print(real_matches)

# Display the matches found at the best threshold
print("\nMatches Found at Best Threshold:")
found_matches_at_best = set()
for i, p_elem in enumerate(prince2_elements.keys()):
    for j, s_elem in enumerate(scrum_elements.keys()):
        if similarity_df_matrix.iloc[i, j] >= best_threshold:
            found_matches_at_best.add((p_elem, s_elem))
print(found_matches_at_best)
