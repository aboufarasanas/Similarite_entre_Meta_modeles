import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
uml_elements = [
    {"name": "Activity", "description": "A high-level task or process"},
    {"name": "Action", "description": "A specific task or step in the process"},
    {"name": "ControlFlow", "description": "Flow that controls the process"},
    {"name": "DecisionNode", "description": "A point where decisions are made"},
    {"name": "MergeNode", "description": "Merges different paths"},
    {"name": "ForkNode", "description": "Splits a process into multiple paths"},
    {"name": "JoinNode", "description": "Joins multiple paths into one"},
    {"name": "InitialNode", "description": "Starting point of the process"},
    {"name": "FinalNode", "description": "End point of the process"},
    {"name": "Partition", "description": "Group of activities"},
    {"name": "ObjectNode", "description": "Object used by actions"}
]

bpmn_elements = [
    {"name": "Process", "description": "A sequence of tasks and events"},
    {"name": "Task", "description": "An individual work item or step"},
    {"name": "SequenceFlow", "description": "Flow between tasks"},
    {"name": "ExclusiveGateway", "description": "Decision point in the process"},
    {"name": "ParallelGateway", "description": "Forks and joins multiple paths"},
    {"name": "StartEvent", "description": "Starting point of the process"},
    {"name": "EndEvent", "description": "End point of the process"},
    {"name": "Lane", "description": "Categorizes process participants"},
    {"name": "DataObject", "description": "Data used or produced by tasks"}
]

# Convert descriptions to text data for LSA
uml_descriptions = [element['description'] for element in uml_elements]
bpmn_descriptions = [element['description'] for element in bpmn_elements]

# Combine all descriptions for vectorization
all_descriptions = uml_descriptions + bpmn_descriptions

# Vectorize descriptions using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(all_descriptions)

# Apply LSA (SVD) to reduce dimensions
lsa = TruncatedSVD(n_components=2)  # Number of topics
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# Split LSA matrix back into UML and BPMN matrices
uml_lsa_matrix = lsa_matrix[:len(uml_descriptions)]
bpmn_lsa_matrix = lsa_matrix[len(uml_descriptions):]

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(uml_lsa_matrix, bpmn_lsa_matrix)

# Create a DataFrame for better visualization
uml_names = [element['name'] for element in uml_elements]
bpmn_names = [element['name'] for element in bpmn_elements]
lsa_similarity_df = pd.DataFrame(similarity_matrix, index=uml_names, columns=bpmn_names)

print("LSA Similarity Matrix:")
print(lsa_similarity_df)

# Find the best threshold using a similar approach as before
thresholds = np.arange(0.0, 1.0, 0.01)
real_matches = {
    ("InitialNode", "StartEvent"), ("Partition", "Lane"), ("MergeNode", "ExclusiveGateway"),
    ("Activity", "Process"), ("JoinNode", "ParallelGateway"), ("DecisionNode", "ExclusiveGateway"),
    ("ObjectNode", "DataObject"), ("Action", "Task"), ("ForkNode", "ParallelGateway"),
    ("ControlFlow", "SequenceFlow"), ("FinalNode", "EndEvent")
}

def calculate_metrics(matrix, real_matches, threshold):
    matches = set()
    for uml_element in matrix.index:
        for bpmn_element in matrix.columns:
            if matrix.loc[uml_element, bpmn_element] >= threshold:
                matches.add((uml_element, bpmn_element))
    
    true_positives = len(matches.intersection(real_matches))
    false_positives = len(matches - real_matches)
    false_negatives = len(real_matches - matches)
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f_measure

best_threshold = 0
best_f_measure = 0
best_precision = 0
best_recall = 0

for threshold in thresholds:
    precision, recall, f_measure = calculate_metrics(lsa_similarity_df, real_matches, threshold)
    if f_measure > best_f_measure:
        best_threshold = threshold
        best_f_measure = f_measure
        best_precision = precision
        best_recall = recall

print(f"Best Threshold: {best_threshold}")
print(f"Precision: {best_precision}")
print(f"Recall: {best_recall}")
print(f"F-Measure: {best_f_measure}")

print("Real Matches:")
print(real_matches)
