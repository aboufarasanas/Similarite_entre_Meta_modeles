import numpy as np
import pandas as pd

def jaccard_similarity(set1, set2):
    intersection = len(set(set1).intersection(set2))
    union = len(set(set1).union(set2))
    return intersection / union if union != 0 else 0

def calculate_jaccard_matrix(uml_elements, bpmn_elements):
    uml_names = [element['name'] for element in uml_elements]
    bpmn_names = [element['name'] for element in bpmn_elements]
    
    uml_attributes = {element['name']: element['attributes'] for element in uml_elements}
    bpmn_attributes = {element['name']: element['attributes'] for element in bpmn_elements}
    
    uml_relationships = {element['name']: set(sum(element['relationships'].values(), [])) for element in uml_elements}
    bpmn_relationships = {element['name']: set(sum(element['relationships'].values(), [])) for element in bpmn_elements}
    
    matrix = pd.DataFrame(np.zeros((len(uml_names), len(bpmn_names))), index=uml_names, columns=bpmn_names)
    
    for uml_name in uml_names:
        for bpmn_name in bpmn_names:
            attr_similarity = jaccard_similarity(uml_attributes[uml_name], bpmn_attributes[bpmn_name])
            rel_similarity = jaccard_similarity(uml_relationships[uml_name], bpmn_relationships[bpmn_name])
            matrix.loc[uml_name, bpmn_name] = (attr_similarity + rel_similarity) / 2
            
    return matrix

uml_elements = [
    {"name": "Activity", "attributes": ["name", "description"], "relationships": {"contains": ["Action", "ControlFlow", "Partition"]}},
    {"name": "Action", "attributes": ["name", "description"], "relationships": {"flows_to": ["Action", "DecisionNode", "MergeNode"]}},
    {"name": "ControlFlow", "attributes": ["source", "target"], "relationships": {"connects": ["Action", "DecisionNode", "MergeNode", "ForkNode", "JoinNode", "InitialNode", "FinalNode"]}},
    {"name": "DecisionNode", "attributes": ["condition"], "relationships": {"flows_to": ["Action", "MergeNode"]}},
    {"name": "MergeNode", "attributes": [], "relationships": {"flows_to": ["Action"]}},
    {"name": "ForkNode", "attributes": [], "relationships": {"flows_to": ["Action"]}},
    {"name": "JoinNode", "attributes": [], "relationships": {"flows_to": ["Action"]}},
    {"name": "InitialNode", "attributes": [], "relationships": {"flows_to": ["Action"]}},
    {"name": "FinalNode", "attributes": [], "relationships": {}},
    {"name": "Partition", "attributes": ["name"], "relationships": {"contains": ["Activity", "Action"]}},
    {"name": "ObjectNode", "attributes": ["name", "type"], "relationships": {"used_by": ["Action"]}}
]

bpmn_elements = [
    {"name": "Process", "attributes": ["name", "description"], "relationships": {"contains": ["Task", "SequenceFlow", "Lane"]}},
    {"name": "Task", "attributes": ["name", "description"], "relationships": {"flows_to": ["Task", "ExclusiveGateway", "ParallelGateway"]}},
    {"name": "SequenceFlow", "attributes": ["source", "target"], "relationships": {"connects": ["Task", "ExclusiveGateway", "ParallelGateway", "StartEvent", "EndEvent"]}},
    {"name": "ExclusiveGateway", "attributes": ["condition"], "relationships": {"flows_to": ["Task", "ParallelGateway"]}},
    {"name": "ParallelGateway", "attributes": [], "relationships": {"flows_to": ["Task"]}},
    {"name": "StartEvent", "attributes": [], "relationships": {"flows_to": ["Task"]}},
    {"name": "EndEvent", "attributes": [], "relationships": {}},
    {"name": "Lane", "attributes": ["name"], "relationships": {"contains": ["Process", "Task"]}},
    {"name": "DataObject", "attributes": ["name", "type"], "relationships": {"used_by": ["Task"]}}
]

# Calculate Jaccard similarity matrix
jaccard_matrix = calculate_jaccard_matrix(uml_elements, bpmn_elements)
print("Jaccard Similarity Matrix:")
print(jaccard_matrix)

# Find best threshold
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
    precision, recall, f_measure = calculate_metrics(jaccard_matrix, real_matches, threshold)
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
