import pandas as pd
import numpy as np
import Levenshtein as lev

# Fonction pour calculer la distance de Levenshtein entre deux chaînes
def levenshtein_distance(str1, str2):
    return lev.distance(str1, str2)

# Convertir les éléments en chaînes pour la comparaison
def convert_to_string(element):
    return ' '.join(element)

# Eléments des méta-modèles UML Activity et BPMN
uml_activity_elements = {
    "Activity": ["name", "description", "contains:Action", "contains:ControlFlow", "contains:Partition"],
    "Action": ["name", "description", "flows_to:Action", "flows_to:DecisionNode", "flows_to:MergeNode"],
    "ControlFlow": ["source", "target", "connects:Action", "connects:DecisionNode", "connects:MergeNode", "connects:ForkNode", "connects:JoinNode", "connects:InitialNode", "connects:FinalNode"],
    "DecisionNode": ["condition", "flows_to:Action", "flows_to:MergeNode"],
    "MergeNode": ["flows_to:Action"],
    "ForkNode": ["flows_to:Action"],
    "JoinNode": ["flows_to:Action"],
    "InitialNode": ["flows_to:Action"],
    "FinalNode": [],
    "Partition": ["name", "contains:Activity", "contains:Action"],
    "ObjectNode": ["name", "type", "used_by:Action"]
}

bpmn_elements = {
    "Process": ["name", "description", "contains:Task", "contains:SequenceFlow", "contains:Lane"],
    "Task": ["name", "description", "flows_to:Task", "flows_to:ExclusiveGateway", "flows_to:ParallelGateway"],
    "SequenceFlow": ["source", "target", "connects:Task", "connects:ExclusiveGateway", "connects:ParallelGateway", "connects:StartEvent", "connects:EndEvent"],
    "ExclusiveGateway": ["condition", "flows_to:Task", "flows_to:ParallelGateway"],
    "ParallelGateway": ["flows_to:Task"],
    "StartEvent": ["flows_to:Task"],
    "EndEvent": [],
    "Lane": ["name", "contains:Process", "contains:Task"],
    "DataObject": ["name", "type", "used_by:Task"]
}

# Correspondances réelles (Mappings fournies)
real_matches = {
    ("Activity", "Process"),
    ("Action", "Task"),
    ("ControlFlow", "SequenceFlow"),
    ("DecisionNode", "ExclusiveGateway"),
    ("MergeNode", "ExclusiveGateway"),
    ("ForkNode", "ParallelGateway"),
    ("JoinNode", "ParallelGateway"),
    ("InitialNode", "StartEvent"),
    ("FinalNode", "EndEvent"),
    ("Partition", "Lane"),
    ("ObjectNode", "DataObject")
}

# Calcul de la matrice de distance de Levenshtein
distance_matrix = []
for u_elem in uml_activity_elements:
    row = []
    for b_elem in bpmn_elements:
        distance = levenshtein_distance(convert_to_string(uml_activity_elements[u_elem]), convert_to_string(bpmn_elements[b_elem]))
        row.append(distance)
    distance_matrix.append(row)

# Conversion en DataFrame pour un affichage plus lisible
distance_df_matrix = pd.DataFrame(distance_matrix, 
                                  index=uml_activity_elements.keys(), 
                                  columns=bpmn_elements.keys())

# Normalisation pour obtenir des similarités (1 - distance normalisée)
max_distance = np.max(distance_df_matrix.values)
similarity_df_matrix = 1 - (distance_df_matrix / max_distance)

# Recherche du meilleur seuil
thresholds = np.arange(0.1, 1.0, 0.1)
best_threshold = 0
best_f_measure = 0
best_precision = 0
best_recall = 0

for threshold in thresholds:
    found_matches = set()
    for i, u_elem in enumerate(uml_activity_elements.keys()):
        for j, b_elem in enumerate(bpmn_elements.keys()):
            if similarity_df_matrix.iloc[i, j] >= threshold:
                found_matches.add((u_elem, b_elem))
    
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

# Affichage des résultats
print("Levenshtein Distance Matrix:")
print(distance_df_matrix)
print("\nSimilarity Matrix (normalized):")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {best_precision}")
print(f"Recall: {best_recall}")
print(f"F-Measure: {best_f_measure}")

# Affichage des correspondances réelles pour vérification
print("\nReal Matches:")
print(real_matches)
