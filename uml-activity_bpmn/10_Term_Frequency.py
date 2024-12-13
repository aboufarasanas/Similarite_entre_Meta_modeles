from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Définir les éléments des méta-modèles avec leurs attributs
uml_activity_elements = {
    "Activity": "name description contains Action ControlFlow Partition",
    "Action": "name description flows_to Action DecisionNode MergeNode",
    "ControlFlow": "source target connects Action DecisionNode MergeNode ForkNode JoinNode InitialNode FinalNode",
    "DecisionNode": "condition flows_to Action MergeNode",
    "MergeNode": "flows_to Action",
    "ForkNode": "flows_to Action",
    "JoinNode": "flows_to Action",
    "InitialNode": "flows_to Action",
    "FinalNode": "",
    "Partition": "name contains Activity Action",
    "ObjectNode": "name type used_by Action"
}

bpmn_elements = {
    "Process": "name description contains Task SequenceFlow Lane",
    "Task": "name description flows_to Task ExclusiveGateway ParallelGateway",
    "SequenceFlow": "source target connects Task ExclusiveGateway ParallelGateway StartEvent EndEvent",
    "ExclusiveGateway": "condition flows_to Task ParallelGateway",
    "ParallelGateway": "flows_to Task",
    "StartEvent": "flows_to Task",
    "EndEvent": "",
    "Lane": "name contains Process Task",
    "DataObject": "name type used_by Task"
}

# Combiner les textes des deux méta-modèles
all_elements = list(uml_activity_elements.values()) + list(bpmn_elements.values())

# Utiliser TfidfVectorizer avec use_idf=False pour ne calculer que le TF
tf_vectorizer = TfidfVectorizer(use_idf=False)
X = tf_vectorizer.fit_transform(all_elements)

# Calculer la similarité cosine entre les vecteurs TF
similarity_matrix = cosine_similarity(X[:len(uml_activity_elements)], X[len(uml_activity_elements):])

# Créer une DataFrame pour visualiser les résultats
uml_activity_keys = list(uml_activity_elements.keys())
bpmn_keys = list(bpmn_elements.keys())
similarity_df_matrix = pd.DataFrame(similarity_matrix, index=uml_activity_keys, columns=bpmn_keys)

# Fonction pour calculer les métriques de performance
def calculate_metrics(predicted, real):
    tp = len(predicted & real)
    fp = len(predicted - real)
    fn = len(real - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure

# Correspondances réelles
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

# Trouver le meilleur seuil dynamiquement
best_threshold = 0
best_f_measure = 0

for threshold in np.arange(0, 1.0, 0.01):
    predicted_matches = set(
        (uml_activity_keys[i], bpmn_keys[j]) 
        for i in range(len(uml_activity_keys)) 
        for j in range(len(bpmn_keys)) 
        if similarity_matrix[i, j] >= threshold
    )
    _, _, f_measure = calculate_metrics(predicted_matches, real_matches)
    if f_measure > best_f_measure:
        best_f_measure = f_measure
        best_threshold = threshold

# Utiliser le meilleur seuil pour obtenir les correspondances prédites finales
predicted_matches = set(
    (uml_activity_keys[i], bpmn_keys[j]) 
    for i in range(len(uml_activity_keys)) 
    for j in range(len(bpmn_keys)) 
    if similarity_matrix[i, j] >= best_threshold
)

# Calculer les métriques finales
precision, recall, f_measure = calculate_metrics(predicted_matches, real_matches)

# Afficher les résultats
print("Term Frequency (TF) Similarity Matrix:")
print(similarity_df_matrix)
print(f"\nBest Threshold: {best_threshold}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Measure: {f_measure}")

# Afficher les correspondances réelles pour vérification
print("\nReal Matches:")
print(real_matches)
