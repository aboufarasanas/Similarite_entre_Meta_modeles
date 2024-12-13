import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Conv1DTranspose, Reshape, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Chargez les données à partir du fichier Excel
df = pd.read_excel("combined_data.xlsx")

# Pré-traitez les données
le = LabelEncoder()
df['Métamodèle 1'] = le.fit_transform(df['Métamodèle 1'])
df['Élément 1'] = le.fit_transform(df['Élément 1'])
df['Métamodèle 2'] = le.fit_transform(df['Métamodèle 2'])
df['Élément 2'] = le.fit_transform(df['Élément 2'])

# Préparez les données d'entrée
X1 = df[['Métamodèle 1', 'Élément 1']].values
X2 = df[['Métamodèle 2', 'Élément 2']].values
y = df['Correspondance'].values

# Divisez les données
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

# Redimensionnez les données pour les réseaux siamois
X1_train = X1_train.reshape(-1, 2, 1)
X1_test = X1_test.reshape(-1, 2, 1)
X2_train = X2_train.reshape(-1, 2, 1)
X2_test = X2_test.reshape(-1, 2, 1)

# Définissez les réseaux siamois
def create_siamese_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=2, activation='relu')(input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    return x

input_shape = (2, 1)
base_network = create_siamese_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(lambda x: K.abs(x[0] - x[1]))([processed_a, processed_b])

model = Model(inputs=[input_a, input_b], outputs=distance)

# Compilez le modèle
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Entraînez le modèle
history = model.fit([X1_train, X2_train], y_train, validation_split=0.2, epochs=50, batch_size=32)

# Évaluez le modèle
loss, accuracy = model.evaluate([X1_test, X2_test], y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Faites des prédictions
y_pred = model.predict([X1_test, X2_test])
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculez des métriques supplémentaires
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

# Calculez la matrice de confusion
cm = confusion_matrix(y_test, y_pred_binary)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("Matrice de confusion:")
print(cm)