import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from spektral.data import BatchLoader
from spektral.datasets import TUDataset
from spektral.layers import GlobalAttentionPool, GlobalAvgPool
from spektral.models import GeneralizedFilter

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

# Convertissez les données en graphes
A_train, X_train, E_train, y_train = ..., ..., ..., ...
A_test, X_test, E_test, y_test = ..., ..., ..., ...

# Définissez les CNN pour les graphes
def create_cnn_graph(input_shape):
    x_in = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x_in)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    return x

input_shape = (None, 2)
cnn_graph = create_cnn_graph(input_shape)

model = Model(inputs=Input(shape=input_shape), outputs=cnn_graph)

# Compilez le modèle
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Entraînez le modèle
batch_size = 32
loader_tr = BatchLoader(A_train, X_train, E_train, y_train, batch_size=batch_size, mask=True)
loader_va = BatchLoader(A_test, X_test, E_test, y_test, batch_size=batch_size, mask=True)

history = model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, epochs=50, 
                    validation_data=loader_va.load(), validation_steps=loader_va.steps_per_epoch)

# Évaluez le modèle
loss, accuracy = model.evaluate(loader_va.load(), steps=loader_va.steps_per_epoch)
print(f"Test accuracy: {accuracy:.4f}")

# Faites des prédictions
y_pred = model.predict(loader_va.load())
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