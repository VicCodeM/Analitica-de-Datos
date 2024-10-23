import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset correctamente con el separador ';'
df_net = pd.read_csv('Ojos1.csv')
df_net.columns = ['Ocupación', 'Color de Ojos', 'País', 'Género']

# Limpiar espacios en blanco adicionales
df_net = df_net.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Mostrar los primeros registros para verificar la carga correcta
print(df_net.head())

# Codificar las variables categóricas a valores numéricos
from sklearn.preprocessing import LabelEncoder
le_ocupacion = LabelEncoder()
le_color_ojos = LabelEncoder()
le_pais = LabelEncoder()
le_genero = LabelEncoder()

df_net['Ocupación'] = le_ocupacion.fit_transform(df_net['Ocupación'])
df_net['Color de Ojos'] = le_color_ojos.fit_transform(df_net['Color de Ojos'])
df_net['País'] = le_pais.fit_transform(df_net['País'])
df_net['Género'] = le_genero.fit_transform(df_net['Género'])

# Mostrar el dataset después de la codificación
print(df_net.head())

# Calcular y mostrar la matriz de correlación
correlation_matrix = df_net.corr()
print("Matriz de correlación:\n", correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
plt.title("Matriz de correlación entre las variables")
plt.show()

# Separar en características (X) y variable objetivo (y), usando el género como variable objetivo
x = df_net.iloc[:, :-1].values
y = df_net.iloc[:, -1].values

# Dividir el dataset en conjunto de entrenamiento y de prueba
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=True)

# Escalar el dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Entrenar el clasificador para predecir el género
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Realizar predicciones
y_pred = classifier.predict(x_test)

# Ejemplo de predicción con datos de entrada
print("Predicción para entrada [30, 2, 5]:", classifier.predict(sc.transform([[30, 2, 5]]))) # Ejemplo de ocupación 30, color de ojos 2, país 5

# Matriz de confusión
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Matriz de confusión")
plt.show()

# Reporte de clasificación
from sklearn.metrics import classification_report
print(f'Reporte de clasificación:\n{classification_report(y_test, y_pred)}')

# Mostrar la comparación entre las predicciones y los valores reales
comparison = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
print("Comparación entre las predicciones y los valores reales:\n", comparison)
