import math
import random
import numpy as np
import pandas as pd
import seaborn as sns

# Cargar el conjunto de datos
df_net = pd.read_csv('data.csv')

# Mostrar las primeras filas del conjunto de datos
df_net.head()

# Eliminar la columna 'id' si existe, ya que no es relevante para el análisis
if 'id' in df_net.columns:
    df_net.drop('id', axis=1, inplace=True)

# Mostrar las primeras filas para confirmar la eliminación de 'id'
df_net.head()

# Descripción del conjunto de datos
df_net.describe()

from sklearn.preprocessing import LabelEncoder
# Cambiar etiquetas a 1 y 0 (diagnosis)
le = LabelEncoder()
df_net['diagnosis'] = le.fit_transform(df_net['diagnosis'])

# Calcular la correlación
df_net.corr()

# Mostrar el mapa de calor de la correlación
sns.heatmap(df_net.corr())

# Separar las características (X) y la variable objetivo (y)
x = df_net.iloc[:, 1:].values  # Excluir la columna de diagnóstico para las características
y = df_net.iloc[:, 0].values   # El diagnóstico es la variable objetivo

from sklearn.model_selection import train_test_split
# Dividir el conjunto de datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Escalar los datos
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Entrenar el clasificador Naive Bayes Gaussiano
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predecir los resultados del conjunto de prueba
y_pred = classifier.predict(x_test)

# Hacer una predicción de ejemplo (ajustar los valores de entrada según sea necesario)
# Por ejemplo, usar valores ficticios para todas las características
ejemplo_prediccion = np.zeros(x_train.shape[1])
print(classifier.predict(sc.transform([ejemplo_prediccion])))

# Matriz de confusión
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)

# Reporte de clasificación
from sklearn.metrics import classification_report
print(f'Reporte de clasificación:\n{classification_report(y_test, y_pred)}')

# Comparar las predicciones con los valores reales
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Hacer otra predicción de ejemplo
otra_prediccion = np.zeros(x_train.shape[1])  # Ajustar los valores según las características del conjunto de datos
print(classifier.predict(sc.transform([otra_prediccion])))
