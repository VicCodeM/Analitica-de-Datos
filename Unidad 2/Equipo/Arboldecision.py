# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv('salesproject.csv', engine='python', index_col=None)


# Selección de las variables predictoras y la variable a predecir
# Usaremos "Unit price", "Quantity", "Total", "gross income" como X y "Customer type" como Y
x = df[['Unit price', 'Quantity', 'Total', 'gross income']]
y = df['Customer type']

# División del conjunto de datos en entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.80, random_state=0)

# Construcción y entrenamiento del modelo de árbol de decisión
arbol = DecisionTreeClassifier(random_state=0)
arbol_model = arbol.fit(X_train, Y_train)

# Graficar el árbol de decisión
fig = plt.figure(figsize=(25, 20))
tree.plot_tree(arbol_model, feature_names=list(x.columns.values), class_names=y.unique(), filled=True)
plt.show()

# Predecir los valores del conjunto de prueba
y_pred = arbol_model.predict(X_test)

# Calcular la matriz de confusión
matriz_confusion = confusion_matrix(Y_test, y_pred)

# Calcular la precisión global
precision_global = np.sum(matriz_confusion.diagonal()) / np.sum(matriz_confusion)

# Calcular la precisión por clase
# Para cada clase en la matriz de confusión
precision_por_clase = matriz_confusion.diagonal() / matriz_confusion.sum(axis=1)

# Resultados
matriz_confusion, precision_global, precision_por_clase
