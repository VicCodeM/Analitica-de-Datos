import pandas as pd

# Cargar datos
df = pd.read_csv('iris.csv', engine='python', index_col=None)

# Obteniendo información del dataset
df.info()

# Variables predictoras (las primeras cuatro columnas)
x = df.iloc[:, :4]

# Variable a predecir (la última columna)
y = df.iloc[:, 4]

# Mostrar los primeros registros de X e Y
x.head()
y.head()

# Dividir el dataset en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.80, random_state=0)

# Información del conjunto de entrenamiento
X_train.info()

# Construir el árbol de decisión
from sklearn.tree import DecisionTreeClassifier
arbol = DecisionTreeClassifier()

# Entrenar el modelo
arbol_flores = arbol.fit(X_train, Y_train)

# Graficar el árbol de decisión
from matplotlib import pyplot as plt
from sklearn import tree

fig = plt.figure(figsize=(25, 20))
tree.plot_tree(arbol_flores, feature_names=list(x.columns.values), class_names=list(y.unique()), filled=True)
plt.show()

# Predecir los valores del conjunto de prueba
y_pred = arbol_flores.predict(X_test)
y_pred

# Calcular la matriz de confusión
from sklearn.metrics import confusion_matrix
matriz_confusion = confusion_matrix(Y_test, y_pred)
matriz_confusion

# Calcular la precisión global
import numpy as np
precicion_global = np.sum(matriz_confusion.diagonal()) / np.sum(matriz_confusion)
print(precicion_global)

# Calcular la precisión por clase
presicion_setosa = matriz_confusion[0, 0] / sum(matriz_confusion[0,])
print(presicion_setosa)

presicion_versicolor = matriz_confusion[1, 1] / sum(matriz_confusion[1,])
presicion_versicolor
print(presicion_versicolor)

presicion_virginica = matriz_confusion[2, 2] / sum(matriz_confusion[2,])
print(presicion_virginica)
