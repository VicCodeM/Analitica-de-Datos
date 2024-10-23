import pandas as pd

#cargar datos
pacientes= pd.read_csv('Pacientes2.csv', engine= 'python', index_col=0)
#pacientes.head()


#on=bteniendi datos del datase
#pacientes.info()


#variables prefdictora
x= pacientes.iloc[:,1:11]

#variable a,predecir
y= pacientes.iloc[:,0]

x.head()

y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_traian, Y_test= train_test_split(x,y, train_size=0.80, random_state=0) 

X_train.info()

from sklearn.tree import DecisionTreeClassifier

arbol = DecisionTreeClassifier()

#contruytendo el arbol
arbol_enfermedad = arbol.fit(X_train,Y_traian)


#graficar nuesto arbol

from matplotlib import pyplot as plt
from sklearn import tree

fig = plt.figure(figsize=(25,20))
tree.plot_tree(arbol_enfermedad, feature_names=list (x.columns.values), class_names=list(y.values), filled=True)
plt.show()

#precir
y_pred= arbol_enfermedad.predict(X_test)
y_pred

from sklearn.metrics import confusion_matrix
matriz_confusion = confusion_matrix(Y_test,y_pred)
matriz_confusion

import numpy as np
preciciosn_global = np.sum(matriz_confusion.diagonal())/np.sum(matriz_confusion)
preciciosn_global

presicion_no= ((matriz_confusion[0,0])/sum(matriz_confusion[0,]))
presicion_no

presicion_si= ((matriz_confusion[1,1])/sum(matriz_confusion[1,]))
presicion_si