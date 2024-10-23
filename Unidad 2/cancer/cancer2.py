import math
import random
import numpy as np
import pandas as pd
import seaborn as sns

# Cargar el conjunto de datos
df_net = pd.read_csv('data.csv')

df_net.head()

# Eliminar la columna 'id'
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
x = df_net.iloc[:, 1:].values 
y = df_net.iloc[:, 0].values   

from sklearn.model_selection import train_test_split
# Dividir el conjunto de datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)



#Sacle dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

#print(classifier.predict(sc.transform([[30,87000]])))

#Matriz de confusion
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True, fmt='d',cmap='Blues', cbar=False)


#Reporte de clasificacion
from sklearn.metrics import classification_report
print(f'clasificacion reprote: \n{classification_report(y_test, y_pred)}')
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#print(classifier.predict(sc.transform([[22,15000]])))