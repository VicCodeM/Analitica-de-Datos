import math 
import random
import numpy as np
import pandas as pd
import seaborn as sns

#cargar el dtaset
df_net = pd.read_csv('Social_Network_Ads.csv')
df_net.head()

#Eliminar user ID
df_net.drop('User ID', axis=1, inplace=True)
df_net.head()

df_net.describe()


from sklearn.preprocessing import LabelEncoder
#cambiar etiquetas  a 1 y 0 
le = LabelEncoder()
df_net['Gender'] = le.fit_transform(df_net['Gender'])
df_net

# #correlacion
df_net.corr()

sns.heatmap(df_net.corr())
#Eiminar genero
df_net.drop(columns='Gender', inplace=True)

x=df_net.iloc[:,:-1].values
y=df_net.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


#Sacle dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(classifier.predict(sc.transform([[30,87000]])))

#Matriz de confusion
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True, fmt='d',cmap='Blues', cbar=False)


#Reporte de clasificacion
from sklearn.metrics import classification_report
print(f'clasificacion reprote: \n{classification_report(y_test, y_pred)})')
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print(classifier.predict(sc.transform([[22,15000]])))





