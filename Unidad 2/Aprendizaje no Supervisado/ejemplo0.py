import pandas as pd
from sklearn.datasets import fetch_california_housing

california_house = fetch_california_housing(as_frame=True)
features_of_interest = ['Longitude','Latitude','MedHouseVal']
home_data = california_house.frame[features_of_interest]
home_data.head()


#Creando un grafico de dispersion
import seaborn as sns
#sns.scatterplot(data = home_data,x="Longitude",y="Latitude",hue="MedHouseVal")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(home_data[['Latitude','Longitude']],home_data[['MedHouseVal']],test_size=0.33,random_state=0)

from sklearn import preprocessing

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

from sklearn import cluster

kmeans = cluster.KMeans(n_clusters=3,random_state=0,n_init='auto')
kmeans.fit(X_train_norm)

#sns.scatterplot(data = X_train,x="Longitude",y="Latitude",hue=kmeans.labels_)

from sklearn.metrics import silhouette_score

silhouette_score(X_train_norm,kmeans.labels_,metric="euclidean")

K = range(2,8)
fits = []
score = []

for k in K:
    model=cluster.KMeans(n_clusters=k,random_state=0,n_init='auto').fit(X_train_norm)
    
    fits.append(model)
    
    score.append(silhouette_score(X_train_norm,model.labels_,metric='euclidean'))
    score2 = silhouette_score(X_train_norm, model.labels_,metric='euclidean')
    print ("Cpon k= "+ str(k)+" el score es de:" + str(score2))

sns.lineplot(x=K,y=score)
sns.scatterplot(data = X_train,x="Longitude",y="Latitude",hue=fits[3].labels_)