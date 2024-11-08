import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.cm as cm
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

X,y= load_iris(return_X_y=True)

# iris = datasets.load_iris()
# X= iris.data
# y=iris.target
# names = iris.feature_names
# X,y = shuffle(X,y,random_state=42)

sse= [] #s
for k in range(1,11):
    km= KMeans(n_clusters=k, random_state=2)
    km.fit(X)
    sse.append(km.inertia_)
    
sns.set_style("whitegrid")
g=sns.lineplot(x=range(1,11),y=sse)

g.set(xlabel = "Number of clouser (k)", ylabel = "Sum Squared Error", title = "Elbow Method")    
plt.show()

kmmeans= KMeans(n_clusters=3,random_state=2)
kmmeans.fit(X)
kmmeans.cluster_centers_

pred = kmmeans.fit_predict(X)
pred

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],c=pred,cmap=cm.Accent)
plt.grid(True)
for center in kmmeans.cluster_centers_:
    center=center[:2]
    plt.scatter(center[0],center[1], marker='^', c='red')
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")

plt.subplot(1,2,2)
plt.scatter(X[:,2],X[:,3],c=pred,cmap=cm.Accent)
plt.grid(True)
for center in kmmeans.cluster_centers_:
    center=center[2:4]
    plt.scatter(center[0],center[1], marker='^', c='red')
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.show()