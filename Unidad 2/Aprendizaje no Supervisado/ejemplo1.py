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