import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df_diamonds = sns.load_dataset('diamonds')
df_diamonds.head()

x= df_diamonds['carat'].values.reshape(-1,1)
y= df_diamonds['price'].values.reshape(-1,1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)

regressor.predict(y_test)

psc=regressor.predict([[0.21]])
print(psc)

sc=regressor.score(x_test,y_test)
print(sc)

