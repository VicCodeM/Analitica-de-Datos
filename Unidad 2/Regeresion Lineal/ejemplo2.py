import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Advertising.csv')
future_col=['TV','Radio','Newspaper']
x=data[future_col]
y=data['Sales']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

valores_prediccion = [0000,000,5000]

pr= regressor.predict([valores_prediccion])
print(int(pr))

sc=regressor.score(x_test,y_test)
print(sc)


