import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df_diamonds = sns.load_dataset('diamonds')
df_diamonds.head()
variabeles_independientes=['x','y']
x= df_diamonds[variabeles_independientes].values.reshape(-1,1)
y= df_diamonds['price'].values.reshape(-1,1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)

regressor.predict(y_test)

psc=regressor.predict([[0.21]])
print(psc)

sc=regressor.score(x_test,y_test)
print(sc)

#graficar
sns.set_theme()
fig, ax = plt.subplots(1,2, figsize=(10,10), sharey=True)
ax[0].scatter(x_train,y_train)
ax[0].plot(x_train, regressor.predict(x_train), c='g')
ax[0].set_title('Cart vs precie')
ax[0].set_xlabel('Cart')
ax[0].set_ylabel('Precie, USD')

ax[1].scatter(x_test,y_test)
ax[1].plot(x_test, regressor.predict(x_test), c='g')
ax[1].set_title('Cart vs precie')
ax[1].set_xlabel('Cart')
ax[1].set_ylabel('Precie, USD')
plt.suptitle('Lenar regression Model')
plt.show()