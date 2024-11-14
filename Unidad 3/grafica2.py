import matplotlib.pyplot as plt
import numpy as np

year = [str(year)for year in range(2010,2021)]
visitors = (1241,50927,162242,222093,665004,2071987,2460407,3799215,5399000,5474016,6003672)
plt.bar(year,visitors, color="blue")
plt.xlabel("Years")
plt.ylabel("values")
plt.title("bar chart exeple")
plt.plot()
plt.show()