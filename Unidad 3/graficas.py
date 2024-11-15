import matplotlib.pyplot as plt
plt.plot([-1,-4.5,16,23])
plt.show()

plt.plot([-1,-4.5,16,23],"ob")
plt.show()

days= range(1,9)
celsius_values = [25.6,24.1,26.7,28.3,27.5,30.5,32.8,33.1]
fig, ax= plt.subplots()
ax.plot(days,celsius_values)
ax.set(xlabel='Day',ylabel = 'Temeperature in Celcuis', title='Temperature Graph')
plt.show()


days= list(range(1,9))
celsius_min = [19.6,24.1,26.7,28.3,27.5,30.5,32.8,33.1]
celsius_max = [24.8,28.9,31.3,33.0,34.9,35.6,38.4,39.2]
fig, ax= plt.subplots()

ax.set(xlabel='Day',ylabel = 'Temeperature in Celcuis', title='Temperature Graph')
ax.plot((days,celsius_min,days,
         celsius_min,'oy',
         days,celsius_max,
         days,celsius_max,'or'))
plt.show()


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

import matplotlib.pyplot as plt
import numpy as np
gussian_numbers = np.random.normal(size=10000)
gussian_numbers
plt.hist(gussian_numbers,bins=20)
plt.title("Gussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()