import matplotlib.pyplot as plt
import numpy as np
x= np.arange(0,11)
y1= np.random.randint(2,7,(11,))
y2= np.random.randint(9,14,(11,))
y3 = np.random.randint(15,25,(11,))


plt.scatter(x,y1)
plt.scatter(x,y2, marker='v', color='r')
plt.scatter(x,y3, marker='^', color='m')
plt.title('Scatter Plot Example')
plt.show()