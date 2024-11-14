import matplotlib.pyplot as plt
import numpy as np
gussian_numbers = np.random.normal(size=10000)
gussian_numbers
plt.hist(gussian_numbers,bins=20)
plt.title("Gussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()