import matplotlib.pyplot as plt
labels = 'C','Python','Java','C++','C#'
sizes = [13.38,11.87,11.74,7.81,4.41]
explode= (0,.1,0,0,0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("TIOBE Index for may 2021")
plt.show()