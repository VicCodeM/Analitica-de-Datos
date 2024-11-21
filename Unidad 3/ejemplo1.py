import matplotlib.pyplot as plt

idex= [1,2,3,4,5,6,7,8,9]
y1= [23,42,33,43,8,44,43,18,21]
y2= [9,31,25,14,17,17,42,22,28]
y3 = [18,29,19,22,18,16,13,32,21]


plt.stackplot(idex,y1,y2,y3)
plt.title('Stack plot Exemple')
plt.show()