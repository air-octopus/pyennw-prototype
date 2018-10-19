import matplotlib.pyplot as plt

x = list(range(0, 10))
y1 = [x * x for x in x]
y2 = [81 - x * x for x in x]
y3 = [y1[x] + y2[x] for x in x]

fig = plt.figure()
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)

plt.show()
