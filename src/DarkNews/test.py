import matplotlib.pyplot as plt
import const
import numpy as np


x = np.linspace(0,1,100)
#plt.plot(x, const.FWEAKcoh(x, 40))
y = np.vectorize(const.Fpauli_blocking)(x, 40.)
plt.plot(x, y)
plt.show()
