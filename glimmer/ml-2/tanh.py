import  matplotlib.pyplot as plt

import numpy as np

# frac{e^x-e^{-x}}{e^x+e^{-x}
x=np.linspace(-100,100,1000)
y=((np.e)**x-(np.e)**(-x))/((np.e)**x+(np.e)**(-x))
plt.plot(x,y)
plt.show()
