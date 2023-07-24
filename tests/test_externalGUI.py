import numpy as np
import sys
import matplotlib.pyplot as plt
import pyphoplacecellanalysis.External.matplotlib.mplAdvancedOptions

if __name__ == "__main__":
	x = np.linspace(0,4*np.pi,10000)
	y = np.sin(x)
	y1 = 40000*x;
	y2 = np.sin(2*x)

	plt.figure(1)
	plt.plot(x,y)
	plt.plot(x,y1)
	plt.plot(x,y2)
	plt.show()

