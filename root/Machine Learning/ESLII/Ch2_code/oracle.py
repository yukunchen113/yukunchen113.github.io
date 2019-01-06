#the distribution for "Revealining the oracle" on pg. 16
"""
- 10 means are drawn from bivariate gaussian distribution:
	- centered at (1,0) for blue, stddev =1 (dist for means)
	- centered at (0,1) for orange, stddev =1 (dist for means)

- 100 samples from each class
- uniform probabilty to pick any of the means, non weighted. stddev=1/5



"""

import numpy as np 
import gaussian_mixture as gm
import matplotlib.pyplot as plt 

class oracle():
	def __init__(self, num_means = 10, ndim = 2):
		#make the oracle
		#this will create the oracle.

		#generate the means
		#2 dimensional mean, one for x, one for y.
		bmeans = np.random.normal([0,1],[1,1],[num_means, ndim]).reshape(num_means, ndim, 1)#blue
		omeans = np.random.normal([1,0],[1,1],[num_means, ndim]).reshape(num_means, ndim, 1)#orange

		stddev = np.ones(bmeans.shape)/5

		blue_gaussians = np.concatenate((bmeans, stddev), axis =-1)
		orange_gaussians = np.concatenate((omeans, stddev), axis =-1)
		mixture_instance = {}
		for color, gaussians in {"blue":blue_gaussians, "orange":orange_gaussians}.items():
			#create mixtures and sample from each
			mix = gm.mixture(num_gaussians=num_means)

			#create set_gaussian dictionary.
			gaussian_set = {(i,j):list(gaussians[i,j]) for i in range(num_means) for j in range(ndim)}

			mix.set_gaussian(gaussian_set)
			mixture_instance[color]=mix
		self.mixtures = mixture_instance
		self.gaussians = {"blue":blue_gaussians, "orange":orange_gaussians}

	def sample(self, num_samples = 100):
		#this will allow you to sample from your data
		#returns a dictionary, with key for the class, and the value being a matrix of X,Y points.
		mixtures = self.mixtures
		data = {}
		for i in ["blue", "orange"]:
			data[i] = mixtures[i].sample(num_samples)
		return data

	def plot(self, data=None):
		#give data in the form that sample give out.
		if data is None:
			data = self.sample()
		for k,v in data.items():
			plt.scatter(v[:,0], v[:,1], s =20, label= k)
		plt.legend()	
		plt.show()

if __name__ == "__main__":
	test = oracle()
	test.plot()

