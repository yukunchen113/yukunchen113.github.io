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
#generate the means 
num_samples = 100
num_means = 10
ndim = 2
#2 dimensional mean, one for x, one for y.
bmeans = np.random.normal([0,1],[1,1],[num_means, ndim]).reshape(num_means, ndim, 1)#blue
omeans = np.random.normal([1,0],[1,1],[num_means, ndim]).reshape(num_means, ndim, 1)#orange

stddev = np.ones(bmeans.shape)/5

blue_gaussians = np.concatenate((bmeans, stddev), axis =-1)
orange_gaussians = np.concatenate((omeans, stddev), axis =-1)

for gaussians in [blue_gaussians, orange_gaussians]:
	#create mixtures and sample from each
	mix = gm.mixture(num_gaussians=num_means)

	#create set_gaussian dictionary.
	gaussian_set = {(i,j):list(gaussians[i,j]) for i in range(num_means) for j in range(ndim)}

	mix.set_gaussian(gaussian_set)
	data = mix.sample(num_samples)

	plt.scatter(data[:,0], data[:,1], s =20)
plt.show()