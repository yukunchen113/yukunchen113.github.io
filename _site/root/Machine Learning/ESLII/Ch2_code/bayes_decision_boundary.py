"""
This is a program of the bayes decision boundary.

Here we have information of the underlying distributions that the
points are sampled from.

Since the distribution for the sample points are different every time,
the decision boundary is different as well.

Overall, the boundary is were there is equal probability (or likelihood)
that a point can come from either class. 

This is actually quite simple. For a given point, we just need to know
when the probatility of one mixture of gaussians equals another. It 
is also the same result if we equate the likelihoods.

the distributions that generate our sample data are normal 
distributions with variable mean and a standard deviation of 0.2
Since we can equate the likelihoods, we can eliminate the 
constant scaling that is common across all distributions,

In the end we are left with e^(-(x-mu)^2) where mu is the mean of 
various distributions. x is a point. Since we are dealing with a 
mixture of gaussians, then, it should be a summation of the 
likelihoods of each mixture along a certain dimension.





"""




import numpy as np 
import gaussian_mixture as gm
import math
import matplotlib.pyplot as plt 
#3d plot import
from mpl_toolkits.mplot3d import Axes3D

#transiton to plot.ly
import plotly as py 
import plotly.graph_objs as go 




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

plot2=[]
graph_colour = ["Blues", "Reds"]
k = 0
probabilities = []
for gaussians in [blue_gaussians, orange_gaussians]:
	#create mixtures and sample from each
	mix = gm.mixture(num_gaussians=num_means)

	#create set_gaussian dictionary.
	gaussian_set = {(i,j):list(gaussians[i,j]) for i in range(num_means) for j in range(ndim)}

	mix.set_gaussian(gaussian_set)
	data = mix.sample(num_samples)

	#plot sample data
	fig = plt.figure(0)
	plt.scatter(data[:,0], data[:,1], s =20)

	#plot gaussian surfaces.
	#construct normals
	norm_func = lambda x, mu, sig: 1/(sig*np.sqrt(2*math.pi))*np.exp(-(x-mu)**2/(2*sig**2))
	resolution = 100
	x = np.linspace(-3,3,resolution)
	y = np.linspace(-3,3,resolution)
	X, Y = np.meshgrid(x,y)
	comp = lambda xmu,ymu: norm_func(X,xmu,1/5)*norm_func(Y,ymu,1/5)
	Z=0
	for mu in gaussians[:,:,0]:
		Z=Z+comp(*mu)
	probabilities.append(Z)

	#plot
	plot2.append(go.Surface(z=Z, colorscale=graph_colour[k]))
	k+=1

#py.offline.plot(plot2)


#create boundary
boundary_threshold = 0.1
boundary = np.where(probabilities[0] < probabilities[1], 255, 0)
#plt.contour(X,Y,X+1, cmap=plt.cm.Paired)
fig.show()
input()