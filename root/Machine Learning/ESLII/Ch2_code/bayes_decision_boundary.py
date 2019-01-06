"""
This is a program of the bayes decision boundary for the oracle.

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


Here, I plot the boundary, and the distributions as well.
The 3D distribution graph is not perfect, as it is not continuous.


HOW TO RUN----
call in terminal
press enter to exit the program.

"""
import numpy as np 
import matplotlib.pyplot as plt 
import oracle as oc
import math
import plotly as py 
import plotly.graph_objs as go 


oracle = oc.oracle()
gaussians = oracle.gaussians

resolution = 100
x = np.linspace(-3,3,resolution)
y = np.linspace(-3,3,resolution)
X, Y = np.meshgrid(x,y)

norm_func = lambda x, mu, sig: 1/(sig*np.sqrt(2*math.pi))*np.exp(-(x-mu)**2/(2*sig**2))
comp = lambda xmu,ymu: norm_func(X,xmu,1/5)*norm_func(Y,ymu,1/5)

contour = 0
outputs = {}
i = 0
for col, gaussian in gaussians.items():
	Z = 0
	for mu in gaussian[:,:,0]:
		Z=Z+comp(*mu)
	outputs[col] = Z
	if i%2 == 0:
		contour +=Z 
	else:
		contour -=Z
	i+=1

#plot probability distributions.
plot=[]
graph_colour = ["Blues", "Reds"]
k=0
for Z in outputs.values():
	plot.append(go.Surface(x=X,y=Y,z=Z, colorscale=graph_colour[k]))
	k+=1
fig = go.Figure(data=plot)
py.offline.plot(fig)


#plot data and contour
fig, ax = plt.subplots()

data = oracle.sample()
for k,v in data.items():
	plt.scatter(v[:,0], v[:,1], s =20, label= k)

CS = ax.contour(X, Y, contour, levels=[0])
ax.set_title('Bayes Decision boundary')
fig.show()
input()