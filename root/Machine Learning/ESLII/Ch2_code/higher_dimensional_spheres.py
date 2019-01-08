"""
for creating graph for higher dimensional bounding sheres.
using 2D cross section for viewing the diameter on these spheres.
I got help plotting this from:
https://stackoverflow.com/questions/36060933/matplotlib-plot-a-plane-and-points-in-3d-simultaneously


Has one graph for cutting the sphere,
and another graph for what the cross section would be.
"""
import numpy as np 
import matplotlib.pyplot as plt 
import oracle as oc
import math

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
msg = ["""

	Here we can see the 3D version and the 2D version. 

	The 2D version with circles is also equivalent to the cross section
	defined by the plane in the 3D version.


	(Press ENTER to continue...)""",
	"""

	This is the new cross section that more accuately shows the diameter,
	and how the the center sphere is getting closer to the edges.

	(Press ENTER to continue...)"""]



for msg_frame_num in range(2):
	fig = plt.figure(0)
	plt.clf()
	ax = fig.gca(projection='3d')
	ax.set_aspect("equal")

	# draw cube
	r = [-2, 2]
	for s, e in combinations(np.array(list(product(r, r, r))), 2):
	    if np.sum(np.abs(s-e)) == r[1]-r[0]:
	        ax.plot3D(*zip(s, e), color="b")

	#draw 4 spheres
	for i in range(9):
		if i:
			i-=1
			scale = 1
			center_x = i%2 or -1
			center_y = i//2%2 or -1
			center_z = i//4 or -1
			color="r"
		else:
			scale = 1/(np.sqrt(3)-1)
			center_x = 0
			center_y = 0
			center_z = 0
			color="b"

		u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
		x = np.cos(u)*np.sin(v)/scale + center_x
		y = np.sin(u)*np.sin(v)/scale + center_y
		z = np.cos(v)/scale + center_z
		ax.plot_wireframe(x, y, z, color=color)
		
		# draw a point
		ax.scatter([center_x], [center_y], [center_z], s=100, c="g")

	resolution = 2
	x = np.linspace(*r,resolution)
	y = np.linspace(*r,resolution)
	X, Y = np.meshgrid(x,y)
	
	if not msg_frame_num:
		Z=Y/Y
	else:
		Z =Y
	ax.plot_surface(X,Y,Z)
	fig.show()

	#draw another figure with circles cross section.
	fig = plt.figure(1)
	plt.clf()
	ax= fig.gca()
	addition_x = (np.sqrt(1+msg_frame_num)-1)
	ax.set_xlim(-2-addition_x,2+addition_x)
	ax.set_ylim(-2,2)
	ax.set_aspect("equal")
	for i in range(4):
		center_x = i%2 or -1
		center_y = i//2 or -1

		center_x = center_x+center_x*addition_x

		circle = plt.Circle((center_x,center_y),1,color="r",fill=False)
		ax.scatter([center_x], [center_y], s=100, c="g")
		ax.add_artist(circle)
	circle = plt.Circle((0,0),np.sqrt(2+msg_frame_num)-1,color="b",fill=False)
	ax.add_artist(circle)
	fig.show()

	input(msg[msg_frame_num])