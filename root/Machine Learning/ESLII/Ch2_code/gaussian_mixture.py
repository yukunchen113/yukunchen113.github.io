""" 
This code will create a gaussian mixture model.
Example usages can be seen with the coded versions of
scenarios one and two below in the main() function.

Specifications to be achieved:
	- can generate a mixture of gaussian with num_gaussians number of 
		component gaussians (one mixture per class)
	
	- each component gaussian will have a randomly 
		generated mean and standard deviation within
		a specified range.
	
	- can individually specify the parameters for 
		each component gaussian, the unspecified
		ones will be random, and if the specification is
		greater than num_gaussians, the first few will be selected 
		and a warning will be prompted. same rules
		apply for unspecified dimensions of multivariate 
		gaussians.

	- can specify N dimensional gausian (will
		apply across whole mixture.)

	- can plot the result if N is less than or equal to
		3. Otherwise, a marignalization will be applied, 
		to decrease the dimensionality. The kept 
		dimensions can be specified, otherwise is random.
		- plot takes in a N matrix of datapoints and plots
		them together, seperate variables are different 
		classes

	- number of samples can be specified. default = 100
"""
import numpy as np 
import matplotlib.pyplot as plt
import random as rd
def get_random(dist_type, shape, val_range):
	range_val = val_range[1] - val_range[0]
	if dist_type == "uniform":
		out = np.random.rand(*shape)*range_val+val_range[0]
	elif dist_type == "normal":
		out = np.random.normal((val_range[1] + val_range[0])/2, range_val/2, shape)
	return out

class mixture():
	#gaussian mixture. Used for one class.
	def __init__(self, N=2, num_gaussians=1, mu_range=[-1, 1], std_range=[0, 0.5], **kwargs):
		#num_gaussians is the number of component gaussians
		#N is the number of dimensions the gaussian
		#mu_range is the range of the means sampled
		#std_range is the range of values for standard deviation
		#kwargs inclues special options.
		if "mu_dist" in kwargs:
			self._mu_dist = kwargs["mu_dist"]
				#distribution of mean
				#current options"
					#uniform
					#normal - will use mu_range's range as 
						#2xstd of mu, and mu_range's midpoint as mu of mu distribution.
		else:
			self._mu_dist = "uniform"
		self._std_dist = "uniform"
		self._N = N
		self._mu_range = mu_range
		self._std_range = std_range
		self._num_gaussians = num_gaussians
		self._gaussians = None
		self._current_set = {}
		self._gen_gaussians()

	def _gen_gaussians(self):
		#creates a num_gaussians x N x 2 matrix, where the first column 
			#is the mean and the second column is the standard deviation.
		#variables to use:
		N = self._N
		mu_range = self._mu_range
		std_range = self._std_range
		num_gaussians = self._num_gaussians
		mu_dist = self._mu_dist
		std_dist = self._std_dist

		#make the means
		means = get_random(mu_dist, (num_gaussians, N, 1), mu_range)

		#make the stddevs
		std = get_random(std_dist, (num_gaussians, N, 1), std_range)

		#combine the two matricies
		gaussians = np.concatenate((means,std),axis=-1)
		
		#setting variables
		self._gaussians = gaussians

	def set_gaussian(self, set_dict):
		#takes in dictionary that as values as [mu, std] and 
			#key must be the index for the dictionary to replace.
			#key must be a tuple of indicies where
			#first integer is between 0 and num_gaussians
			#second integer is between 0 and N.
		#get gaussians must be already called.
		#variables to use:
		current_set = self._current_set
		gaussians = self._gaussians
		N = self._N
		mu_range = self._mu_range
		std_range = self._std_range
		num_gaussians = self._num_gaussians
		#do checks to see if set_dict is in a correct format
		if gaussians is None:
			raise Exception("gaussians were not generated before set gaussian was called!")

		for k,v in set_dict.items():	
			if not type(k) == tuple:
				raise Exception("set_dict must have keys as tuples!")
			if not type(v) == list:
				raise Exception("set_dict must have values as lists!")
			if not len(k) == 2:
				raise Exception("set_dict keys must be indicies for [gaussian num, N], and thus have a size of 2!")
			if not len(v) == 2:
				raise Exception("set_dict values must be values of [mean, stddev], and thus have a size of 2!")

		#set the gaussians
		for k,v in set_dict.items():
			gaussians[k]=v

		current_set.update(set_dict)		
		#setting variables
		self._gaussians = gaussians
		self._current_set = current_set
	def regen_gaussians(self, clear=False):
		#regenerates with the conditions specified.
		#if clear is true, the set gaussians from before will be
			#forgotten.
		#variables to use:
		current_set = self._current_set

		#regenerate gaussians
		self._gen_gaussians()
		self.set_gaussian(current_set)

	def sample(self, num_samples=100, weights=None):
		#samples from current distributions
		#the ith weight is the probability of sampling from the ith 
			# gaussian, if weight = None, sample uniformly.
		gaussians = self._gaussians
		num_gaussians = self._num_gaussians

		#generate how many samples will come from each distribution
		select = np.random.randint(0,num_samples, num_gaussians)
		if not weights is None:
			assert len(weights) == num_gaussians, "weights are not specified properly for the gaussians"
			select = select*weights

		#make the size equal to 100.
		select = np.round(select/np.sum(select)*num_samples)

		#correct number of samples due to rounding error.
		difference = np.sum(select) - num_samples
		for i in np.random.randint(0, num_gaussians, int(np.abs(difference))):
			select[i] = select[i] - np.sign(difference)

		#make sure that we are selecting the right number
		assert np.sum(select) == num_samples

		#select samples for each gaussian.
		samples = None
		i=0
		for sample_size in select:
			#current gaussians
			c_norm = np.transpose(gaussians[i])

			new_samples = np.random.normal(c_norm[0],c_norm[1],[int(sample_size),c_norm.shape[1]])

			if samples is None:
				samples = new_samples
			else:
				samples = np.concatenate((samples,new_samples),axis=0)
			i+=1
		np.random.shuffle(samples)
		return samples

def main():
	num_classes = 2
	num_samples = 100#samples per class
	#scenario 1
	scen1 = {}
	for i in range(num_classes):
		mix = mixture(mu_range = [-5,5], std_range=[7,10])
		scen1[mix] = mix.sample(num_samples)

	#scenario 2
	scen2 = {}
	for i in range(num_classes):
		mix = mixture(num_gaussians=10, mu_range = [-4,4], std_range=[0,2], mu_dist="normal")
		scen2[mix] = mix.sample(num_samples)
	j = 0
	for scen in [scen1, scen2]:
		fig = plt.figure(j)
		for k,v in scen.items():
			plt.scatter(v[:,0], v[:,1])
		j+=1
		fig.show()
	input()
	
if __name__ == "__main__":
	main()