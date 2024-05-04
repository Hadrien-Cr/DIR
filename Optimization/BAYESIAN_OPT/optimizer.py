from numpy import argmax
from numpy.random import uniform
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm

def acquisition(X, Xsamples, model, acq_function, k):
	"""Returns an array that represent the acquisition function discretized"""
	def acquisition_function(function, mu, std, best, k):
		args = {"mu": mu, "std": std, "best": best, "k": k}
		result = function(args)
		return result

	# calculate the best surrogate score found so far
	yhat= model.predict(X) 
	best = max(yhat)

	# calculate mean and stdev via surrogate function
	mu, std = model.predict(Xsamples,return_std=True) 

	probs = acquisition_function(acq_function, mu, std, best, k) 
	return probs


def opt_acquisition(X, y, model, size, acq_function, k, bounds):
	"""Maximize the acquisition function over a discretize interval, by random search"""
	Xsamples = uniform(bounds[:, 0], bounds[:, 1],(size,(1 if len(X.shape)<2 else X.shape[1])))

	# calculate the acquisition function for each sample
	scores = acquisition(X, Xsamples, model, acq_function, k)

	# locate the index of the largest scores
	ix = argmax(scores)
	return Xsamples[ix]


"""
Parameters: 

	- s, the size of the random searches 
	- the number of generations
	- an acquisition function u (that determines where to draw the next samples)
	- a prior
	- an objective function f
	- kernel: the hyperparameter that defines the shape that the surrogate objective function can take
	- bounds for f [[min_x1,max1],...,[min_x,maxn]] where n is the dimension of the domain

PSEUDO CODE
Initialization: 
	- Make a initializing draw X (s samples) with the prior
	- Evaluate y=f(X)
	- Store them to arrays lx and ly
	- Initialize the surrogate objective function to default arbitrary weights

Repeat for the number of generations:
	- given lx and ly, and a random search of size s onthe acquisition function u, find the where to sample x_next 
	- Evaluate y_next=f(x_next)
	- Add them to arrays lx and ly
	- Fit the surrogate objective function to lx and ly

Returns:
	- the surrogate objective function
	- the arrays lx and ly
"""

def bayesianOptmization(n_generations, size, acq_function, objective_fun, prior, bounds,kernel = None, k = 0):
		assert size>=2, "Size should be >=2"
		# sample the domain with the prior
		X , y  = prior(objective_fun, size,bounds)

		# define the model
		model = GaussianProcessRegressor(kernel) #you can set the kernel and the optimizer

		for _ in tqdm(range(n_generations)):
				x_next = opt_acquisition(X, y, model, size, acq_function, k, bounds)
				value = objective_fun(x_next)
				X = np.concatenate((X, [x_next]))
				y = np.concatenate((y, [value]))
				model.fit(X, y)

		return X, y, model
