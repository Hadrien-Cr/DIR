from numpy import argmax
from numpy.random import uniform
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm
from utils_graphics import *
from copy import copy
import os
import imageio

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
	Xsamples = uniform(bounds[:, 0], bounds[:, 1],(1000,(1 if len(X.shape)<2 else X.shape[1])))

	# calculate the acquisition function for each sample
	scores = acquisition(X, Xsamples, model, acq_function, k)

	# locate the index of the largest scores
	ix = argmax(scores)
	return Xsamples[ix]


"""
Parameters: 

	- s, the size of the initialiazing draw
	- the number of generations
	- an acquisition function u (that determines where to draw the next samples)
	- a prior
	- an objective function f
	- model: a 
	- bounds for f [[min_x1,max1],...,[min_x,maxn]] where n is the dimension of the domain

PSEUDO CODE
Initialization: 
	- Make a initializing draw X (s samples) with the prior
	- Evaluate y=f(X)
	- Store them to arrays lx and ly
	- Initialize the surrogate objective function to default arbitrary weights

Repeat for the number of generations:
	- given lx and ly, and a random search on the acquisition function u, find the where to sample x_next 
	- Evaluate y_next=f(x_next)
	- Add them to arrays lx and ly
	- Fit the surrogate objective function to lx and ly

Returns:
	- the surrogate objective function
	- the arrays lx and ly
"""

def bayesianOptmization(n_generations, size, acq_function, objective_fun, prior, bounds,
						model=None,k = 3,make_gif=False,plot_bounds_est=False):
		
		def plot_and_store(filename):
				if dim==1:
						plot_1d(X, y, model, size, objective_fun=objective_fun,bounds=bounds,
										plot_bounds_est=plot_bounds_est,filename=filename)
				elif dim==2:
						plot_2d(X, y, model, size, objective_fun=objective_fun,bounds=bounds,
										plot_bounds_est=plot_bounds_est,filename=filename)
										
		assert size>=2, "Size should be >=2"
		# sample the domain with the prior
		X , y  = prior(objective_fun, size,bounds)
		dim=len(X[0])
		# define the model
		if model is None:
				model=GaussianProcessRegressor()
		else:
				model=copy(model)

		model.fit(X, y)
		if make_gif:
				image_files=[]
				temp_dir="Optimization\BAYESIAN_OPT\progress_gif_"+str(dim)+"d"
				filename = os.path.join(temp_dir, f'frame_{0}.png')
				output="Optimization\BAYESIAN_OPT\progress_gif_"+str(dim)+"d"+".gif"
				plot_and_store(filename)
				image_files.append(filename)	

		for gen in tqdm(range(n_generations)):
				x_next = opt_acquisition(X, y, model, size, acq_function, k, bounds)
				value = objective_fun(x_next)
				X = np.concatenate((X, [x_next]))
				y = np.concatenate((y, [value]))
				model.fit(X, y)
				if make_gif:
					filename = os.path.join(temp_dir, f'frame_{gen+1}.png')
					plot_and_store(filename)
					image_files.append(filename)
						
						

		with imageio.get_writer(output, mode='I', duration=100*n_generations,loop=0) as writer:
			for filename in image_files:
				image = imageio.imread(filename)
				writer.append_data(image)
			print("Gif saved to Optimization " + output)
											

		return X, y, model
