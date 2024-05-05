import matplotlib.pyplot as plt
from numpy import asarray,arange
import numpy as np

def plot_1d(X, y, model, size, objective_fun, bounds,plot_bounds_est,filename):
	""" plot real observations vs surrogate function """
	# scatter plot of inputs and real objective function
	plt.figure(figsize=(10,5))
	plt.scatter(X[size:], y[size:], marker='^',c="orange",label='New samples',s=40)
	plt.scatter(X[:size], y[:size],c="brown",label='Initial samples',s=40)


	# line plot of surrogate function across domain5
	Xsamples = np.linspace(bounds[0][0], bounds[0][1], 50)
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples,sigmas = model.predict(Xsamples,return_std=True)

	plt.plot(Xsamples, objective_fun(Xsamples) ,label='actual function',c="red")
	plt.plot(Xsamples, ysamples,label='surrogate',alpha=0.8,c="blue")

	if plot_bounds_est:
		plt.plot(Xsamples, ysamples+3*sigmas,label='upper bound surrogate',alpha=0.5,c="purple")
		plt.plot(Xsamples, ysamples-3*sigmas,label='lower bound surrogate',c="green")
	# show the plot
	plt.legend(loc='upper right')
	plt.ylim((-3,3))
	plt.savefig(filename)
	plt.close()

	
def plot_2d(X, y, model, size, objective_fun, bounds,plot_bounds_est,filename):
    	
	"""Plot real observations vs surrogate function in 2D."""
	# Scatter plot of inputs and real objective function
	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(X[size:, 0], X[size:, 1], y[size:], marker='^', c='orange', label='New samples',s=40)
	ax.scatter(X[:size, 0], X[:size, 1], y[:size], c='brown', label='Initial samples',s=40)

	# Create a meshgrid for sampling across the domain
	x = np.linspace(bounds[0][0], bounds[0][1], 30)
	y = np.linspace(bounds[1][0], bounds[1][1], 30)
	Xsamples, Ysamples = np.meshgrid(x, y)
	grid_samples = np.c_[Xsamples.ravel(), Ysamples.ravel()]

	# Predict surrogate function and standard deviation across the grid
	ysamples, sigmas = model.predict(grid_samples, return_std=True)
	ysamples = ysamples.reshape(Xsamples.shape)
	sigmas = sigmas.reshape(Xsamples.shape)

	# Calculate the upper and lower bounds
	upper_bound = ysamples + 3 * sigmas
	lower_bound = ysamples - 3 * sigmas
	actual_z = np.array([objective_fun(grid_samples[i]) for i in range(len(grid_samples))]).reshape(Xsamples.shape)

	ax.plot_surface(Xsamples, Ysamples, actual_z, color='red',alpha=0.8, label='Actual function')
	ax.plot_surface(Xsamples, Ysamples, ysamples, color='blue',alpha=0.8, label='Surrogate')
	if plot_bounds_est:
		ax.plot_surface(Xsamples, Ysamples, upper_bound, color="purple",alpha=0.3,label='Upper bound')
		ax.plot_surface(Xsamples, Ysamples, lower_bound, color="green",alpha=0.3,label='Lower bound')

	# Add legend and labels
	ax.set_xlabel('X1')
	ax.set_ylabel('X2')
	ax.set_zlabel('Objective Function Value')
	ax.axes.set_zlim3d((-3,3))
	ax.legend(loc='best')
	plt.savefig(filename)
	plt.close()

