import matplotlib.pyplot as plt
import numpy as np

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

def plot_1d(X, y, model, acq_function, k, size, objective_fun, bounds,plot_bounds_est,filename):
	""" plot real observations vs surrogate function """
	# scatter plot of inputs and real objective function
	plt.figure(figsize=(10,10))
	ax1=plt.subplot(2,1,1)
	
	ax1.scatter(X[size:], y[size:], marker='^',c="orange",label='New samples',s=40)
	ax1.scatter(X[:size], y[:size],c="brown",label='Initial samples',s=40)


	# line plot of surrogate function across domain5
	Xsamples = np.linspace(bounds[0][0], bounds[0][1], 50)
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples,sigmas = model.predict(Xsamples,return_std=True)

	ax1.plot(Xsamples, objective_fun(Xsamples) ,label='actual function',c="red")
	ax1.plot(Xsamples, ysamples,label='surrogate',alpha=0.8,c="blue")

	if plot_bounds_est:
		ax1.plot(Xsamples, ysamples+3*sigmas,label='upper bound surrogate',alpha=0.5,c="purple")
		ax1.plot(Xsamples, ysamples-3*sigmas,label='lower bound surrogate',c="green")
	# show the plot
	ax1.legend(loc='upper right')
	plt.ylim((-3,3))

	#plot the acq scores
	ax2=plt.subplot(2,1,2)
	scores = acquisition(X, Xsamples, model, acq_function, k)
	ax2.plot(Xsamples, scores ,label='Acquisition Score',c="lightblue")
	ax2.legend(loc='upper right')

	#save
	plt.savefig(filename)
	plt.close()

	
def plot_2d(X, y, model, acq_function, k, size, objective_fun, bounds,plot_bounds_est,filename):
		
	"""Plot real observations vs surrogate function in 2D."""
	# Scatter plot of inputs and real objective function
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16), gridspec_kw={'height_ratios': [2, 2]})
	ax1.axis("off")
	ax2.axis("off")
	ax1 = fig.add_subplot(211, projection='3d')
	


	ax1.scatter(X[size:, 0], X[size:, 1], y[size:], marker='^', c='orange', label='New samples',s=40)
	ax1.scatter(X[:size, 0], X[:size, 1], y[:size], c='brown', label='Initial samples',s=40)

	# Create a meshgrid for sampling across the domain
	x = np.linspace(bounds[0][0], bounds[0][1], 10)
	y = np.linspace(bounds[1][0], bounds[1][1], 10)
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

	ax1.plot_surface(Xsamples, Ysamples, actual_z, color='red',alpha=0.8, label='Actual function')
	ax1.plot_surface(Xsamples, Ysamples, ysamples, color='blue',alpha=0.5, label='Surrogate')
	if plot_bounds_est:
		ax1.plot_surface(Xsamples, Ysamples, upper_bound, color="purple",alpha=0.3,label='Upper bound')
		ax1.plot_surface(Xsamples, Ysamples, lower_bound, color="green",alpha=0.3,label='Lower bound')

	# Add legend and labels
	ax1.set_xlabel('X1')
	ax1.set_ylabel('X2')
	ax1.set_zlabel('Objective Function Value')
	ax1.axes.set_zlim3d((-3,3))
	ax1.legend(loc='lower left')


	#plot the acq scores
	ax2 = fig.add_subplot(212, projection='3d')
	scores = acquisition(X, np.concatenate((Xsamples.reshape(-1,1),Ysamples.reshape(-1,1)),axis=1), model, acq_function, k).reshape(Xsamples.shape)
	ax2.plot_surface(Xsamples, Ysamples, scores ,label='Acquisition Score',color="lightblue")
	ax2.legend(loc='upper right')

	#save
	plt.savefig(filename)
	plt.close()

