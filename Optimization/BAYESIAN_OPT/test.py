from sklearn.gaussian_process.kernels import DotProduct, RBF, ExpSineSquared,ConstantKernel, WhiteKernel
from utils_graphics import *
from bayesian_opt import *
from acquistion_functions import *
from priors import *
from custom_gaussian_process_regressor import *
from custom_kernels import *

def main_1d(objective_fun,
			bounds,
			n_generations=10,
			size=2,
			k=0.1,
			acq_function=UCB, 			
			prior= prior_random,
			model=None,
			make_gif=False,
			plot_bounds_est=False):

	try:
		objective_fun(0)
	except:
		assert False, "wrong dim for objective_fun"

	X, y, model = bayesianOptmization(n_generations=n_generations, 
									size=size, 
									acq_function=acq_function,
									k=k, 
									objective_fun=objective_fun,
									prior= prior,
									bounds=bounds,
									make_gif=make_gif,
									plot_bounds_est=plot_bounds_est,
									model=model) 

def main_2d(objective_fun,
			bounds,
			n_generations=10,
			size=2,
			k=0.1,
			acq_function=UCB, 
			prior= prior_random,
			model=None,
			make_gif=False,
			plot_bounds_est=False):

	try:
		objective_fun([0,0])
	except:
		assert False, "wrong dim for objective_fun"

	X, y, model = bayesianOptmization(n_generations=n_generations, 
									size=size, 
									acq_function=acq_function,
									k=k, 
									objective_fun=objective_fun,
									prior= prior,
									bounds=bounds,
									model=model,
									make_gif=make_gif,
									plot_bounds_est=plot_bounds_est) 

""" Experiment with 1d domain"""

if __name__=='__main__':

	def multimodal_1d(x):
		return(+np.sin(3*(x+2)**2)-np.sin((10/3)*x))

	model=CustomGaussianProcessRegressor(CustomRBF(c=0.2))
	main_1d(size=2,
			objective_fun=multimodal_1d,		
			model=model,
			k=0.1,
			acq_function=EI,
			n_generations=10,
			bounds=np.array([[-1,1]]),make_gif=True,plot_bounds_est=True)





""" Experiment with 2d domain"""
if __name__=='__main__':
		
	def multimodal_2d(x):
		return(-3*np.sin(x[0]**2+x[1]**2)+5*np.cos(x[1]))
	
	model=CustomGaussianProcessRegressor(CustomRBF(c=0.5))
	main_2d(size=2,
			objective_fun=multimodal_2d,
			model=model,
			k=0.1,
			acq_function=EI,
			n_generations=10,
			bounds=np.array([[-1,1],[-1,1]]),
			make_gif=True,plot_bounds_est=True)