from sklearn.gaussian_process.kernels import DotProduct, RBF, ExpSineSquared
from utils_graphics import *
from optimizer import *
from acquistion_functions import *
from priors import *



def main_1d(objective_fun,
			bounds,
			n_generations=10,
			size=5,
			acq_function=UCB, 			
			prior= prior_random,
			kernel=None):

	try:
		objective_fun(0)
	except:
		assert False, "wrong dim for objective_fun"

	X, y, model = bayesianOptmization(n_generations=n_generations, 
									size=size, 
									acq_function=acq_function, 
									objective_fun=objective_fun,
									prior= prior,
									bounds=bounds,
									kernel=kernel) 
	plot_1d(X, y, model, size, objective_fun=objective_fun,bounds=bounds)
	# best result
	# ix = argmax(y)
	# print('Best Result: x=%.3f, y=%.3f' % (X[ix][0], y[ix][0]))

def main_2d(objective_fun,
			bounds,
			n_generations=10,
			size=5,
			acq_function=UCB, 
			prior= prior_random,
			kernel=None):

	try:
		objective_fun([0,0])
	except:
		assert False, "wrong dim for objective_fun"

	X, y, model = bayesianOptmization(n_generations=n_generations, 
									size=size, 
									acq_function=acq_function, 
									objective_fun=objective_fun,
									prior= prior,
									bounds=bounds,
									kernel=kernel) 
	plot_2d(X, y, model, size, objective_fun=objective_fun,bounds=bounds)
	# best result
	# ix = argmax(y)
	# print('Best Result: x=%.3f, y=%.3f' % (X[ix][0], y[ix][0]))


def multimodal_1d(x):
  return(-np.sin(3*x)+np.sin((10/3)*x))


def multimodal_2d(x):
  return((-np.sin(10*x[0])+np.sin((10/3)*x[0]))*(-np.sin(x[1]+1)+np.sin((8/3)*x[1])))

main_1d(objective_fun=multimodal_1d,kernel=RBF(0.1),n_generations=100,bounds=np.array([[-1,1]]))
main_2d(objective_fun=multimodal_2d,kernel=RBF(0.1),n_generations=100,bounds=np.array([[-1,1],[-1,1]]))