import numpy as np
from numpy.random import uniform


def prior_random(objective_fun, size,bounds):
  """
  Prior implementation, returns a tuple (X, Y), where `X` is a random array of size `size` and `Y` is the array of `f(x)`.
  - objective_fun: function that implements the objective function in use. Type(objective_fun) = <class 'function'>
  - size: an Integer that specifies the size of X and Y arrays
  - bounds: [[min_x1,max1],...,[min_x,maxn]] where n is the dimension of the domain
  Returns:
  - X, Y
  """
  dim=len(bounds)
  X=uniform(bounds[:,0],bounds[:,1],(size,dim))
  Y=np.array([objective_fun(X[i]) for i in range(size)])
  return(X,Y)

  """
  Prior implementation, returns a tuple (X, Y), where `X` is an array of size `size` of numbers sampled from a uniform distribution `[0, 0.5]` and `Y` is the array of `f(x)`.
  - objective_fun: function that implements the objective function in use. Type(objective_fun) = <class 'function'>
  - size: an Integer that specifies the size of X and Y arrays

  Returns:
  - X, Y
  """
  X=uniform(0,0.5,size)
  Y=np.array([objective_fun(X[i]) for i in range(size)])
  return(X,Y)