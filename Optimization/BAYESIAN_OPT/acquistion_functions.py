import numpy as np
from scipy.stats import norm

def is_all_pos(array):
    for i in range(len(array)):
        if array[i]<= 0:
            return(False)
    return(True)

def UCB(args):
  """
  Acquisition function: Upper Confidence Bound implementation
  The objective is to sample where the UCB is the greatest

  - args: dictionary containing arguments needed for the implementation:
    - "mu"
    - "std"
    - "best"
    - "k"

  Returns:
  - array u(X), result of acquisition function for each sample
  """
  mu =args['mu']
  std =args['std']
  k =args['k']
  
  assert len(mu)==len(std)

  return np.array([mu[i]+k*std[i] for i in range(len(mu))])

def LCB(args):
  """
  Acquisition function: Lower Confidence Bound implementation
  The objective is to sample where the LCB is the greatest
  - args: dictionary containing arguments needed for the implementation:
    - "mu"
    - "std"
    - "best"
    - "k"

  Returns:
  - array u(X), result of acquisition function for each sample
  """
  mu =args['mu']
  std =args['std']
  k =args['k']
  
  assert len(mu)==len(std)

  return np.array([mu[i]-k*std[i] for i in range(len(mu))])

def PI(args):
  """
  Acquisition function: Probabilty of Improvement implementation
  The objective is to maximize the probability of improvement

  - args: dictionary containing arguments needed for the implementation:
    - "mu"
    - "std"
    - "best"
    - "k"

  Returns:
  - array u(X), result of acquisition function for each sample
  """
  mu =args['mu']
  std =args['std']
  best=args['best']
  k=args['k']

  assert len(mu)==len(std)
  assert is_all_pos(std)

  z=np.array([(best+k-mu[i])/std[i] for i in range(len(mu))])
  return(norm.cdf(-z, loc=0, scale=1))

def EI(args):
  """
  Acquisition function: Expected Improvement implementation
  The objective is to maximize the expected improvement
  
  - args: dictionary containing arguments needed for the implementation:
    - "mu"
    - "std"
    - "best"
    - "k"

  Returns:
  - array u(X), result of acquisition function for each sample
  """
  mu =args['mu']
  std =args['std']
  best=args['best']
  k=args['k']
  
  assert len(mu)==len(std)
  assert is_all_pos(std)

  z=np.array([(best+k-mu[i])/std[i] for i in range(len(mu))])
  
  return(np.array([(mu[i]-best-k)*norm.cdf(-z[i],0,1) +std[i]*norm.cdf(z[i],0,1) for i in range(len(mu))]))
