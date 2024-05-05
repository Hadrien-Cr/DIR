import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from custom_kernels import *


class CustomGaussianProcessRegressor:

    """ 
    Pseudo code for predicting
    input: X (training inputs), y (targets), k (covariance function), σ^2 n (noise level), X* (test inputs) 
    - L = cholesky(K + σ 2 n I) 
    - α = LT\(L\y) 
    - f(x*) = k(x,x*)T α 
    - v := L\(x,x*)
    - V[f*] := k(x*, x*) -vTv
    - log p(y|X) =-(1/2)yTα-sum(Lii)-(n/2) log(2pi)
    return: f(x*), V[f](x*)
    
    """

    def __init__(self, kernel, noise_level=1e-5, optimizer=fmin_l_bfgs_b):
        self.kernel = kernel
        self.noise_level = noise_level
        self.optimizer = optimizer
        self.L = None
        self.alpha = None
        self.theta = None
        self.X_train=None

    def fit(self, X, y):
        def search_theta(theta):
            n = len(X)
            K = self.kernel.function(X, X, theta)
            K += self.noise_level * np.eye(n)  
            L = np.linalg.cholesky(K)  # Cholesky decomposition
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))  # Solve equation

            log_marginal_likelihood = -0.5 * y.T @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi) - 0.5*n*np.log(2*np.pi)

            return theta,log_marginal_likelihood  

        # Initial theta and bounds
        initial_theta = self.kernel.initial_theta
        bounds = self.kernel.theta_bounds

        # Optimize theta
        result = self.optimizer(search_theta, x0=initial_theta, bounds=bounds)
        self.theta = result[0]  # Optimized theta

        # Calculate covariance matrix K and Cholesky decomposition
        K = self.kernel.function(X, X, self.theta)
        K += self.noise_level * np.eye(len(X))
        self.L = np.linalg.cholesky(K)  # Cholesky decomposition
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))  # Solve equation
        self.X_train=X


    def predict(self, X_new, return_std=False):
        assert (self.L is not None) and (self.alpha is not None) and (self.theta is not None) and (self.X_train is not None), "Model not fitted yet."
        assert X_new.shape[1]==self.X_train.shape[1],"Inconsistent dimensions"

        # Calculate covariance between training and test data
        K = self.kernel.function(self.X_train, X_new, self.theta)
        # Calculate mean prediction
        f_new=np.dot(K.T, self.alpha).reshape(-1)

        if return_std:
            v = np.linalg.solve(self.L, K)
            # Calculate variance of the predictions
            vf_new = self.kernel.function(X_new, X_new, self.theta) - np.dot(v.T, v)
            std = np.sqrt(np.diag(vf_new)).reshape(-1)
            return f_new, std
        else:
            return f_new




if __name__=='__main__':

    # Create regressor instance
    model = CustomGaussianProcessRegressor(kernel=CustomRBF(c=0.5))

    def fun(x):
        return(np.sin(2 * np.pi * (x[:,0]+2))*3*x[:,1])


    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')
    X = np.random.uniform(0, 1, (20,2))
    Y = fun(X)
    model.fit(X,Y)
        # Create a meshgrid for sampling across the domain
    x = np.linspace(0,1, 20)
    y = np.linspace(0,1, 20)
    print('start pred')
    Xsamples, Ysamples = np.meshgrid(x, y)
    grid_samples = np.c_[Xsamples.ravel(), Ysamples.ravel()]
    ysamples, sigmas = model.predict(grid_samples, return_std=True)
    ysamples = ysamples.reshape(Xsamples.shape)
    sigmas = sigmas.reshape(Xsamples.shape)
    print('end pred')


        # Calculate the upper and lower bounds
    upper_bound = ysamples + 3 * sigmas
    lower_bound = ysamples - 3 * sigmas
    actual_z = fun(grid_samples).reshape(Xsamples.shape)

    ax.plot_surface(Xsamples, Ysamples, actual_z, color='red',alpha=0.8, label='Actual function')
    ax.plot_surface(Xsamples, Ysamples, ysamples, color='blue',alpha=0.8, label='Surrogate')

    ax.plot_surface(Xsamples, Ysamples, upper_bound, color="purple",alpha=0.3,label='Upper bound')
    ax.plot_surface(Xsamples, Ysamples, lower_bound, color="green",alpha=0.3,label='Lower bound')
    ax.axes.set_zlim3d((-3,3))
    plt.show()