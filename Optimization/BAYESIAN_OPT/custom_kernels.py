import numpy as np

class CustomKernel:
    def __init__(self, kernel_function, initial_theta, theta_bounds=None):
        self.function = kernel_function
        self.initial_theta = initial_theta
        self.theta_bounds = theta_bounds


def CustomRBF(c=0.1):
    def rbf(x1, x2, theta):
        n,m = len(x1),len(x2)
        mat=np.zeros((n,m))

        for i in range(n):
            for j in range(m):
                squared_distance = np.sum((x1[i] - x2[j]) ** 2)
                mat[i][j]= np.exp(-squared_distance / (2 * theta ** 2))

        return mat

    return CustomKernel(kernel_function=rbf, initial_theta=c, theta_bounds=[(c*1e-3, c*1e3)])

