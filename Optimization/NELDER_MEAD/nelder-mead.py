import numpy as np
import matplotlib.pyplot as plt
from math import *
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize,brute


def nelder_mead(f, initial_simplex, max_iter=100, tol=1e-10, alpha=1, beta=0.5, gamma=5,return_history=False):
    """Nelder-Mead optimization algorithm."""
    n = len(initial_simplex[0])
    simplex = np.array(initial_simplex)
    simplex_history = [simplex.copy()]
    
    for i in range(max_iter):
        # Sort vertices by function value
        simplex = simplex[np.argsort([f(x) for x in simplex])]
        best, second_best, worst = simplex[0], simplex[1], simplex[-1]
        
        # Compute the centroid xn (except worst point)
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        reflected = centroid + alpha * (centroid - worst)

        # Expansion: If the reflected point is better than the second-worst, 
        #   then try expand in that direction (keep if improves the worst)
        if f(reflected) < f(second_best):
            expanded = centroid + gamma * (reflected - centroid)
            if f(expanded) < f(second_best):
                simplex[-1] = expanded
            else:
                simplex[-1] = reflected
        
        elif f(reflected) >= f(second_best):
            #  Check if the reflected point is worse than the worst, if not replace
            if f(reflected) < f(worst):
                simplex[-1] = reflected

            # Contraction:  If the reflected point is worse than the worst, 
            #   then try to contract the simplex towards the best point.
            contracted = centroid + beta * (worst - centroid)
            if f(contracted) < f(worst):
                simplex[-1] = contracted

            # Shrink: Else: Reduce the simplex towards the best point.
            else:
                for i in range(1, n+1):
                    simplex[i] = best + 0.5 * (simplex[i] - best)
        
        # Check convergence
        if np.linalg.norm(simplex[0] - simplex[-1]) < tol:
            break
        
        simplex_history.append(simplex.copy())

    if return_history:
        return np.array(simplex_history)
    else:
        return(f(simplex_history[-1][0]))

'''

MAKE A GIF THAT ILLUSTRATES THE PROCESS

'''


def make_gif(initial_simplex,custom_loss):
    # Define initial simplex
    initial_simplex = [[3, 3], [2.5, 2.5], [3, 2.5]]
    # Run Nelder-Mead algorithm
    simplex_history = nelder_mead(custom_loss, initial_simplex,max_iter=25,return_history=True)
    solution=simplex_history[-1][0]
    # Plotting
    x = np.linspace(-4, 4, 400)
    y = np.linspace(-4, 4, 400)
    X, Y= np.meshgrid(x, y)
    Z = custom_loss([X,Y])
    print(Z.shape)

    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Nelder-Mead Optimization')
    ax.grid(True)

    def update(frame):
        ax.clear()
        ax.contour(X, Y, Z, levels=200)
        simplex = simplex_history[frame]
        ax.plot(simplex[:,0], simplex[:,1], 'r--o', alpha=0.5)
        ax.plot([simplex[-1,0], simplex[0,0]], [simplex[-1,1], simplex[0,1]], 'r--o', alpha=0.6)
        ax.plot(solution[0], solution[1], 'go', label='Optimal Solution')
        ax.legend()

    ani = FuncAnimation(fig, update, frames=len(simplex_history), interval=300)
    ani.save('Optimization\NELDER_MEAD\GIF_nelder_mead.gif', writer='imagemagick')
    plt.show()

def custom_loss(x):
    return np.sqrt((x[0]**2+1) +(x[0]**2+1))+ ((1+x[0]**2+1))*(np.log(x[1]**2+1)) + (1.2+np.sin(x[0]**2))

#make_gif([[3, 3], [2.5, 2.5], [3, 2.5]],custom_loss)



'''

BENCHMARK

For 4 different loss functions, I made graphs that plot the error of my algo and scipy implementation,
 for different max_iter value (in the x axis of the graph)

'''


# Benchmark loss functions
def rosenbrock(x):
    return (1-x[0])**2+100*(x[0]-x[1]**2)**2

def sphere(x):
    return (x[0]**2+x[1]**2)

def rastrigin(x):
    return 2  + (x[0]**2+x[1]**2) -  np.cos(2 * np.pi * x[0])  -  np.cos(2 * np.pi * x[1])

def ackley(x):
    return -20.0 * np.exp(-0.2 *  np.sqrt(0.5 * (x[0]**2 + x[1]**2)))- np.exp(0.5 * ( np.cos(2 * np.pi * x[0])+ np.cos(2 *  np.pi * x[1]))) + np.exp(1) + 20

# Define parameters for benchmarking
max_iters = [ k for k in range(2,60)]
loss_functions = [rosenbrock, sphere, rastrigin, ackley]

initial_simplex=[[3, 3],[1,2], [2,1.5]]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

for i, loss_func in enumerate(loss_functions):
    row = i // 2
    col = i % 2

    ax = axs[row, col]

    error_custom=[]
    error_scipy=[]

    for max_iter in max_iters:
    

        # Run Nelder-Mead algorithm
        result_custom = nelder_mead(loss_func, initial_simplex,tol =1e-50, max_iter=max_iter)
        error_custom.append(abs(result_custom))

        # Run scipy's implementation
        result_scipy = minimize(loss_func,initial_simplex[0], method='Nelder-Mead',tol=1e-50,options={'maxiter': max_iter,'initial_simplex': np.array(initial_simplex)}).fun
        error_scipy.append(abs(result_scipy))


        # Calculate errors
        

    ax.plot(max_iters, error_custom,linewidth=2, label=f'Custom, {loss_func.__name__}')
    ax.plot(max_iters, error_scipy, linestyle='dotted',linewidth=3,label=f'Scipy, {loss_func.__name__}')
    ax.set_yscale('log')
    ax.set_xlabel('Max Iterations')
    ax.set_ylabel('Error on the minimum found')
    ax.set_title(f'Comparison of Implementation Errors on {loss_func.__name__} function')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('Optimization\NELDER_MEAD\GIF_nelder_mead.gif')
plt.show()

