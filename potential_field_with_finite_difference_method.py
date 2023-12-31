"""

@author: Tarmoboy

"""
import numpy as np
import matplotlib.pyplot as plt

# This example uses the finite difference Method to solve the electric 
# potential and visualize the electric field in a 2D space with charge 
# distributions. It shows the electric potential using contour plots and the 
# electric field using arrow fields. The simulation represents basic 
# electrostatics in a 2D space.

# Variables
N = 100     # Amount of grid points
L = 1       # Length and height of the rectangular area
h = L/(N-1) # Lattice constant
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y, indexing='ij')
max_iter = 500

# Density
rho = np.zeros((N, N))
rho[20:30, 20:30] = 1000    # Square where rho=1000
rho[60:70, 60:70] = -1000   # Square where rho=-1000

# Potential field
phi = np.zeros((N, N))

# Iteration using finite difference method
for k in range(max_iter):
    for i in range(1, N-1):
        for j in range(1, N-1):
            # Tried to put phi=0 inside the squares
            #if (i >= 20 and i < 30 and j >= 20 and j < 30)\
                #or (i >= 60 and i < 70 and j >= 60 and j < 70):
                #phi[i, j] = 0
            #else:
            # Have not used over-relaxation
            phi[i, j] = 1/4*(phi[i-1, j] + phi[i+1, j] + phi[i, j-1]\
                              + phi[i, j+1] + h**2 *rho[i, j])

# A collection of points of contours for drawing
levels = np.linspace(-1, 1, 10)
# Contours of the potential field
plt.contour(phi, levels=levels)

# Flow field, a little bit more rough
step = 5
x = np.arange(0, N, step)
y = np.arange(0, N, step)
x, y = np.meshgrid(x, y, indexing='ij')
# The flow field can be calculated from the potential Phi in directional 
# derivatives
vx, vy = np.gradient(phi[::step, ::step], 1)
# Draw the velocity field as an arrow field, where the length of the arrow 
# reflects flow intensity and direction flow direction
plt.quiver(x, y, vx, -vy)
plt.show()