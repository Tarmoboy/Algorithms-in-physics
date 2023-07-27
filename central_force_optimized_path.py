"""

@author: Tarmo Ilves

"""

import numpy as np
import matplotlib.pyplot as plt
import powell

# Generate an optimized path for a particle under the influence of a central 
# force so that the particle starts and finishes at the same position.
# In other words the aim is to find an initial position and velocity for the 
# particle such that it follows a closed trajectory.

def vel_verlet(p, v):
    '''
    Velocity-Verlet algorithm for solving equations of motion

    Parameters
    ----------
    p : numpy array
        position vector
    v : numpy array
        velocity vector

    Returns
    -------
    p : numpy array
        position vector
    v : numpy array
        velocity vector

    '''

    a = acc(p)
    p = p + v * dt + 0.5 * a * dt**2
    v = v + 0.5 * a * dt
    a = acc(p)
    v = v + 0.5 * a * dt
    return p, v

def solve_path(p, v, T):
    '''
    Simulate the motion of a particle under the influence of a central force 
    using the Velocity-Verlet algorithm.

    Parameters
    ----------
    p : numpy array
        Position vector of the particle with x and y components.
    v : numpy array
        Velocity vector of the particle with x and y components.
    T : float
        Total simulation time.

    Returns
    -------
    numpy array
        Array containing the positions of the particle at each time step 
        with x and y components.
    numpy array
        Array containing the velocities of the particle at each time step 
        with x and y components.
    '''
    # Determine time
    time = np.arange(0, T, dt)
    # Initialize location and velocity
    location = np.zeros((time.shape[0], 2))
    velocity = np.zeros((time.shape[0], 2))
    # Update location and velocity using vel_verlet
    for i, t in enumerate(time):
        p, v = vel_verlet(p, v)
        location[i, :] = p
        velocity[i, :] = v
    return location, velocity

def optimize_this(params):
    '''
    Objective function for optimization to find the optimal initial 
    conditions for the particle's path.

    Parameters
    ----------
    params : numpy array
        Array containing the initial guesses for the particle's position and 
        velocity.

    Returns
    -------
    float
        The objective value to be minimized, which is the sum of the total 
        distance traveled and the change in velocity.
    '''
    p = params[:2]
    v = params[2:]
    # Determine location and velocity using solve_path
    location, velocity = solve_path(p, v, T)
    # Distance between last location and initial location
    distance = np.linalg.norm(location[-1] - location[0])
    # Distance between last velocity and initial velocity
    velocity_diff = np.linalg.norm(velocity[-1] - velocity[0])
    return distance + velocity_diff

# 
def acc(p):
    '''
    Calculate the acceleration of a particle under the influence of a 
    central force.

    Parameters
    ----------
    p : numpy array
        Position vector of the particle with x and y components.

    Returns
    -------
    numpy array
        Acceleration vector of the particle with x and y components.
    '''
    # 
    r2 = p[0]**2 + p[1]**2
    return -G*M*p/(r2*np.sqrt(r2+(L**2/4)))

# Initial guesses for optimized symmetrical path
p = np.array([0.2, -0.3])
v = np.array([-1, -0.2])

# Constants
L = 2   # Angular momentum
M = 10  # Mass of the central object, eg. a star
G = 1   # Gravitational constant
# Time and time-step
T = 1
dt = 0.01

# Location and velocity with initial guess using solve_path
location, velocity = solve_path(p, v, T)

# Initialization using initial guesses
best = np.array([0.2, -0.3, -1, -0.2])
# Finding best initial values using powell
best = powell.powell(optimize_this, best)
# Optimal location and velocity using best initial values
opt_location, opt_velocity = solve_path(best[:2], best[2:], T)

# Draw optimized path
plt.plot(opt_location[:, 0], opt_location[:, 1], label='Optimized')

# Draw path based on initial guess
plt.plot(location[:, 0], location[:, 1], label='Initial guess')

plt.legend()
plt.show()