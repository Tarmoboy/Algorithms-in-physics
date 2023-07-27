import numpy as np

def gold_search(func,a,b,tol=1e-6):
    '''Golden-section search, minimization algorithm

    Arguments:
      func:  Function func(x)
      a:     Lower bound
      b:     Upper bound
      tol:   Tolerance

    Returns:
      x:     Estimate of the location of minimum
    '''

    # Initialisation
    niter = int( np.ceil(-2.078087*np.log(tol/abs(b-a))) )
    R = 0.618033989
    C = 1.0 - R

    # First interior points
    x1 = R*a + C*b
    x2 = C*a + R*b
    f1 = func(x1)
    f2 = func(x2)

    for i in range(niter):
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = C*a + R*b
            f2 = func(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = R*a + C*b
            f1 = func(x1)

    if f1 < f2:
        return x1,f1
    return x2,f2


def gold_bracket(func,x1,h):
    '''Searching for the minimum limits in the golden-section search

    Arguments:
        func:   Function func(x) to be minimized
        x1:     Initial value
        h:      Step size
    Returns:
        a:      x lower than the minimum
        b:      x higher than the minimum
    '''

    R = 1.618033989
    f1 = func(x1)
    x2 = x1 + h
    f2 = func(x2)

    # Way down
    if f2 > f1:
        h = -h
        x2 = x1 + h
        f2 = func(x2)

        # Whether minimum is in between x1-h and x1+h or not
        if f2 > f1:
            return x2,x1-h

    # Search loop
    for i in range(100):
        h = R*h
        x3 = x2 + h
        f3 = func(x3)
        if f3 > f2:
            return x1,x3
        x1 = x2
        f1 = f2
        x2 = x3
        f2 = f3

    raise ValueError('Did not find a minimum')


def powell(func,x,h=1,tol=1e-6):
    '''Powell algorithm for minimization

    Arguments:
      func:    Function, which returns a scalar value with a vector
               argument x.
      x:       Starting values for iteration
      h:       Initial step size for line search
      tol:     Tolerance

    Returns:
      x:       Vector of parameters, which minimize the function func
    '''

    n = len(x)          # Amount of dimensions
    df = np.zeros(n)    # Change in the value of function
    u = np.identity(n)  # Direction vectors

    for j in range(100):
        xold = x.copy()
        fold = func(xold)

        # Line search in the direction of the direction vectors
        for i in range(n):
            v = u[i]
            def f(s):
                return func(x + s*v)
            a,b = gold_bracket(f,0.0,h)
            s,fmin = gold_search(f,a,b)
            df[i] = fold - fmin
            fold = fmin
            x = x + s*v

        # Last line search of the cycle
        v = x-xold
        def f(s): return func(x + s*v)
        a,b = gold_bracket(f,0.0,h)
        s,flast = gold_search(f,a,b)
        x = x + s*v

        # Check for convergence
        if np.sqrt(np.dot(x-xold,x-xold)/n) < tol:
            return x

        # Maximum change and new directions
        imax = np.argmax(df)
        for i in range(imax,n-1):
            u[i] = u[i+1]
        u[n-1] = v

    raise ValueError('Powell did not converge')