""" Various useful functions. """

import numpy as np
import numpy.typing as npt

def logIo(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculate the natural log of the 0th order modified Bessel function of 1st kind.

    Arguments: 
        x: numerical vector

    Output:
        y: natural log of the 0th order modified Bessel function of the 1st kind of x

    Follows the exponential implementation of the Bessel function in Numerical 
    Recipes, Ch. 6.Translated to Python from C++ from the FSL source code and 
    vectorized for faster computations in Python.
    """

    y = np.zeros(x.shape)
    b = np.abs(x)
    a1 = (x[b < 3.75]/3.75)**2
    a2 = 3.75/b[b >= 3.75]
    y[b < 3.75] = np.log(1.0 + a1*(3.5156229 + a1*(3.0899424 + a1*(1.2067492 + a1*(0.2659732 + a1*(0.0360768 + a1*0.0045813))))))
    y[b >= 3.75] = b[b >= 3.75] + np.log((0.39894228+a2*(0.01328592 + a2*(0.00225319 + a2*(-0.00157565 + a2*
                                                                           (0.00916281 + a2*
                                                                            (-0.02057706 + a2*
                                                                             (0.02635537 + a2*
                                                                              (-0.01647633 + a2*0.00392377))))))))
                                                                               /np.sqrt(b[b >= 3.75]))

    return y

def halfSampleMode(X: npt.NDArray[np.float64]):
    """
    Calculate the half sample mode for each row in X.
     
    Arguments:
        X:   numerical matrix

    Output:
        hsm: half sample mode of each row in X

    The method build on a recursive call to _HSM_rec.
    
    Based on: Bickel DR, Fruhwirth R. "On a fast, robust estimator of the 
    mode: comparisons to other robust estimators with applications" Comput
    Stat Data Anal 2006;50:3500–3530.
    """

    def _HSM_rec(n, X):
        """
        Recursively extract a subsample of each row in X that contains half the 
        sample and has the smallest width. When each row is one, two or three
        values in length, calculate and return the half sample mode of the
        remaining values for each row.
        """

        # Ending cases
        if n == 1:
            return X
        elif n == 2:
            return np.sum(X, axis=1)/2  # Median of two numbers is the average.
        elif n == 3:   
            hsm = np.zeros(X.shape[0])
            # Values close to each other => mode.
            low = (X[:, 1]-X[:, 0]) < (X[:, 2]-X[:, 1])  # Use lower pair.
            eq = (X[:, 1] - X[:, 0]) == (X[:, 2]-X[:, 1])  # Use mid value.
            
            if np.any(low):
                hsm[low] = np.sum(X[low, :2], axis=1)/2
            
            if np.any(eq):
                hsm[eq] = X[eq, 1]
            
            if np.any(~(low|eq)):  # Upper pair
                hsm[~(low|eq)] = np.sum(X[~(low|eq), 1:], axis=1)/2  # Otherwise
            
            return hsm
        
        # General case (n > 3)
        wmin = X[:, -1] - X[:, 0]  # Set initial width as maximum width.
        N = int(np.ceil(n/2))  # Number of elements in half sample.
        j = np.zeros(X.shape[0])
        for i in range(n - N + 1):  # Find the smallest width that contains half of the sample.
            w = X[:, i + N - 1] - X[:, i]
            m = w < wmin
            wmin[m] = w[m]
            j[m] = i
        
        # Extract the subsample.
        columns = np.tile(np.arange(X.shape[1]), (X.shape[0], 1))
        Xsub = np.reshape(X[(columns >= np.tile(j, (X.shape[1], 1)).T) & (columns < np.tile(j+N, (X.shape[1], 1)).T)], (X.shape[0], N))
        # Repeat the process until n =< 3.
        hsm = _HSM_rec(N,Xsub)

        return hsm

    X = np.sort(X, axis=-1)
    if len(X.shape) > 1:
        n = X.shape[1]
    else:
        n = 1

    hsm = _HSM_rec(n, X)
    
    return hsm