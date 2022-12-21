import numpy as np
from sigmoid import *

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    hypothesis = 0.0
    # Write your code here
    function = 0
    index = len(theta)

    for j in range(index):
        function += X[i, j] * theta[j]

    hypothesis = function
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.

    result = sigmoid(hypothesis)
    
    return result

