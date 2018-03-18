import numpy as np
import math

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    
    ### YOUR CODE HERE
    a=1 + np.exp(-x)
    
    dd = 1 / (a)

    ### END YOUR CODE
    
    return dd

def sigmoid_grad(dd):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input f should be the sigmoid
    function value of your original input x. 
    """
    #inter=1-sig
    #gr=sig*inter
    ### YOUR CODE HERE
    gra = dd * (1 - dd)
    ### END YOUR CODE

    return gra
    
    

def test_sigmoid_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    assert np.amax(f - np.array([[0.73105858, 0.88079708], 
        [0.26894142, 0.11920292]])) <= 1e-6
    print g
    assert np.amax(g - np.array([[0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])) <= 1e-6
    print "You should verify these results!\n"

def test_sigmoid(): 
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    x = np.array([[3, 4], [-5, -6]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    print g
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    test_sigmoid_basic()
    test_sigmoid()
