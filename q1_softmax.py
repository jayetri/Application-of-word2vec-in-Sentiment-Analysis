import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """

    ### YOUR CODE HERE
    l=len(x.shape)
    if l == 1:
        up = np.exp(x-np.max(x))
        down= np.sum(up)
        x = up/ down
    else:
        
            up = np.apply_along_axis(lambda x: np.exp(x - np.max(x)), 1, x)
            down = np.apply_along_axis(lambda up: 1.0/(np.sum(up)), 1, up)
           

            if len(down.shape) == 1:
                p=down.shape[0]
                down = down.reshape((p, 1))
            x = up * down
        ### END YOUR CODE
   
    return x 
    

def test_softmax_basic():
  
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    print "ran test2 result"
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6
    print "ran test3"
    print "You should verify these results!\n"

def test_softmax():
    """ 
    Use this space to test your softmax implementation by running:
        python q1_softmax.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    
    test2 = softmax(np.array([1000,500,300]))
    print test2
    #
    ### END YOUR CODE  

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()