import numpy as np
import random
from math import e
from array import array

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    for i in range(0, numIterations):
        h = np.dot(x, theta)
        loss = h - y
        # # avg cost per example (the 2 in 2*m doesn't really matter here.
        # # But to be consistent with the gradient, I include it)
        # cost = np.sum(loss ** 2) / (2 * m)
        # print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(x.T, loss)
        # update
        theta = theta - alpha / m * gradient
    return theta

def gradient_descent(X, y, theta, alpha, m, num_iters):

    for i in range(num_iters):
 
        h = X.dot(theta)
 
        errors_x1 = (h - y) * X[:, 0]
        errors_x2 = (h - y) * X[:, 1]
 
        theta[0] = theta[0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1] = theta[1] - alpha * (1.0 / m) * errors_x2.sum()
 
    return theta

def sgd(X, y, theta, alpha, m,num_iters):

    z = np.random.permutation(m)

    for k in range(num_iters):

        z = np.random.permutation(z)

        for i in z:
 
            theta[0] = theta[0] - alpha*(X[i].dot(theta) - y[i])*X[i,0]
            theta[1] = theta[1] - alpha*(X[i].dot(theta) - y[i])*X[i,1]
 
    return theta


def gradient_p(X,y,theta,alpha,m,numIterations):

    errors1_x1 = 0
    errors1_x2 = 0

    errors2_x1 = 0
    errors2_x2 = 0

    x1,x2 = np.split(X,2)
    y1,y2 = np.split(y,2)

    for i in range(0,numIterations):
        
        h1 = x1.dot(theta)
        errors1_x1 = (h1 - y1) * x1[:, 0]
        errors1_x2 = (h1 - y1) * x1[:, 1]

        h2 = x2.dot(theta)
        errors2_x1 = (h2 - y2) * x2[:, 0]
        errors2_x2 = (h2 - y2) * x2[:, 1]
    
        theta[0]=theta[0]-(alpha/m)*(errors1_x1.sum()+errors2_x1.sum())
        theta[1]=theta[1]-(alpha/m)*(errors1_x2.sum()+errors2_x2.sum())
        
    return theta


def genData(numPoints, bias, variance):
    x = np.ones(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        x[i][1] = i
        
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = genData(100, 25, 10)
m, n = np.shape(x)
numIterations= 1000000
alpha = 0.0002

theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)

theta1 = np.ones(n)
theta1 = sgd(x, y, theta1, alpha, m, 1000)
print(theta1)

print(np.polyfit(x[:,1],y,1))