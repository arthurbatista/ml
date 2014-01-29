import numpy as np
import random

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        # print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta

def gradient(X,y,theta,alpha,m,num_it):
    for i in range(0,num_it):
        h=np.dot(X,theta)

        # print("Iteration %d | Cost: %f" % (i, np.sum((h-y)**2)))

        loss_t0 = np.sum(h-y)
        loss_t1 = np.sum((h-y)*X[:,1])
        
        theta[0]=theta[0]-(alpha/m)*(loss_t0)
        theta[1]=theta[1]-(alpha/m)*(loss_t1)
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
alpha = 0.0005

theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)

theta1 = np.ones(n)
theta1 = gradient(x, y, theta1, alpha, m, numIterations)
print(theta)

print(np.polyfit(x[:,1],y,1))