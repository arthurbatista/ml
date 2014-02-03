#!/usr/bin/python
#
# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
try:
    import numpy
    import math
    import scipy
    from scipy.stats import pearsonr
except:
    print "This implementation requires the numpy module."
    exit(0)

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    y = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * y * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * y * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

def matrix_factorization_2(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    y = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * y * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * y * P[i][k] - beta * Q[k][j])
    return P, Q.T

def matrix_factorization_lr(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    theta_t = Q.T[:,j]
                    x = P[i,:]
                    r_aprox =  numpy.dot(x,theta_t) - R[i][j]
                    for k in xrange(K):
                        P[i][k] = x[k]       - alpha * (r_aprox * theta_t[k] + beta * x[k])
                        Q[j][k] = theta_t[k] - alpha * (r_aprox * x[k]       + beta * theta_t[k])
    return P, Q

def matrix_factorization_lr_sr(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    theta_t = Q.T[:,j]
                    x = P[i,:]
                    r_aprox =  numpy.dot(x,theta_t) - R[i][j]
                    for k in xrange(K):
                        P[i][k] = x[k]       - alpha * (r_aprox * theta_t[k] + beta * x[k] + social_regularization(j,R,Q,k))
                        Q[j][k] = theta_t[k] - alpha * (r_aprox * x[k]       + beta * theta_t[k] + social_regularization(j,R,Q,k))
    return P, Q

def social_regularization(u, R, Q, k, beta=0.02):
    N = [
         [0,0,0,1],
         [0,0,0,1],
         [1,1,0,1],
         [1,0,0,0]
        ]

    N = numpy.array(N)

    reg = 0.0

    for i in xrange(len(N)):
        if u != i:
            if N[i][u] == 1:
                reg += pearson_def(R[:,u], R[:,i])*(Q[u,k]-Q[i,k])

    return beta*reg

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)


###############################################################################

if __name__ == "__main__":
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    P1 = numpy.copy(P);
    Q1 = numpy.copy(Q);
    
    P2 = numpy.copy(P);
    Q2 = numpy.copy(Q);

    P3 = numpy.copy(P);
    Q3 = numpy.copy(Q);
    
    nP, nQ = matrix_factorization(R, P, Q, K)
    nR = numpy.dot(nP, nQ.T)
    print nR

    nP1, nQ1 = matrix_factorization_2(R, P1, Q1, K)
    nR1 = numpy.dot(nP1, nQ1.T)
    print nR - nR1

    nP2, nQ2 = matrix_factorization_lr(R, P2, Q2, K)
    nR2 = numpy.dot(nP2, nQ2.T)
    print nR - nR2

    nP3, nQ3 = matrix_factorization_lr_sr(R, P3, Q3, K)
    nR3 = numpy.dot(nP3, nQ3.T)
    print nR - nR3

    # reg = social_regularization(0,R,Q,0)
    # print reg