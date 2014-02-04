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
    U     : an initial matrix of dimension M x K
    V     : an initial matrix of dimension N x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""

def matrix_factorization_sr(R, U, V, K, GS, steps=5000, alpha=0.0002, lamb=0.02 ,beta=0.02):
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    Ui = U[i,:]
                    Vj = V[j,:]
                    Ra =  numpy.dot(Ui.T,Vj) - R[i][j]
                    for k in xrange(K):
                        U[i][k] = Ui[k] - alpha * ( (Ra * Vj[k]) + (lamb * Ui[k]) + (beta * sr_f(i,R,U,k,GS)) + (beta * sr_g(i,R,U,k,GS)) )
                        V[j][k] = Vj[k] - alpha * ( (Ra * Ui[k]) + (lamb * Vj[k]) )
    return U, V

def sr_f(i, R, U, k, GS):
    reg = 0.0

    for f in xrange(len(GS)):
        if i != f:
            if GS[f][i] == 1:
                reg += sim(R[i,:], R[f,:]) * (U[i,k]-U[f,k])

    return reg

def sr_g(i, R, U, k, GS):
    l = []
    reg = 0.0

    for f in xrange(len(GS)):
        if i != f:
            if GS[f][i] == 1:
                for g in xrange(len(GS)):
                    if f != g and i != g:
                        if GS[g][f] == 1 and GS[g][i] == 0:
                            try:
                                l.index(g)
                            except ValueError:
                                l.append(g)

    for g in l:
        reg += sim(R[i,:], R[g,:]) * (U[i,k]-U[g,k])

    return reg

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def sim(x, y):
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

    sim = diffprod / math.sqrt(xdiff2 * ydiff2)

    return (sim+1)/2


###############################################################################

if __name__ == "__main__":

    GS = [
         [0,1,1,0,0],
         [1,0,0,1,0],
         [1,0,0,1,0],
         [0,1,1,0,1],
         [0,0,0,1,0]
        ]

    GS = numpy.array(GS)

    # matrix m x n
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

    R = numpy.array(R)

    # users
    M = len(R)

    # itens
    N = len(R[0])

    K = 2

    U = numpy.random.rand(M,K)
    V = numpy.random.rand(N,K)
    
    nP, nQ = matrix_factorization_sr(R, U, V, K, GS)
    nR = numpy.dot(nP, nQ.T)

    print R
    print nR
    print R - nR