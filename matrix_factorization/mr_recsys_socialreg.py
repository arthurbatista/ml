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
    from random import randint
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

def rmse(predictions, R):
    rmse = 0
    for i in xrange(len(R)):
        for j in xrange(len(R[i])):
            if R[i][j] > 0:
                rmse += numpy.sqrt(((predictions[i][j] - R[i][j]) ** 2).mean())

    return rmse

def sgd(R, U, V, K=2, steps=1800000, alpha=0.0002, lamb=0.02 ,beta=0.02):
    for step in xrange(steps):

        i = randint(0,len(U)-1)
        j = randint(0,len(V)-1)

        if R[i][j] > 0:
            Ui = U[i,:]
            Vj = V[j,:]
            Ra =  R[i][j] - numpy.dot(Ui.T,Vj)
            u_temp = Ui + 2*alpha * ( (Ra * Vj) - (lamb/len(U) * Ui) )
            V[j]   = Vj + 2*alpha * ( (Ra * Ui) - (lamb/len(V) * Vj) )
            U[i]   = u_temp

    return U, V

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

def dsgd_3x3(R, U, V):

    v_split_R = numpy.split(R,3)
    
    h_split_0 = numpy.hsplit(v_split_R[0],3)
    h_split_1 = numpy.hsplit(v_split_R[1],3)
    h_split_2 = numpy.hsplit(v_split_R[2],3)

    b_11 = h_split_0[0]
    b_12 = h_split_0[1]
    b_13 = h_split_0[2]

    b_21 = h_split_1[0]
    b_22 = h_split_1[1]
    b_23 = h_split_1[2]

    b_31 = h_split_2[0]
    b_32 = h_split_2[1]
    b_33 = h_split_2[2]

    split_U = numpy.split(U,3)
    split_V = numpy.split(V,3)

    T = 100
    
    U1 = split_U[0]
    U2 = split_U[1]
    U3 = split_U[2]

    V1 = split_V[0]
    V2 = split_V[1]
    V3 = split_V[2]

    for step in xrange(1000): 

        i = randint(0,5)

        if i == 0:
            U1,V1 = sgd(b_11, U1, V1, 2, T)
            U2,V2 = sgd(b_22, U2, V2, 2, T)
            U3,V3 = sgd(b_33, U3, V3, 2, T)
        if i == 1:
            U1,V2 = sgd(b_12, U1, V2, 2, T)
            U2,V3 = sgd(b_23, U2, V3, 2, T)
            U3,V1 = sgd(b_31, U3, V1, 2, T)
        if i == 2:
            U1,V3 = sgd(b_13, U1, V3, 2, T)
            U2,V1 = sgd(b_21, U2, V1, 2, T)
            U3,V2 = sgd(b_32, U3, V2, 2, T)
        if i == 3:
            U1,V1 = sgd(b_11, U1, V1, 2, T)
            U2,V3 = sgd(b_23, U2, V3, 2, T)
            U3,V2 = sgd(b_32, U3, V2, 2, T)
        if i == 4:
            U1,V2 = sgd(b_12, U1, V2, 2, T)
            U2,V1 = sgd(b_21, U2, V1, 2, T)
            U3,V3 = sgd(b_33, U3, V3, 2, T)
        if i == 5:
            U1,V3 = sgd(b_13, U1, V3, 2, T)
            U2,V2 = sgd(b_22, U2, V2, 2, T)
            U3,V1 = sgd(b_31, U3, V1, 2, T)

    U = numpy.concatenate((U1, U2, U3))
    V = numpy.concatenate((V1, V2, V3))
    
    return U, V

def generate_matrix():
    matrix = numpy.zeros((100,100))

    for i in xrange(100):

        for j in xrange(30):

            item = randint(0,len(matrix)-1)

            rating = 0
                
            while(rating==0):
                rating = randint(0,6)

                matrix[i][item] = rating

    return matrix

def dsgd_2x2(R, U, V):

    v_split_R = numpy.split(R,2)

    h_split_0 = numpy.hsplit(v_split_R[0],2)
    h_split_1 = numpy.hsplit(v_split_R[1],2)

    b_11 = h_split_0[0]
    b_12 = h_split_0[1]

    b_21 = h_split_1[0]
    b_22 = h_split_1[1]

    split_U = numpy.split(U,2)
    split_V = numpy.split(V,2)

    U0 = split_U[0]
    U1 = split_U[1]

    V0 = split_V[0]
    V1 = split_V[1]

    T=100

    for step in xrange(1000):
        i = randint(0,1)

        if i==0:
            U0,V0 = sgd(b_11, U0, V0, 2, T)
            U1,V1 = sgd(b_22, U1, V1, 2, T)
        else:
            U0,V1 = sgd(b_12, U0, V1, 2, T)
            U1,V0 = sgd(b_21, U1, V0, 2, T)

    U = numpy.concatenate((U0, U1))
    V = numpy.concatenate((V0, V1))
    return U, V

###############################################################################

if __name__ == "__main__":

    # matrix m x n
    # R = [
    #      [5,4,0,1,3,0],
    #      [5,0,4,0,0,1],
    #      [0,5,0,1,1,0],
    #      [1,0,1,5,4,0],
    #      [0,1,0,0,5,4],
    #      [1,0,2,5,0,0]
    #     ]

    R = numpy.loadtxt(open("/home/arthur/projects/mestrado/bigdata/foursquare/NY_MATRIX","rb"),delimiter=",")

    R = numpy.array(R)


    # R = generate_matrix()

    # users
    M = len(R)

    # itens
    N = len(R[0])

    K = 10

    U = numpy.random.rand(M,K)
    V = numpy.random.rand(N,K)

    U0 = numpy.copy(U);
    V0 = numpy.copy(V);

    U1 = numpy.copy(U); 
    V1 = numpy.copy(V);

    U2 = numpy.copy(U);
    V2 = numpy.copy(V);

    nP0, nQ0 = sgd(R, U0, V0)
    nR0 = numpy.dot(nP0, nQ0.T)
    print rmse(nR0,R)

    nP1, nQ1 = matrix_factorization_2(R, U1, V1, K)
    nR1 = numpy.dot(nP1, nQ1.T)
    print rmse(nR1,R)

    # nP2, nQ2 = dsgd_3x3(R, U2, V2)
    # nR2 = numpy.dot(nP2, nQ2.T)
    # print rmse(nR2,R)
   

    
