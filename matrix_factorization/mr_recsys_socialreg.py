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

def rmse(predictions, R):
    rmse = 0
    for i in xrange(len(R)):
        for j in xrange(len(R[i])):
            if R[i][j] > 0:
                rmse += numpy.sqrt(((predictions[i][j] - R[i][j]) ** 2).mean())

    return rmse

def matrix_factorization(R, U, V, K=2, steps=5000, alpha=0.0002, lamb=0.02 ,beta=0.02):
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    Ui = U[i,:]
                    Vj = V[j,:]
                    Ra =  numpy.dot(Ui.T,Vj) - R[i][j]
                    for k in xrange(K):
                        U[i][k] = Ui[k] - alpha * ( (Ra * Vj[k]) + (lamb * Ui[k]) )
                        V[j][k] = Vj[k] - alpha * ( (Ra * Ui[k]) + (lamb * Vj[k]) )

    return U, V

def split_R(R):

    v_split_R = numpy.split(R,3)
    
    h_split_1 = numpy.hsplit(v_split_R[0],3)
    h_split_2 = numpy.hsplit(v_split_R[0],3)
    h_split_3 = numpy.hsplit(v_split_R[0],3)

    print h_split_1


def dsgd(R, U, V):

    #Verify if it is a quadratic matrix
    # if len(R)!=len(R[i]):
    #     return False

    #R size must to be divided by block_size
    # if len(R)%block_size!=0
        # return False

    v_split_R = numpy.split(R,3)
    
    h_split_0 = numpy.hsplit(v_split_R[0],3)
    h_split_1 = numpy.hsplit(v_split_R[0],3)
    h_split_2 = numpy.hsplit(v_split_R[0],3)

    split_U = numpy.split(U,3)
    split_V = numpy.split(V,3)

    ############ 1st Pass ############

    T = 5
    alpha = 0.0002

    U0 = split_U[0]
    U1 = split_U[1]
    U2 = split_U[2]

    V0 = split_V[0]
    V1 = split_V[1]
    V2 = split_V[2]

    for step in xrange(5000): 

        U0,V0 = matrix_factorization(h_split_0[0], U0, V0, 2, T, alpha)
        U1,V1 = matrix_factorization(h_split_1[1], U1, V1, 2, T, alpha)
        U2,V2 = matrix_factorization(h_split_2[2], U2, V2, 2, T, alpha)

        U1,V2 = matrix_factorization(h_split_1[2], U1, V2, 2, T, alpha)
        U0,V1 = matrix_factorization(h_split_0[1], U0, V1, 2, T, alpha)
        U2,V0 = matrix_factorization(h_split_2[0], U2, V0, 2, T, alpha)


        U1,V0 = matrix_factorization(h_split_1[0], U1, V0, 2, T, alpha)
        U2,V1 = matrix_factorization(h_split_2[1], U2, V1, 2, T, alpha)
        U0,V2 = matrix_factorization(h_split_0[2], U0, V2, 2, T, alpha)

        U1,V2 = matrix_factorization(h_split_1[2], U1, V2, 2, T, alpha)
        U0,V0 = matrix_factorization(h_split_0[0], U0, V0, 2, T, alpha)
        U2,V1 = matrix_factorization(h_split_2[1], U2, V1, 2, T, alpha)

        U2,V2 = matrix_factorization(h_split_2[2], U2, V2, 2, T, alpha)
        U1,V0 = matrix_factorization(h_split_1[0], U1, V0, 2, T, alpha)
        U0,V1 = matrix_factorization(h_split_0[1], U0, V1, 2, T, alpha)
        
        U0,V2 = matrix_factorization(h_split_0[2], U0, V2, 2, T, alpha)
        U1,V1 = matrix_factorization(h_split_1[1], U1, V1, 2, T, alpha)
        U2,V0 = matrix_factorization(h_split_2[0], U2, V0, 2, T, alpha)

    U = numpy.concatenate((U0, U1, U2))
    V = numpy.concatenate((V0, V1, V2))

    return U,V

def dsgd1(R, U, V):

    #Verify if it is a quadratic matrix
    # if len(R)!=len(R[i]):
    #     return False

    #R size must to be divided by block_size
    # if len(R)%block_size!=0
        # return False

    v_split_R = numpy.split(R,2)
    
    h_split_0 = numpy.hsplit(v_split_R[0],2)
    h_split_1 = numpy.hsplit(v_split_R[0],2)
    

    split_U = numpy.split(U,2)
    split_V = numpy.split(V,2)

    ############ 1st Pass ############

    T = 5
    alpha = 0.0002

    U0 = split_U[0]
    U1 = split_U[1]
    
    V0 = split_V[0]
    V1 = split_V[1]
    
    for step in xrange(5000): 

        U0,V0 = matrix_factorization(h_split_0[0], U0, V0, 2, T, alpha)
        U1,V1 = matrix_factorization(h_split_1[1], U1, V1, 2, T, alpha)
       
        U0,V1 = matrix_factorization(h_split_0[1], U0, V1, 2, T, alpha)
        U1,V0 = matrix_factorization(h_split_1[0], U1, V0, 2, T, alpha)

    U = numpy.concatenate((U0, U1, U2))
    V = numpy.concatenate((V0, V1, V2))

    return U,V


###############################################################################

if __name__ == "__main__":

    # matrix m x n
    R = [
         [5,4,0,1,3,0],
         [5,0,4,0,0,1],
         [0,5,0,1,1,0],
         [1,0,1,5,4,0],
         [0,1,0,0,5,4],
         [1,0,2,5,0,0]
        ]

    R = numpy.array(R)

    # users
    M = len(R)

    # itens
    N = len(R[0])

    K = 2

    U = numpy.random.rand(M,K)
    V = numpy.random.rand(N,K)

    U0 = numpy.copy(U);
    V0 = numpy.copy(V);

    U1 = numpy.copy(U);
    V1 = numpy.copy(V);

    U2 = numpy.copy(U);
    V2 = numpy.copy(V);

    nP0, nQ0 = matrix_factorization(R, U0, V0, K, 5000, 0.0002)
    # print nP0
    # print nQ0
    # print '############################'

    nP1, nQ1 = dsgd(R, U1, V1)
    # print nP1
    # print nQ1
    # print '############################'

    nP2, nQ2 = dsgd1(R, U2, V2)

    nR0 = numpy.dot(nP0, nQ0.T)
    print rmse(nR0,R)

    nR1 = numpy.dot(nP1, nQ1.T)
    print rmse(nR1,R)

    nR2 = numpy.dot(nP2, nQ2.T)
    print rmse(nR2,R)
    



   