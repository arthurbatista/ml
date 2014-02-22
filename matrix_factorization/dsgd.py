#!/usr/bin/python
#
# Created by Arthur Batista (2014)
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

def generate_stratus_matrix(R, U, V, block_size, step):
    R = numpy.array(R)

    M = len(R)
    N = len(R[0])

    index_r = []
    index_c = []

    r = M/block_size
    c = N/block_size

    if M%block_size == 0:
        while len(index_r) < block_size:
            index_r.append(M/block_size)
    else:
        r = int(r)
        while len(index_r) < block_size:
            if len(index_r) + 1 == block_size:
                index_r.append(M - r*(block_size - 1))
            else:
                index_r.append(r)

    if N%block_size == 0:
        while len(index_c) < block_size:
            index_c.append(N/block_size)
    else:
        c = int(c)
        while len(index_c) < block_size:
            if len(index_c) + 1 == block_size:
                index_c.append(N - c*(block_size - 1))
            else:
                index_c.append(c)

    index_stratus = numpy.zeros( (block_size,block_size) )

    pointer_r = 0
    pointer_c = 0

    for i in xrange(block_size):
        for j in xrange(block_size):
            index_stratus[i][j] = float(`pointer_r`+'.'+`pointer_c`)
            pointer_c += index_c[j]
        pointer_c = 0
        pointer_r += index_r[i]

    index_stratus_selected = []
    index_stratus_selected.append(randint(0,block_size-1))
    while len(index_stratus_selected) < block_size:
        index = randint(0,block_size-1)
        valid = True
        for i in xrange(len(index_stratus_selected)):
            if index_stratus_selected[i] == index:
                valid = False
                break
        if valid:
            index_stratus_selected.append(index)

    list_stratus = []

    for k in xrange(block_size):
        
        pointer_r, pointer_c = str(index_stratus[k][index_stratus_selected[k]]).split('.')

        pointer_r = int(pointer_r)
        pointer_c = int(pointer_c)
        
        if(M>=N):
            total_r = index_r[index_stratus_selected[k]]
            total_c = index_c[index_stratus_selected[k]]
        else:
            total_r = index_r[k]
            total_c = index_c[k]

        stratus = numpy.zeros( (total_r,total_c) )

        for i in xrange(total_r):
            for j in xrange(total_c):
                stratus[i][j] = R[pointer_r][pointer_c]
                pointer_c += 1
            pointer_c = pointer_c - total_c
            pointer_r += 1

        list_stratus.append(stratus)

    index_to_split = numpy.zeros((3))

    for i in xrange(len(index_r)):
        if i == 0:
            index_to_split[i] = index_r[i]
        else:
            index_to_split[i] = index_r[i] + index_to_split[i-1]

    temp_V = numpy.split(V,index_to_split);

    reorded_V = []

    for i in xrange(len(index_c)):
        reorded_V.append(temp_V[index_stratus_selected[i]])

    return list_stratus, numpy.split(U,index_to_split), reorded_V, index_stratus_selected

def rmse(predictions, R):
    rmse = 0
    for i in xrange(len(R)):
        for j in xrange(len(R[i])):
            if R[i][j] > 0:
                rmse += numpy.sqrt(((predictions[i][j] - R[i][j]) ** 2).mean())

    return rmse

def sgd(R, U, V, list_index, steps=1800000, alpha=0.0002, lamb=0.02):

    len_list_index = len(list_index)

    for step in xrange(steps):

        index = randint(0,len_list_index-1)

        sI,sJ =  list_index[index].split(',')

        i = int(sI)
        j = int(sJ)

        if R[i][j] > 0:
            Ui = U[i,:]
            Vj = V[j,:]
            Ra = numpy.dot(Ui.T,Vj) - R[i][j]
            u_temp = Ui - 2*alpha * ( (Ra * Vj) + (lamb * Ui) )
            V[j]   = Vj - 2*alpha * ( (Ra * Ui) + (lamb * Vj) )
            U[i]   = u_temp

    return U, V

def dsgd(R, U, V):

    T=100
    block_size = 3

    for step in xrange(1000):

        list_stratus, list_U, list_V, index_stratus_selected = generate_stratus_matrix(R, U, V, block_size, step)

        for i in xrange(block_size):

            compressed_R = load_index_array(list_stratus[i])

            list_U[i],list_V[i] = sgd(list_stratus[i], list_U[i], list_V[i], compressed_R, T)

        index_U=0
        for index_array in xrange(block_size):
            temp_U = list_U[index_array]
            
            for i in xrange(len(temp_U)):
                for j in xrange(len(temp_U[0])):
                    U[index_U][j] = temp_U[i][j]
                index_U += 1

        index_V=0
        for x in xrange(block_size):
            temp_V = list_V[x]
            index_array = index_stratus_selected[x] * len(temp_V)

            for i in xrange(len(temp_V)):
                for j in xrange(len(temp_V[0])):
                    V[index_array + i][j] = temp_V[i][j]

    return U, V

def load_index_array(R):
    list_index = []
    for i in xrange(len(R)):
        for j in xrange(len(R[0])):
            if R[i][j] > 0:
                list_index.append(`i`+','+`j`)
    return list_index

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

    # R = numpy.loadtxt(open("../dataset/NY_MATRIX","rb"),delimiter=",")

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

    nP0, nQ0 = dsgd(R, U0, V0)
    nR0 = numpy.dot(nP0, nQ0.T)
    print rmse(nR0,R)

    nP1, nQ1 = sgd(R, U1, V1,load_index_array(R))
    nR1 = numpy.dot(nP1, nQ1.T)
    print rmse(nR1,R)
   

    
