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

    print U

    for step in xrange(1): 

        # i = randint(0,5)
        i = 2

        # if i == 0:
        #     U1,V1 = sgd(b_11, U1, V1, T)
        #     U2,V2 = sgd(b_22, U2, V2, T)
        #     U3,V3 = sgd(b_33, U3, V3, T)

        # if i == 1:
        #     U1,V2 = sgd(b_12, U1, V2, T)
        #     U2,V3 = sgd(b_23, U2, V3, T)
        #     U3,V1 = sgd(b_31, U3, V1, T)
        if i == 2:
            U1,V3 = sgd(b_13, U1, V3, T)
            U2,V1 = sgd(b_21, U2, V1, T)
            U3,V2 = sgd(b_32, U3, V2, T)

        # if i == 3:
        #     U1,V1 = sgd(b_11, U1, V1, T)
        #     U2,V3 = sgd(b_23, U2, V3, T)
        #     U3,V2 = sgd(b_32, U3, V2, T)
        # if i == 4:
        #     U1,V2 = sgd(b_12, U1, V2, T)
        #     U2,V1 = sgd(b_21, U2, V1, T)
        #     U3,V3 = sgd(b_33, U3, V3, T)
        # if i == 5:
        #     U1,V3 = sgd(b_13, U1, V3, T)
        #     U2,V2 = sgd(b_22, U2, V2, T)
        #     U3,V1 = sgd(b_31, U3, V1, T)

    print U
    
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

def generate_stratus_matrix(R, U, V, block_size):
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

    # print '--------------'
    # print index_stratus_selected
    # print numpy.split(U,index_to_split)
    # print reorded_V
    # print '--------------'

    return list_stratus, numpy.split(U,index_to_split), reorded_V


def rmse(predictions, R):
    rmse = 0
    for i in xrange(len(R)):
        for j in xrange(len(R[i])):
            if R[i][j] > 0:
                rmse += numpy.sqrt(((predictions[i][j] - R[i][j]) ** 2).mean())

    return rmse

# def sgd(R, U, V, list_index, steps=1800000, alpha=0.0002, lamb=0.02):

    # len_list_index = len(list_index)

    # for step in xrange(steps):

    #     index = randint(0,len_list_index-1)

    #     sI,sJ =  list_index[index].split(',')

    #     i = int(sI)
    #     j = int(sJ)

    #     if R[i][j] > 0:
    #         Ui = U[i,:]
    #         Vj = V[j,:]
    #         Ra = numpy.dot(Ui.T,Vj) - R[i][j]
    #         u_temp = Ui - 2*alpha * ( (Ra * Vj) + (lamb * Ui) )
    #         V[j]   = Vj - 2*alpha * ( (Ra * Ui) + (lamb * Vj) )
    #         U[i]   = u_temp

    # return U, V


def sgd(R, U, V, steps=180000, alpha=0.0002, lamb=0.02 ,beta=0.02):
    for step in xrange(steps):

        i = randint(0,len(U)-1)
        j = randint(0,len(V)-1)

        if R[i][j] > 0:
            Ui = U[i,:]
            Vj = V[j,:]
            Ra = R[i][j] - numpy.dot(Ui.T,Vj)
            u_temp = Ui + 2*alpha * ( (Ra * Vj) - (lamb/len(U) * Ui) )
            V[j] = Vj + 2*alpha * ( (Ra * Ui) - (lamb/len(V) * Vj) )
            U[i] = u_temp

    return U, V

def gd(R, U, V, list_index, steps=5000, alpha=0.0002, lamb=0.02):

    len_list_index = len(list_index)

    for step in xrange(5000):
        for index in xrange(len_list_index):

        # index = randint(0,len_list_index-1)

        # i = randint(0,len(U)-1)
        # j = randint(0,len(V)-1)

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

def dsgd(R, U, V):

    T=100
    block_size = 3

    for step in xrange(1000): 

        list_stratus, list_U, list_V = generate_stratus_matrix(R, U, V, block_size)

        for i in xrange(block_size):

            compressed_R = load_index_array(list_stratus[i])

            list_U[i],list_V[i] = sgd(list_stratus[i], list_U[i], list_V[i], T)


        index_U=0
        for index_array in xrange(block_size):
            temp_U = list_U[index_array]
            
            for i in xrange(2):
                for j in xrange(2):
                    U[index_U][j] = temp_U[i][j]
                index_U += 1

        index_V=0
        for index_array in xrange(block_size):
            temp_V = list_V[index_array]
            
            for i in xrange(2):
                for j in xrange(2):
                    V[index_V][j] = temp_V[i][j]
                index_V += 1


        # print U
        # print V

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


    # R = generate_matrix()

    # users
    M = len(R)

    # itens
    N = len(R[0])

    # for Y in xrange(20):
    # K = 2

    # list_index = load_index_array(R)

        # K = randint(4,11)
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

    # print nP0
    # print nQ0

    print '#######################'

    # U2 = numpy.copy(U);
    # V2 = numpy.copy(V);


    nP1, nQ1 = dsgd_3x3(R, U1, V1)
    nR1 = numpy.dot(nP1, nQ1.T)
    print rmse(nR1,R)
    # print nP1
    # print nQ1

    # nP2, nQ2 = dsgd_3x3(R, U2, V2)
    # nR2 = numpy.dot(nP2, nQ2.T)
    # print rmse(nR2,R)
   

    
