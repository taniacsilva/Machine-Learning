import numpy as np

###vector_vector_multiplication
u = np.array([2, 4, 5, 6])
v = np.array([1, 0, 0, 2])

u.dot(v) 

#write the function from scratch
def vector_vector_multiplication(u, v):
    assert u.shape[0]==v.shape[0]
    
    n=u.shape[0]
    result=0.0
    
    for i in range(n):
        result= result + u[i] * v[i]
    return result

vector_vector_multiplication(u, v)

###matrix_vector_multiplication

U=np.array([[2, 4, 5, 6], [1, 2, 1, 2], [3, 1, 2, 1]])

U.dot(v)

#write the function from scratch
def matrix_vector_multiplication(U,v):
    assert U.shape[1] == v.shape[0]

    num_rows=U.shape[0]

    result=np.zeros(num_rows)

    for i in range(num_rows):
        result[i] = vector_vector_multiplication(U[i],v)

    return result

matrix_vector_multiplication(U,v)

###matrix_matrix_multiplication

V=np.array([[1, 1, 2], [0, 0.5, 1], [0, 2, 1], [2, 1, 0 ]])

U.dot(V)

#write the function from scratch
def matrix_matrix_multiplication(U,V):
    assert U.shape[1] == V.shape[0]

    num_rows= U.shape[0]
    num_cols=V.shape[1]
    result=np.zeros((num_rows, num_cols))

    for i in range(num_cols):
        vi = V[:, i]
        Uvi=matrix_vector_multiplication(U,vi)
        result[:, i] = Uvi

    return result

matrix_matrix_multiplication(U,V)  

#Identity Matrix
I=np.eye(3)
V.dot(I)

#Matrix Inverse
Vs=V[[0,1,3]]

Vs_inv=np.linalg.inv(Vs)

Vs_inv.dot(Vs)

