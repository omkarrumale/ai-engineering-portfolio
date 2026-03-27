import numpy as np

# # -----VECTOR OPERATIONS-----
u=np.array([3,5,7,2])
v=np.array([1,4,6,8])

# Addition of u and v
add_result=u+v
print(add_result)

# Subtraction of u-v
sub_result=u-v
print(sub_result)

# Dot product of u and v
dot_result=np.dot(u,v)
print(dot_result)

# Euclidean distance between u and v
Euc_dist=np.sqrt(np.sum(u-v)**2)
print(Euc_dist)

#-----MATRIX OPERATIONS-----

A=np.array([[2, 3, 1],
     [4, 1, 5],
     [3, 2, 4]])

B = np.array([[1, 2],
     [3, 4],
     [5, 6]])

# # Getting shape of A nd B to understande their dimensions 
print(A.shape)
print(B.shape)

# Matrix multiplication of A and B and their shape
multiply= A @ B
print(multiply)
print(multiply.shape)

# Transpose of A and its shape
transpose=(A.T)
print(transpose)
print(transpose.shape)

# Finding determinant of A
det=np.linalg.det(A)
print(det)

# Taking inverse of A
A_inv=np.linalg.inv(A)
print(A_inv)

# Verifying A @ A_inv = I (identity matrix)-using .round to round off the decimals
identity= A @ A_inv
print(identity.round(0))

# -----Eigenvalues-----

C = np.array([[4, 2],
     [1, 3]])

#Finding eigenvalue and eigenvectors-using .round to round off the decimals
eigenvalue,eigenvector = np.linalg.eig(C)
print(eigenvalue)
print(eigenvector.round(0))

# Verifying det(C) = product eigenvalue
det_c = np.linalg.det(C)
x = (det_c.round(1))
y = (eigenvalue.prod().round(1))

if x == y:
    print(f"Determinant of C = {x} 'EQUALS' to the product of eigenvalue = {y}")
else:
    print('They are not equal')

#Checking which eigen value is larger and what does it mean?
print(f"Eigenvalues are {eigenvalue}")
print(f"The eigen values are {eigenvalue} hence 5. is the larger value,it means that the 5 contain more information than 2 and the direction is stretched by 5 in the positive x direction.")


# Dataset of 5 students, 3 features each:
data = np.array([[85, 92, 78],
        [90, 88, 95],
        [78, 85, 80],
        [92, 95, 88],
        [88, 79, 92]])

print(data.shape)

# Accessing 3rd student second feature
print(data[(3),(2)])

# All student first features
print(data[:,1])

# Finding out mean of each features to analyze the mean of all featurs 
row_0=np.mean(data[0])
row_1=np.mean(data[1])
row_2=np.mean(data[2])
row_3=np.mean(data[3])
row_4=np.mean(data[4])

print(f"The mean of features of\nRow 0 = {row_0}\nRow 1 = {row_1}\nRow 2 = {row_2}\nRow 3 = {row_3.round(2)}\nRow 4 = {row_4.round(2)}")
