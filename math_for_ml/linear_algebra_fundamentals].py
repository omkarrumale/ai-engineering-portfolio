
import numpy as np

# # -----VECTOR OPERATIONS-----
u=np.array([3,5,7,2])
v=np.array([1,4,6,8])

# Addition of u and v
add_result=u+v
print(f"Addition u+v = {add_result}")

# Subtraction of u-v
sub_result=u-v
print(f"Subtraction u-v = {sub_result}")

# Dot product of u and v
dot_result=np.dot(u,v)
print(f"Dot product u-v = {dot_result}")

# Euclidean distance between u and v
euc_dist = np.sqrt(np.sum((u-v)**2))
print(f"Euclidean distance u and v = {euc_dist}")

#-----MATRIX OPERATIONS-----

A=np.array([[2, 3, 1],
     [4, 1, 5],
     [3, 2, 4]])

B = np.array([[1, 2],
     [3, 4],
     [5, 6]])

# # Getting shape of A nd B to understande their dimensions 
print(f"Shape of A = {A.shape}")
print(f"Shape of B = {B.shape}")

# Matrix multiplication of A and B and their shape
multiply= A @ B
print(f" A @ B = {multiply}")
print(multiply.shape)

# Transpose of A and its shape
transpose=(A.T)
print(f"A.T = {transpose}")
print(transpose.shape)

# Finding determinant of A
det=np.linalg.det(A)
print(f"Determinant of A = {det}")

# Taking inverse of A
A_inv=np.linalg.inv(A)
print(f"Inverse of A = {A_inv}")

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
max_idx = np.argmax(eigenvalue)
print(f"Largest eigenvalue: {eigenvalue[max_idx]:.2f} — carries most information")

# Dataset of 5 students, 3 features each:
data = np.array([[85, 92, 78],
        [90, 88, 95],
        [78, 85, 80],
        [92, 95, 88],
        [88, 79, 92]])

print(data.shape)

# Accessing 3rd student second feature
print(data[2][1])

# All student first features
print(data[:,1])

# Finding out mean of each features to analyze the mean of all featurs 
feature_means = np.mean(data, axis=0) 
# mean of each column = mean of each feature
print(f"The mean of  each features is {feature_means}")
