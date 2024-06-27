# Tensor : Multidimensional array that can run on GPU
# Rank 0 Tensor : Scalar, Rank 1 Tensor : Vector, Rank 2 Tensor : Matrix, Rank 3 Tensor : Cube
import tensorflow as tf
 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow debug messages
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Initialization of Tensors
x = tf.constant(4)  # Scalar tensor
print(x)  # Output: tf.Tensor(4, shape=(), dtype=int32)

y = tf.constant(4, shape=(1, 1), dtype=tf.float32)  # (1,1) matrix which is essentially a scalar
print(y)  # Output: tf.Tensor([[4.]], shape=(1, 1), dtype=float32)

x1 = tf.constant([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
print(x1)  # Output: tf.Tensor([[1 2 3] [4 5 6]], shape=(2, 3), dtype=int32)

x2 = tf.ones((3, 3))  # 3x3 matrix of ones
print(x2)  # Output: 3x3 matrix of ones

x3 = tf.zeros((2, 3))  # 2x3 matrix of zeros
print(x3)  # Output: 2x3 matrix of zeros

# Identity Matrix
x4 = tf.eye(3)  # 3x3 identity matrix
print(x4)  # Output: 3x3 identity matrix

# Normal Distribution
x5 = tf.random.normal((3, 3), mean=0, stddev=1)  # 3x3 matrix with normal distribution (mean=0, stddev=1)
print(x5)  # Output: 3x3 matrix with normal distribution values

# Uniform Distribution
x6 = tf.random.uniform((1, 3), minval=0, maxval=1)  # 1x3 matrix with uniform distribution (values between 0 and 1)
print(x6)  # Output: 1x3 matrix with uniform distribution values

# Range Function
x7 = tf.range(start=1, limit=19, delta=2)  # Tensor with values starting at 1, ending before 19, with step size of 2
print(x7)  # Output: Tensor with values [1, 3, 5, 7, 9, 11, 13, 15, 17]

# Type Casting to different data types
x7 = tf.cast(x7, dtype=tf.float64)  # Cast x7 to float64
print(x7)  # Output: Tensor with values casted to float64

# Math Operations

# Create two vectors
y1 = tf.constant([1, 2, 3])
y2 = tf.constant([9, 8, 7])

# To add them
y3 = tf.add(y1, y2)  # Add y1 and y2
print(y3)  # Output: Tensor with element-wise addition

# To subtract them
y4 = tf.subtract(y1, y2)  # Subtract y2 from y1
print(y4)  # Output: Tensor with element-wise subtraction

# To divide them
y5 = tf.divide(y1, y2)  # Divide y1 by y2 element-wise
print(y5)  # Output: Tensor with element-wise division

# To multiply them
y6 = tf.multiply(y1, y2)  # Multiply y1 and y2 element-wise
print(y6)  # Output: Tensor with element-wise multiplication

# Element-wise multiplication + Summation = Dot Product 
y7 = tf.tensordot(y1, y2, axes=1)  # Compute the dot product of y1 and y2
print(y7)  # Output: Scalar value of dot product

# Exponential of every element
y8 = y1 ** 5  # Raise each element in y1 to the power of 5
print(y8)  # Output: Tensor with elements raised to the power of 5

# Matrix Multiply
x = tf.random.normal((2, 3))  # Random 2x3 matrix with normal distribution
y = tf.random.normal((3, 4))  # Random 3x4 matrix with normal distribution
z = tf.matmul(x, y)  # Matrix multiplication of x and y
print(z)  # Output: Resultant 2x4 matrix after matrix multiplication
z = x @ y  # Alternative way to perform matrix multiplication using @ operator
print(z)  # Output: Resultant 2x4 matrix after matrix multiplication

# --------------------------------------------------------------------------------------

import tensorflow as tf

# Creating a 1D tensor
x = tf.constant([0, 1, 2, 4, 5, 6, 22, 3, 13])

# Basic Indexing and Slicing
print(x[:])       # Prints all elements: [0, 1, 2, 4, 5, 6, 22, 3, 13]
# x[:]: This format retrieves all elements of the tensor x.

print(x[1:])      # Prints everything except the first element: [1, 2, 4, 5, 6, 22, 3, 13]
# x[start:]: This retrieves elements starting from index start to the end of the tensor.

print(x[1:3])     # Prints elements from index 1 to 2 (non-inclusive of index 3): [1, 2]
# x[start:stop]: This retrieves elements from index start to stop - 1.

print(x[::2])     # Prints every second element: [0, 2, 5, 22, 13]
# x[start:stop:step]: This retrieves elements from index start to stop - 1, stepping by step.

print(x[::-1])    # Prints all elements in reverse order: [13, 3, 22, 6, 5, 4, 2, 1, 0]
# x[::-1]: This retrieves all elements in reverse order.

# Using tf.gather for specific indices
indices = tf.constant([0, 3])
x_ind = tf.gather(x, indices)
print(x_ind)      # Prints elements at index 0 and 3: [0, 4]

# Creating a 2D tensor (matrix)
x = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])  # 3x2 matrix

# Indexing in 2D tensors
print(x[0, :])    # Prints the first row: [1, 2]
# x[dim1, dim2]: Retrieves elements based on the indices specified for each dimension.
# Here, x[0, :] means "retrieve the first row (dim1=0) and all columns (dim2=:)".

print(x[0:2, :])  # Prints the first two rows: [[1, 2], [3, 4]]
# x[dim1_start:dim1_stop, dim2_start:dim2_stop]: Retrieves a sub-tensor based on slicing for each dimension.
# Here, x[0:2, :] means "retrieve rows from index 0 to 1 (non-inclusive of 2) and all columns".

# Additional examples with 2D tensor
print(x[:, 0])    # Prints the first column: [1, 3, 5]
# x[:, dim2]: Retrieves all rows (dim1=:) and the specified column (dim2=0).

print(x[:, 1])    # Prints the second column: [2, 4, 6]
# x[:, dim2]: Retrieves all rows (dim1=:) and the specified column (dim2=1).

print(x[-1, :])   # Prints the last row: [5, 6]
# x[dim1, dim2]: Retrieves elements based on the indices specified for each dimension.
# Here, x[-1, :] means "retrieve the last row (dim1=-1) and all columns (dim2=:)".

print(x[:, -1])   # Prints the last column: [2, 4, 6]
# x[:, dim2]: Retrieves all rows (dim1=:) and the specified column (dim2=-1).

# Creating a 3D tensor (cube)
y = tf.constant([[[1, 2, 3], 
                  [4, 5, 6]],
                 [[7, 8, 9], 
                  [10, 11, 12]]])  # 2x2x3 tensor

# Indexing in 3D tensors
print(y[0, :, :])      # Prints the first 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
# y[dim1, dim2, dim3]: Retrieves elements based on the indices specified for each dimension.
# Here, y[0, :, :] means "retrieve the first 2x3 matrix (dim1=0), all rows (dim2=:) of this matrix, and all columns (dim3=:)".

print(y[:, 0, :])      # Prints the first row of each 2x3 matrix: [[1, 2, 3], [7, 8, 9]]
# y[:, dim2, dim3]: Retrieves elements from all matrices (dim1=:), the first row (dim2=0) of each matrix, and all columns (dim3=:) of these rows.

print(y[:, :, 0])      # Prints the first column of each 2x3 matrix: [[1, 4], [7, 10]]
# y[:, dim2, dim3]: Retrieves elements from all matrices (dim1=:), all rows (dim2=:) of these matrices, and the first column (dim3=0) of these rows.

print(y[0, 0, :])      # Prints the first row of the first matrix: [1, 2, 3]
# y[dim1, dim2, dim3]: Retrieves elements from the first matrix (dim1=0), the first row (dim2=0) of this matrix, and all columns (dim3=:) of this row.

print(y[1, :, -1])     # Prints the last column of the second matrix: [9, 12]
# y[dim1, dim2, dim3]: Retrieves elements from the second matrix (dim1=1), all rows (dim2=:) of this matrix, and the last column (dim3=-1) of these rows.

# Using tf.gather on 2D tensor
indices = tf.constant([0, 2])
x_gather = tf.gather(x, indices)
print(x_gather)   # Prints the first and third rows: [[1, 2], [5, 6]]

# Using tf.gather_nd for more complex indexing
indices_nd = tf.constant([[0, 0], [1, 1], [2, 0]])
x_gather_nd = tf.gather_nd(x, indices_nd)
print(x_gather_nd)  # Prints elements at specific indices: [1, 4, 5]


#------------------------------------------------------------------------------

# Reshaping Tensor
x = tf.range(9)
print(x)  # Prints the original tensor: [0 1 2 3 4 5 6 7 8]

# Reshape into a 3x3 matrix
x = tf.reshape(x, (3, 3))
print(x)  
# Prints the reshaped tensor:
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

# Transpose the matrix
x = tf.transpose(x, perm=[1, 0])  # Swaps the axes
print(x)  

# Prints the transposed tensor:
# [[0 3 6]
#  [1 4 7]
#  [2 5 8]]

# Additional examples with reshaping
y = tf.range(24)
print(y)  # Prints the original tensor: [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]

# Reshape into a 2x3x4 tensor
y = tf.reshape(y, (2, 3, 4))
print(y)  

# Prints the reshaped tensor:
# [[[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]
#  [[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]]

# Transpose the 3D tensor
y = tf.transpose(y, perm=[0, 2, 1])  # Swaps the last two axes
print(y) 

# Prints the transposed tensor:
# [[[ 0  4  8]
#   [ 1  5  9]
#   [ 2  6 10]
#   [ 3  7 11]]
#  [[12 16 20]
#   [13 17 21]
#   [14 18 22]
#   [15 19 23]]]