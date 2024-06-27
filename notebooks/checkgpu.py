import tensorflow as tf

# Check the number of GPUs available
num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", num_gpus)

# List all physical devices
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Physical GPUs: ", gpus)

# Run a small computation to verify GPU usage
if num_gpus > 0:
    print("GPU is available")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("Matrix multiplication result:\n", c)
else:
    print("GPU is not available")