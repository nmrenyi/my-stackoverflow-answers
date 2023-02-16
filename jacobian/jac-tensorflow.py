# question: https://stackoverflow.com/questions/74171994/diagonal-or-divergence-of-jacobian-matrix/75467342#75467342
# solution from ChatGPT
import tensorflow as tf


# Define the function
def func(x):
    return tf.stack([tf.math.pow(x[0], 2) + tf.math.pow(x[1], 3), tf.math.pow(x[1], 2) + tf.math.pow(x[0], 3)])

# Define the inputs
x = tf.constant([1.0, 2.0])

# Create a GradientTape context to trace the operations
with tf.GradientTape() as tape:
    # Watch the input tensor
    tape.watch(x)
    # Evaluate the function
    y = func(x)

# Compute the Jacobian matrix of y with respect to x
jacobian = tape.jacobian(y, x)

print(jacobian)
