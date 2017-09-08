import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print("dataset loaded")
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

classes = 10

X = tf.placeholder(tf.int32,shape=[None,784])
Y = tf.placeholder(tf.int32,shape=[None,10])
X = tf.reshape(X,[-1,28,28])
k1 = weight_variable([3,3,1,3])
b1 = bias_variable([3])

k2 = weight_variable([4,4,3,5])
b2 = bias_variable([5])

#layer 1
h_conv1 = tf.nn.relu(conv2d(X, k1) + b1)
h_pool1 = max_pool_2x2(h_conv1)

#layer 2

h_conv2 = tf.nn.relu(conv2d(h_pool1, k2) + b2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([28*28*5, 100])
b_fc1 = bias_variable([100])

h_pool2_flat = tf.reshape(h_pool2, [-1, 28*28*5])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

output_w = weight_variable([100,10])
output_b = bias_variable([10])
output_z = tf.add(tf.matmul(h_fc1,output_w),output_b)
output_a = tf.nn.relu(output_z)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_z))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
correct_prediction = tf.equal(tf.argmax(output_a, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(1)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1]})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    break

  print('test accuracy %g' % accuracy.eval(feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))
