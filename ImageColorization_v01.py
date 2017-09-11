import tensorflow as tf
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, s):
  return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')


def convolutional_neural_network(x):#, keep_rate):
	weights = {
	    # block1 
	    'W_conv_b1_1': weight_variable([3, 3, 1, 64]),
	    'W_conv_b1_2': weight_variable([3, 3, 64, 64]),
	    # block2
	    'W_conv_b2_1': weight_variable([3, 3, 64, 128]),
	    'W_conv_b2_2': weight_variable([3, 3, 128, 128]),
	    # block3 
	    'W_conv_b3_1': weight_variable([3, 3, 128, 256]),
	    'W_conv_b3_2': weight_variable([3, 3, 256, 256]),
	    'W_conv_b2_3': weight_variable([3, 3, 256, 256]),
	   # block4 
	    'W_conv_b4_1': weight_variable([3, 3, 256, 512]),
	    'W_conv_b4_2': weight_variable([3, 3, 512, 512]),
	    'W_conv_b2_3': weight_variable([3, 3, 512, 512]),
	    # block5 
	    'W_conv_b5_1': weight_variable([3, 3, 512, 512]),
	    'W_conv_b5_2': weight_variable([3, 3, 512, 512]),
	    'W_conv_b2_3': weight_variable([3, 3, 512, 512]),
	    # block6 
	    'W_conv_b6_1': weight_variable([3, 3, 512, 512]),
	    'W_conv_b6_2': weight_variable([3, 3, 512, 512]),
	    'W_conv_b2_3': weight_variable([3, 3, 512, 512]),
	    # block7
	    'W_conv_b7_1': weight_variable([3, 3, 512, 512]),
	    'W_conv_b7_2': weight_variable([3, 3, 512, 512]),
	    'W_conv_b2_3': weight_variable([3, 3, 512, 512]),
	    # block8 
	    'W_conv_b8_1': weight_variable([4, 4, 512, 256]),
	    'W_conv_b8_2': weight_variable([3, 3, 256, 256]),
	    'W_conv_b2_3': weight_variable([3, 3, 256, 256]),
	    'W_conv_b2_4': weight_variable([3, 3, 256, 256]),
	    
	    'out': weight_variable([1, 1, 256, 313])
	}

	biases = {
		# block1
        'b_conv_b1_1': bias_variable([64]),
        'b_conv_b1_2': bias_variable([64]),
        # block2
        'b_conv_b2_1': bias_variable([128]),
        'b_conv_b2_2': bias_variable([128]),
        # block3
        'b_conv_b3_1': bias_variable([256]),
        'b_conv_b3_2': bias_variable([256]),
        'b_conv_b3_3': bias_variable([256]),
        # block4
        'b_conv_b4_1': bias_variable([512]),
        'b_conv_b4_2': bias_variable([512]),
        'b_conv_b4_3': bias_variable([512]),
        # block5
        'b_conv_b5_1': bias_variable([512]),
        'b_conv_b5_2': bias_variable([512]),
        'b_conv_b5_3': bias_variable([512]),
        # block6
        'b_conv_b6_1': bias_variable([512]),
        'b_conv_b6_2': bias_variable([512]),
        'b_conv_b6_3': bias_variable([512]),
        # block7
        'b_conv_b7_1': bias_variable([512]),
        'b_conv_b7_2': bias_variable([512]),
        'b_conv_b7_3': bias_variable([512]),
        # block8
        'b_conv_b8_1': bias_variable([256]),
        'b_conv_b8_2': bias_variable([256]),
        'b_conv_b8_3': bias_variable([256]),
        'b_conv_b8_4': bias_variable([256]),

        'out': bias_variable([313])
        
	}
	# Convolution Layers, using our function
	#block1
	conv_b1_1 = tf.nn.relu(conv2d(x, weights['W_conv_b1_1']) + biases['b_conv_b1_1'],1)
	conv_b1_2 = tf.nn.relu(conv2d(x, weights['W_conv_b1_2']) + biases['b_conv_b1_2'],2)
	#block2
	conv_b2_1 = tf.nn.relu(conv2d(x, weights['W_conv_2_1']) + biases['b_conv_b2_1'],1)
	conv_b2_2 = tf.nn.relu(conv2d(x, weights['W_conv_2_2']) + biases['b_conv_b2_2'],2)
	#block3
	conv_b3_1 = tf.nn.relu(conv2d(x, weights['W_conv_b3_1']) + biases['b_conv_b3_1'],1)
	conv_b3_2 = tf.nn.relu(conv2d(x, weights['W_conv_b3_2']) + biases['b_conv_b3_2'],1)
	conv_b3_3 = tf.nn.relu(conv2d(x, weights['W_conv_b3_3']) + biases['b_conv_b3_3'],2)
	#block4
	conv_b4_1 = tf.nn.relu(conv2d(x, weights['W_conv_b4_1']) + biases['b_conv_b4_1'],1)
	conv_b4_2 = tf.nn.relu(conv2d(x, weights['W_conv_b4_2']) + biases['b_conv_b4_2'],1)
	conv_b4_3 = tf.nn.relu(conv2d(x, weights['W_conv_b4_3']) + biases['b_conv_b4_3'],1)
	#block5
	conv_b5_1 = tf.nn.relu(conv2d(x, weights['W_conv_b5_1']) + biases['b_conv_b5_1'],1)
	conv_b5_2 = tf.nn.relu(conv2d(x, weights['W_conv_b5_2']) + biases['b_conv_b5_2'],1)
	conv_b5_3 = tf.nn.relu(conv2d(x, weights['W_conv_b5_3']) + biases['b_conv_b5_3'],1)
	#block6
	conv_b6_1 = tf.nn.relu(conv2d(x, weights['W_conv_b6_1']) + biases['b_conv_b6_1'],1)
	conv_b6_2 = tf.nn.relu(conv2d(x, weights['W_conv_b6_2']) + biases['b_conv_b6_2'],1)
	conv_b6_3 = tf.nn.relu(conv2d(x, weights['W_conv_b6_3']) + biases['b_conv_b6_3'],1)
	#block7
	conv_b7_1 = tf.nn.relu(conv2d(x, weights['W_conv_b7_1']) + biases['b_conv_b7_1'],1)
	conv_b7_2 = tf.nn.relu(conv2d(x, weights['W_conv_b7_2']) + biases['b_conv_b7_2'],1)
	conv_b7_3 = tf.nn.relu(conv2d(x, weights['W_conv_b7_3']) + biases['b_conv_b7_3'],1)
	#block8
	conv_b8_1 = tf.nn.relu(tf.nn.conv2d_transpose(x,  weights['W_conv_b8_1']) + biases['b_conv_b8_1'], [1,64,64,1], strides=[1, 2, 2, 1], padding='SAME'))
	conv_b8_2 = tf.nn.relu(conv2d(x, weights['W_conv_b8_2']) + biases['b_conv_b8_2'],1)
	conv_b8_3 = tf.nn.relu(conv2d(x, weights['W_conv_b8_3']) + biases['b_conv_b8_3'],1)
	conv_b8_4 = tf.nn.relu(tf.nn.conv2d_transpose(x, weights['W_conv_b8_4']) + biases['b_conv_b8_4'], [1,256,256,1], strides=[1, 4, 4, 1], padding='SAME'))
	
	output = tf.nn.softmax(conv2d(x, weights['out']) + biases['out'],1)
	return output