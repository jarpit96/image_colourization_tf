import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


batch_size = 100
n_classes = 10 # MNIST total classes (0-9 digits)


# tf Graph input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, n_classes])


def conv2d(x, W):
  	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convolutional_neural_network(x):#, keep_rate):
	weights = {
	    # 3 x 3 convolution, 1 input image, 3 outputs
	    'W_conv1': tf.Variable(tf.random_normal([3, 3, 1, 3])),
	    # 4x4 conv, 3 inputs, 5 outputs 
	    'W_conv2': tf.Variable(tf.random_normal([4, 4, 3, 5])),
	    # fully connected, 28*28*5 inputs, 100 outputs
	    'W_fc': tf.Variable(tf.random_normal([7*7*5, 100])),
	    # 100 inputs, 10 outputs (class prediction)
	    'out': tf.Variable(tf.random_normal([100, n_classes]))
	}

	biases = {
	    'b_conv1': tf.Variable(tf.random_normal([3])),
	    'b_conv2': tf.Variable(tf.random_normal([5])),
	    'b_fc': tf.Variable(tf.random_normal([100])),
	    'out': tf.Variable(tf.random_normal([n_classes]))
	}

	# Reshape input to a 4D tensor 
	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	# Convolution Layer, using our function
	conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
	# Max Pooling (down-sampling)
	conv1 = maxpool2d(conv1)
	# Convolution Layer
	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
	# Max Pooling (down-sampling)
	conv2 = maxpool2d(conv2)

	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer
	print conv2.shape

	fc = tf.reshape(conv2, [-1, 7*7*5])
	print fc.shape
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

	print fc.shape

	output = tf.matmul(fc, weights['out']) + biases['out']
	return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)