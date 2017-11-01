import tensorflow as tf


class Net:
    """
    Main Neural Network Class
    """
    def __init__(self, batch_size):
        self.weights = {
                    # block1
                    'W_conv_b1_1': self.weight_variable([3, 3, 1, 64]),
                    'W_conv_b1_2': self.weight_variable([3, 3, 64, 64]),
                    # block2
                    'W_conv_b2_1': self.weight_variable([3, 3, 64, 128]),
                    'W_conv_b2_2': self.weight_variable([3, 3, 128, 128]),
                    # block3
                    'W_conv_b3_1': self.weight_variable([3, 3, 128, 256]),
                    'W_conv_b3_2': self.weight_variable([3, 3, 256, 256]),
                    'W_conv_b3_3': self.weight_variable([3, 3, 256, 256]),
                    # block4
                    'W_conv_b4_1': self.weight_variable([3, 3, 256, 512]),
                    'W_conv_b4_2': self.weight_variable([3, 3, 512, 512]),
                    'W_conv_b4_3': self.weight_variable([3, 3, 512, 512]),
                    # block5
                    'W_conv_b5_1': self.weight_variable([3, 3, 512, 512]),
                    'W_conv_b5_2': self.weight_variable([3, 3, 512, 512]),
                    'W_conv_b5_3': self.weight_variable([3, 3, 512, 512]),
                    # block6
                    'W_conv_b6_1': self.weight_variable([3, 3, 512, 512]),
                    'W_conv_b6_2': self.weight_variable([3, 3, 512, 512]),
                    'W_conv_b6_3': self.weight_variable([3, 3, 512, 512]),
                    # block7
                    'W_conv_b7_1': self.weight_variable([3, 3, 512, 512]),
                    'W_conv_b7_2': self.weight_variable([3, 3, 512, 512]),
                    'W_conv_b7_3': self.weight_variable([3, 3, 512, 256]),
                    # block8
                    'W_conv_b8_1': self.weight_variable([4, 4, 256, 256]),
                    'W_conv_b8_2': self.weight_variable([3, 3, 256, 256]),
                    'W_conv_b8_3': self.weight_variable([3, 3, 256, 256]),
                    # 'W_conv_b2_4': weight_variable([3, 3, 256, 256]),

                    'out': self.weight_variable([1, 1, 256, 313])
                }
        self.biases = {
                # block1
                'b_conv_b1_1': self.bias_variable([64]),
                'b_conv_b1_2': self.bias_variable([64]),
                # block2
                'b_conv_b2_1': self.bias_variable([128]),
                'b_conv_b2_2': self.bias_variable([128]),
                # block3
                'b_conv_b3_1': self.bias_variable([256]),
                'b_conv_b3_2': self.bias_variable([256]),
                'b_conv_b3_3': self.bias_variable([256]),
                # block4
                'b_conv_b4_1': self.bias_variable([512]),
                'b_conv_b4_2': self.bias_variable([512]),
                'b_conv_b4_3': self.bias_variable([512]),
                # block5
                'b_conv_b5_1': self.bias_variable([512]),
                'b_conv_b5_2': self.bias_variable([512]),
                'b_conv_b5_3': self.bias_variable([512]),
                # block6
                'b_conv_b6_1': self.bias_variable([512]),
                'b_conv_b6_2': self.bias_variable([512]),
                'b_conv_b6_3': self.bias_variable([512]),
                # block7
                'b_conv_b7_1': self.bias_variable([512]),
                'b_conv_b7_2': self.bias_variable([512]),
                'b_conv_b7_3': self.bias_variable([256]),
                # block8
                'b_conv_b8_1': self.bias_variable([256]),
                'b_conv_b8_2': self.bias_variable([256]),
                'b_conv_b8_3': self.bias_variable([256]),
                # 'b_conv_b8_4': bias_variable([256]),

                'out': self.bias_variable([313])
                }
        self.batch_size = batch_size

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, s):
        return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

    def batch_norm(self, inputs, is_training, decay=0.999, is_debug=False):
        self.epsilon = 1e-3
        scale = tf.Variable(tf.ones(inputs.get_shape().as_list()[1:]))
        beta = tf.Variable(tf.zeros(inputs.get_shape().as_list()[1:]))
        pop_mean = tf.Variable(tf.zeros(inputs.get_shape().as_list()[1:]), trainable=False)
        pop_var = tf.Variable(tf.ones(inputs.get_shape().as_list()[1:]), trainable=False)

        if is_debug:
            if is_training:
                batch_mean, batch_var = tf.nn.moments(inputs, [0])
                train_mean = tf.assign(pop_mean,
                                       pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var,
                                      pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,
                                                     batch_mean, batch_var, beta, scale, self.epsilon), batch_mean, batch_var
            else:
                return tf.nn.batch_normalization(inputs,
                                                 pop_mean, pop_var, beta, scale, self.epsilon), pop_mean, pop_var
        else:
            if is_training:
                batch_mean, batch_var = tf.nn.moments(inputs, [0])
                train_mean = tf.assign(pop_mean,
                                       pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var,
                                      pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,
                                                     batch_mean, batch_var, beta, scale, self.epsilon)
            else:
                return tf.nn.batch_normalization(inputs,
                                                 pop_mean, pop_var, beta, scale, self.epsilon)

    def feed_forward(self, x, is_training=True):  # , keep_rate):
        # Convolution Layers, using our function
        # block1
        conv_b1_1 = tf.nn.relu(self.conv2d(x, self.weights['W_conv_b1_1'], 1) + self.biases['b_conv_b1_1'])
        conv_b1_2 = self.conv2d(conv_b1_1, self.weights['W_conv_b1_2'], 2) + self.biases['b_conv_b1_2']
        conv_b1_2 = tf.nn.relu(self.batch_norm(conv_b1_2, is_training))
        # block2
        conv_b2_1 = tf.nn.relu(self.conv2d(conv_b1_2, self.weights['W_conv_b2_1'], 1) + self.biases['b_conv_b2_1'])
        conv_b2_2 = self.conv2d(conv_b2_1, self.weights['W_conv_b2_2'], 2) + self.biases['b_conv_b2_2']
        conv_b2_2 = tf.nn.relu(self.batch_norm(conv_b2_2, is_training))
        # block3
        conv_b3_1 = tf.nn.relu(self.conv2d(conv_b2_2, self.weights['W_conv_b3_1'], 1) + self.biases['b_conv_b3_1'])
        conv_b3_2 = tf.nn.relu(self.conv2d(conv_b3_1, self.weights['W_conv_b3_2'], 1) + self.biases['b_conv_b3_2'])
        conv_b3_3 = self.conv2d(conv_b3_2, self.weights['W_conv_b3_3'], 2) + self.biases['b_conv_b3_3']
        conv_b3_3 = tf.nn.relu(self.batch_norm(conv_b3_3, is_training))
        # block4
        conv_b4_1 = tf.nn.relu(self.conv2d(conv_b3_3, self.weights['W_conv_b4_1'], 1) + self.biases['b_conv_b4_1'])
        conv_b4_2 = tf.nn.relu(self.conv2d(conv_b4_1, self.weights['W_conv_b4_2'], 1) + self.biases['b_conv_b4_2'])
        conv_b4_3 = self.conv2d(conv_b4_2, self.weights['W_conv_b4_3'], 1) + self.biases['b_conv_b4_3']
        conv_b4_3 = tf.nn.relu(self.batch_norm(conv_b4_3, is_training))
        # block5
        conv_b5_1 = tf.nn.relu(self.conv2d(conv_b4_3, self.weights['W_conv_b5_1'], 1) + self.biases['b_conv_b5_1'])
        conv_b5_2 = tf.nn.relu(self.conv2d(conv_b5_1, self.weights['W_conv_b5_2'], 1) + self.biases['b_conv_b5_2'])
        conv_b5_3 = self.conv2d(conv_b5_2, self.weights['W_conv_b5_3'], 1) + self.biases['b_conv_b5_3']
        conv_b5_3 = tf.nn.relu(self.batch_norm(conv_b5_3, is_training))
        # block6
        conv_b6_1 = tf.nn.relu(self.conv2d(conv_b5_3, self.weights['W_conv_b6_1'], 1) + self.biases['b_conv_b6_1'])
        conv_b6_2 = tf.nn.relu(self.conv2d(conv_b6_1, self.weights['W_conv_b6_2'], 1) + self.biases['b_conv_b6_2'])
        conv_b6_3 = self.conv2d(conv_b6_2, self.weights['W_conv_b6_3'], 1) + self.biases['b_conv_b6_3']
        conv_b6_3 = tf.nn.relu(self.batch_norm(conv_b6_3, is_training))
        # block7
        conv_b7_1 = tf.nn.relu(self.conv2d(conv_b6_3, self.weights['W_conv_b7_1'], 1) + self.biases['b_conv_b7_1'])
        conv_b7_2 = tf.nn.relu(self.conv2d(conv_b7_1, self.weights['W_conv_b7_2'], 1) + self.biases['b_conv_b7_2'])
        conv_b7_3 = self.conv2d(conv_b7_2, self.weights['W_conv_b7_3'], 1) + self.biases['b_conv_b7_3']
        conv_b7_3 = tf.nn.relu(self.batch_norm(conv_b7_3, is_training))
        # block8
        # TODO: Verify usage of batch size
        conv_b8_1 = tf.nn.relu(tf.nn.conv2d_transpose(value=conv_b7_3, output_shape=[self.batch_size, 64, 64, 256],
                                                      filter=self.weights['W_conv_b8_1'], strides=[1, 2, 2, 1],
                                                      padding='SAME') + self.biases['b_conv_b8_1'])
        conv_b8_2 = tf.nn.relu(self.conv2d(conv_b8_1, self.weights['W_conv_b8_2'], 1) + self.biases['b_conv_b8_2'])
        conv_b8_3 = self.conv2d(conv_b8_2, self.weights['W_conv_b8_3'], 1) + self.biases['b_conv_b8_3']
        conv_b8_3 = tf.nn.relu(self.batch_norm(conv_b8_3, is_training))

        output = self.conv2d(conv_b8_3, self.weights['out'], 1) + self.biases['out']
        return output