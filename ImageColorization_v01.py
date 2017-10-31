import tensorflow as tf
import numpy as np
from dataset_loader import *
from utils import *
import time


epsilon = 1e-3
batch_size = 30
test_percentage = 2.5
validation_percentage = 2.5
data_loader = dataset(batch_size = batch_size, test_percentage = test_percentage, validation_percentage = validation_percentage)
total_train_data = data_loader.getTrainData()
total_train_mean, total_train_var = tf.nn.moments(total_train_data,[0])

X = tf.placeholder(tf.float32,shape=[None,256,256,1])
Y = tf.placeholder(tf.float32,shape=[None,64,64,313])


class Net:
    """
    Main Neural Network Class
    """
    def __init__(self):
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

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, s):
        return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

    def feed_forward(self, x, is_training=True):  # , keep_rate):
        # Convolution Layers, using our function
        # block1
        conv_b1_1 = tf.nn.relu(self.conv2d(x, self.weights['W_conv_b1_1'], 1) + self.biases['b_conv_b1_1'])
        conv_b1_2 = self.conv2d(conv_b1_1, self.weights['W_conv_b1_2'], 2) + self.biases['b_conv_b1_2']
        conv_b1_2 = tf.nn.relu(batch_norm(conv_b1_2, is_training))
        # block2
        conv_b2_1 = tf.nn.relu(self.conv2d(conv_b1_2, self.weights['W_conv_b2_1'], 1) + self.biases['b_conv_b2_1'])
        conv_b2_2 = self.conv2d(conv_b2_1, self.weights['W_conv_b2_2'], 2) + self.biases['b_conv_b2_2']
        conv_b2_2 = tf.nn.relu(batch_norm(conv_b2_2, is_training))
        # block3
        conv_b3_1 = tf.nn.relu(self.conv2d(conv_b2_2, self.weights['W_conv_b3_1'], 1) + self.biases['b_conv_b3_1'])
        conv_b3_2 = tf.nn.relu(self.conv2d(conv_b3_1, self.weights['W_conv_b3_2'], 1) + self.biases['b_conv_b3_2'])
        conv_b3_3 = self.conv2d(conv_b3_2, self.weights['W_conv_b3_3'], 2) + self.biases['b_conv_b3_3']
        conv_b3_3 = tf.nn.relu(batch_norm(conv_b3_3, is_training))
        # block4
        conv_b4_1 = tf.nn.relu(self.conv2d(conv_b3_3, self.weights['W_conv_b4_1'], 1) + self.biases['b_conv_b4_1'])
        conv_b4_2 = tf.nn.relu(self.conv2d(conv_b4_1, self.weights['W_conv_b4_2'], 1) + self.biases['b_conv_b4_2'])
        conv_b4_3 = self.conv2d(conv_b4_2, self.weights['W_conv_b4_3'], 1) + self.biases['b_conv_b4_3']
        conv_b4_3 = tf.nn.relu(batch_norm(conv_b4_3, is_training))
        # block5
        conv_b5_1 = tf.nn.relu(self.conv2d(conv_b4_3, self.weights['W_conv_b5_1'], 1) + self.biases['b_conv_b5_1'])
        conv_b5_2 = tf.nn.relu(self.conv2d(conv_b5_1, self.weights['W_conv_b5_2'], 1) + self.biases['b_conv_b5_2'])
        conv_b5_3 = self.conv2d(conv_b5_2, self.weights['W_conv_b5_3'], 1) + self.biases['b_conv_b5_3']
        conv_b5_3 = tf.nn.relu(batch_norm(conv_b5_3, is_training))
        # block6
        conv_b6_1 = tf.nn.relu(self.conv2d(conv_b5_3, self.weights['W_conv_b6_1'], 1) + self.biases['b_conv_b6_1'])
        conv_b6_2 = tf.nn.relu(self.conv2d(conv_b6_1, self.weights['W_conv_b6_2'], 1) + self.biases['b_conv_b6_2'])
        conv_b6_3 = self.conv2d(conv_b6_2, self.weights['W_conv_b6_3'], 1) + self.biases['b_conv_b6_3']
        conv_b6_3 = tf.nn.relu(batch_norm(conv_b6_3, is_training))
        # block7
        conv_b7_1 = tf.nn.relu(self.conv2d(conv_b6_3, self.weights['W_conv_b7_1'], 1) + self.biases['b_conv_b7_1'])
        conv_b7_2 = tf.nn.relu(self.conv2d(conv_b7_1, self.weights['W_conv_b7_2'], 1) + self.biases['b_conv_b7_2'])
        conv_b7_3 = self.conv2d(conv_b7_2, self.weights['W_conv_b7_3'], 1) + self.biases['b_conv_b7_3']
        conv_b7_3 = tf.nn.relu(batch_norm(conv_b7_3, is_training))
        # block8
        # TODO: Verify usage of batch size
        conv_b8_1 = tf.nn.relu(tf.nn.conv2d_transpose(value=conv_b7_3, output_shape=[batch_size, 64, 64, 256],
                                                      filter=self.weights['W_conv_b8_1'], strides=[1, 2, 2, 1],
                                                      padding='SAME') + self.biases['b_conv_b8_1'])
        conv_b8_2 = tf.nn.relu(self.conv2d(conv_b8_1, self.weights['W_conv_b8_2'], 1) + self.biases['b_conv_b8_2'])
        conv_b8_3 = self.conv2d(conv_b8_2, self.weights['W_conv_b8_3'], 1) + self.biases['b_conv_b8_3']
        conv_b8_3 = tf.nn.relu(batch_norm(conv_b8_3, is_training))
        # conv_b8_4 = tf.nn.relu(tf.nn.conv2d_transpose(x, weights['W_conv_b8_4']) + biases['b_conv_b8_4'], [1,256,256,1], strides=[1, 4, 4, 1], padding='SAME'))

        output = self.conv2d(conv_b8_3, self.weights['out'], 1) + self.biases['out']
        return output

def batch_norm(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones(inputs.get_shape().as_list()[1:]))
    beta = tf.Variable(tf.zeros(inputs.get_shape().as_list()[1:]))

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        return tf.nn.batch_normalization(inputs,
            batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            total_train_mean, total_train_var, beta, scale, epsilon)

def test_network(net, epoch, dirName = "generatedPics/"):
    image_l, image_lab = data_loader.getTestData()
    image_l = np.array(image_l, dtype=np.float32)
    encoded_img = net.feed_forward(image_l, is_training=False)
    #saver = tf.train.import_meta_graph('./model/model.ckpt.meta')

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        #saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        encoded_img_val = sess.run(encoded_img)
        images_rgb = decode_batch(image_l, encoded_img_val, 2.63)
        i = 0
        for image_rgb in images_rgb:
            i += 1
            imsave(dirName + 'boosted_' + str(epoch) + 'epoch_' + str(i) + 'Gen.jpeg', image_rgb)

def train_network(net):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    output_log = []
    # tf.reset_default_graph()
    prediction = net.feed_forward(X,is_training=True)
    learning_rate = tf.train.exponential_decay(learning_rate=0.01, global_step=global_step, decay_steps=210000, decay_rate = 0.316, staircase=True)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))/batch_size
    #Uncomment for cost with probabilities
    # alpha = 1
    # gamma = 0.5
    # prior_file = 'resources/prior_probs.npy'
    # weighting_factor = get_weighting_factor(alpha,gamma)
    # cost = tf.reduce_mean(-((Y * tf.log(prediction)) * weighting_factor)/batch_size)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
    hm_epochs = 2
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # save_path = saver.save(sess, "./model/model.ckpt", global_step=global_step)
        for epoch in range(hm_epochs):
            epoch_loss = 0
            data_loader.shuffleTrainImages()
            for _ in range(int(data_loader.n_train_records / batch_size)):
                epoch_x, epoch_y = data_loader.getNextBatch()
                encoded_epoch_y = vectorized_encode(epoch_y)
                _, c = sess.run([optimizer, cost], feed_dict={X: epoch_x, Y: encoded_epoch_y})
                # print "Cost: ", c
                epoch_loss += c
            print('Epoch', epoch, 'completed  out of', hm_epochs, 'loss:', epoch_loss)
            output_log.append('Epoch: ' +  str(epoch) + ' loss: ' + str(epoch_loss))
            if epoch!=0 and epoch%10 == 0:
                save_path = saver.save(sess, "./model/model.ckpt")
                print("Model saved in file: %s" % save_path)
                test_network(net, epoch)
        # save_path = saver.save(sess, "./model/model.ckpt")
        # print("Model saved in file: %s" % save_path)
        save_path = saver.save(sess, "./model/model.ckpt")
        print("Model saved in file: %s" % save_path)
        test_network(net, 'final')


    save_log_to_file(output_log)

print "start"
t1 = time.time()
net = Net()
train_network(net)
t2 = time.time()

print (t2-t1), "seconds"
