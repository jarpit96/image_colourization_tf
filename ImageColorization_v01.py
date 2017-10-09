import tensorflow as tf
import numpy as np
from skimage import color
from skimage.io import imsave
import cv2, os
from skimage.transform import resize
import pprint
import dataset_loader

batch_size = 10
test_percentage = 10
validation_percentage = 10
data_loader = dataset_loader.dataset(batch_size = batch_size, test_percentage = test_percentage, validation_percentage = validation_percentage)
train_size = data_loader.n_train_records
pprint = pprint.PrettyPrinter(indent=4)
dirName = "generatedPics/"

img = cv2.imread(dirName +'gray.jpg')
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = img[None, :, :, None]
X_l = (img.astype(dtype=np.float32)) / 255.0 * 100 - 50
Y_l = []

X = tf.placeholder(tf.float32,shape=[])
Y = tf.placeholder(tf.float32,shape=[])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, s):
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')


def convolutional_neural_network(x):  # , keep_rate):
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
        'W_conv_b3_3': weight_variable([3, 3, 256, 256]),
        # block4
        'W_conv_b4_1': weight_variable([3, 3, 256, 512]),
        'W_conv_b4_2': weight_variable([3, 3, 512, 512]),
        'W_conv_b4_3': weight_variable([3, 3, 512, 512]),
        # block5
        'W_conv_b5_1': weight_variable([3, 3, 512, 512]),
        'W_conv_b5_2': weight_variable([3, 3, 512, 512]),
        'W_conv_b5_3': weight_variable([3, 3, 512, 512]),
        # block6
        'W_conv_b6_1': weight_variable([3, 3, 512, 512]),
        'W_conv_b6_2': weight_variable([3, 3, 512, 512]),
        'W_conv_b6_3': weight_variable([3, 3, 512, 512]),
        # block7
        'W_conv_b7_1': weight_variable([3, 3, 512, 512]),
        'W_conv_b7_2': weight_variable([3, 3, 512, 512]),
        'W_conv_b7_3': weight_variable([3, 3, 512, 256]),
        # block8
        'W_conv_b8_1': weight_variable([4, 4, 256, 256]),
        'W_conv_b8_2': weight_variable([3, 3, 256, 256]),
        'W_conv_b8_3': weight_variable([3, 3, 256, 256]),
        # 'W_conv_b2_4': weight_variable([3, 3, 256, 256]),

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
        'b_conv_b7_3': bias_variable([256]),
        # block8
        'b_conv_b8_1': bias_variable([256]),
        'b_conv_b8_2': bias_variable([256]),
        'b_conv_b8_3': bias_variable([256]),
        # 'b_conv_b8_4': bias_variable([256]),

        'out': bias_variable([313])

    }
    # Convolution Layers, using our function
    # block1
    conv_b1_1 = tf.nn.relu(conv2d(x, weights['W_conv_b1_1'], 1) + biases['b_conv_b1_1'])
    conv_b1_2 = tf.nn.relu(conv2d(conv_b1_1, weights['W_conv_b1_2'], 2) + biases['b_conv_b1_2'])
    # block2
    conv_b2_1 = tf.nn.relu(conv2d(conv_b1_2, weights['W_conv_b2_1'], 1) + biases['b_conv_b2_1'])
    conv_b2_2 = tf.nn.relu(conv2d(conv_b2_1, weights['W_conv_b2_2'], 2) + biases['b_conv_b2_2'])
    # block3
    conv_b3_1 = tf.nn.relu(conv2d(conv_b2_2, weights['W_conv_b3_1'], 1) + biases['b_conv_b3_1'])
    conv_b3_2 = tf.nn.relu(conv2d(conv_b3_1, weights['W_conv_b3_2'], 1) + biases['b_conv_b3_2'])
    conv_b3_3 = tf.nn.relu(conv2d(conv_b3_2, weights['W_conv_b3_3'], 2) + biases['b_conv_b3_3'])
    # block4
    conv_b4_1 = tf.nn.relu(conv2d(conv_b3_3, weights['W_conv_b4_1'], 1) + biases['b_conv_b4_1'])
    conv_b4_2 = tf.nn.relu(conv2d(conv_b4_1, weights['W_conv_b4_2'], 1) + biases['b_conv_b4_2'])
    conv_b4_3 = tf.nn.relu(conv2d(conv_b4_2, weights['W_conv_b4_3'], 1) + biases['b_conv_b4_3'])
    # block5
    conv_b5_1 = tf.nn.relu(conv2d(conv_b4_3, weights['W_conv_b5_1'], 1) + biases['b_conv_b5_1'])
    conv_b5_2 = tf.nn.relu(conv2d(conv_b5_1, weights['W_conv_b5_2'], 1) + biases['b_conv_b5_2'])
    conv_b5_3 = tf.nn.relu(conv2d(conv_b5_2, weights['W_conv_b5_3'], 1) + biases['b_conv_b5_3'])
    # block6
    conv_b6_1 = tf.nn.relu(conv2d(conv_b5_3, weights['W_conv_b6_1'], 1) + biases['b_conv_b6_1'])
    conv_b6_2 = tf.nn.relu(conv2d(conv_b6_1, weights['W_conv_b6_2'], 1) + biases['b_conv_b6_2'])
    conv_b6_3 = tf.nn.relu(conv2d(conv_b6_2, weights['W_conv_b6_3'], 1) + biases['b_conv_b6_3'])
    # block7
    conv_b7_1 = tf.nn.relu(conv2d(conv_b6_3, weights['W_conv_b7_1'], 1) + biases['b_conv_b7_1'])
    conv_b7_2 = tf.nn.relu(conv2d(conv_b7_1, weights['W_conv_b7_2'], 1) + biases['b_conv_b7_2'])
    conv_b7_3 = tf.nn.relu(conv2d(conv_b7_2, weights['W_conv_b7_3'], 1) + biases['b_conv_b7_3'])
    # block8
    conv_b8_1 = tf.nn.relu(
        tf.nn.conv2d_transpose(conv_b7_3, weights['W_conv_b8_1'], output_shape=[1, 64, 64, 256], strides=[1, 2, 2, 1],
                               padding='SAME') + biases['b_conv_b8_1'])
    conv_b8_2 = tf.nn.relu(conv2d(conv_b8_1, weights['W_conv_b8_2'], 1) + biases['b_conv_b8_2'])
    conv_b8_3 = tf.nn.relu(conv2d(conv_b8_2, weights['W_conv_b8_3'], 1) + biases['b_conv_b8_3'])
    # conv_b8_4 = tf.nn.relu(tf.nn.conv2d_transpose(x, weights['W_conv_b8_4']) + biases['b_conv_b8_4'], [1,256,256,1], strides=[1, 4, 4, 1], padding='SAME'))

    output = conv2d(conv_b8_3, weights['out'], 1) + biases['out']
    return output


def softmax(x):
    # pprint.pprint(x)
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1)  # only difference


def decode(data_l, conv8_313, rebalance=1):
    """
    Args:
      data_l   : [1, height, width, 1]
      conv8_313: [1, height/4, width/4, 313]
    Returns:
      img_rgb  : [height, width, 3]
    """
    data_l = data_l + 50
    _, height, width, _ = data_l.shape
    data_l = data_l[0, :, :, :]
    conv8_313 = conv8_313[0, :, :, :]
    enc_dir = './resources'
    conv8_313_rh = conv8_313 * rebalance

    class8_313_rh = softmax(conv8_313_rh)
    #print "313: ", class8_313_rh[0][0]

    cc = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))

    data_ab = np.dot(class8_313_rh, cc)
    data_ab = resize(data_ab, (height, width))
    #print(data_ab)
    #data_ab = data_ab + 50
    img_lab = np.concatenate((data_l, data_ab), axis=-1)
    img_rgb = color.lab2rgb(img_lab)

    return img_rgb


def find_closest(a,b):
    pts = np.load("resources/pts_in_hull.npy")
    ans = 0
    dist = (a-pts[0][0])**2 + (b-pts[0][1])**2
    distances = []
    for i in range(len(pts)):
        temp = (a-pts[i][0])**2 + (b-pts[i][1])**2
        distances.append([temp,i])
        if temp < dist:
            dist = temp
            ans = i
    distances = sorted(distances, key=lambda x: x[0])
    return distances


def encode(batch_y):
    height = 64
    width = 64

    compressed_y = []
    for i in range(len(batch_y)):
        temp = resize(batch_y[i],(height,width))
        compressed_y.append(temp)
    compressed_y = np.array(compressed_y)

    updated_batch_y = []
    updated_y = []
    for y in compressed_y:
        for i in range(height):
            row = []
            for j in range(width):
                a = y[i][j][1]
                b = y[i][j][2]
                cl = np.zeros(313)
                # maxima = find_closest(a,b)
                # cl[maxima[0][1]] = 10
                # cl[maxima[1][1]] = 900
                # cl[maxima[2][1]] = 900
                # cl[maxima[3][1]] = 900
                # cl[maxima[4][1]] = 900
                # cl[maxima[5][1]] = 900
                # cl[maxima[6][1]] = 900
                # cl[maxima[7][1]] = 900
                # cl[maxima[8][1]] = 900
                # cl[maxima[9][1]] = 900
                # row.append(cl)
                sigma = 5
                neighbours = find_closest(a,b)[:10]
                dists = []
                indices = []
                for dist in neighbours:
                    dists.append(dist[0])
                    indices.append(dist[1])
                dists = np.array(dists)
                wts = np.exp(dists / (2 * sigma ** 2))
                wts = wts / np.sum(wts)
                for index, w in zip(indices, wts):
                    cl[index] = w+10
                row.append(cl)
            updated_batch_y.append(row)
    return np.array([updated_batch_y])




def train_neural_network(X):
    prediction = convolutional_neural_network(X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y_l))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(train_size / batch_size)):
                epoch_x, epoch_y = data_loader.getNextBatch()
                _, c = sess.run([optimizer, cost], feed_dict={X: epoch_x, Y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({X: X_l, Y: Y_l}))

#train_neural_network(X_l)

image = cv2.imread(dirName+'sample.JPEG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images = []
images.append(image)
images = np.array(images)
image_l, images = data_loader.rgb2lab(images)
#image_l = images[:,:,:,0:1]

pprint.pprint(images.shape)
images = encode(images)
pprint.pprint(images.shape)
image = decode(image_l, images, 2.63)

#image = decode(X_l, sess.run(prediction))
imsave(dirName + 'sample1.jpeg', image)