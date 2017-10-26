import tensorflow as tf
import numpy as np
import cv2
import os
from skimage import color
from skimage.transform import resize
import sklearn.neighbors as nn
import dataset_loader
from skimage.io import imsave

#Main Util Functions


#TODO: imporve efficiency - Make vectorized decode function
def decode_batch(data_l, conv8_313, rebalance=1):
    images_rgb = []
    for (data_l_obj, conv8_313_obj) in zip(data_l, conv8_313):
        images_rgb.append(decode(np.array([data_l_obj]), np.array([conv8_313_obj]), rebalance))
    return images_rgb


def decode(data_l, conv8_313, rebalance=1):
    """
    Args:
      data_l   : [1, height, width, 1]
      conv8_313: [1, height/4, width/4, 313]
    Returns:
      img_rgb  : [height, width, 3]
    """
    # conv8_313 = np.array(conv8_313)
    data_l = data_l + 50
    _, height, width, _ = data_l.shape
    height = int(height)
    width = int(width)
    data_l = data_l[0, :, :, :]
    conv8_313 = conv8_313[0, :, :, :]
    conv8_313 = np.array(conv8_313)
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


#TODO: improve efficiency - Vectorize operations
def vectorized_encode(batch_y):
    height = 64
    width = 64
    NN = 10
    enc_dir = './resources/'
    sigma = 5
    color_space = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))

    compressed_y = []
    for i in range(len(batch_y)):
        temp = resize(batch_y[i],(height,width))
        compressed_y.append(temp)
    compressed_y = np.array(compressed_y)

    updated_batch_y = []
    for y in compressed_y:
        updated_y = []
        flat_image = flatten_image(y)
        (dists, indices) = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(color_space).kneighbors(
            flat_image)
        wts = np.exp(dists / (2 * sigma ** 2))
        wts = wts / (np.sum(wts, axis=1).reshape(-1, 1))
        wts = wts+10

        pts = np.zeros(shape=(height*width, 313))
        x = np.arange(0, height*width, dtype=np.int)[:,np.newaxis]
        pts[x,indices] = wts
        encoded_image = pts.reshape(height, width, -1)

        updated_batch_y.append(encoded_image)
    updated_batch_y = np.array(updated_batch_y)
    return updated_batch_y


def test_encode(dirName = "generatedPics/"):
    data_loader = dataset_loader.dataset()
    image = cv2.imread(dirName + 'sample.JPEG')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images = []
    images.append(image)
    images = np.array(images)
    image_l, images = data_loader.rgb2lab(images)
    # image_l = images[:,:,:,0:1]

    # pprint.pprint(images.shape)
    images = encode(images)
    # pprint.pprint(images.shape)
    image = decode(image_l, images, 2.63)

    # image = decode(X_l, sess.run(prediction))
    imsave(dirName + 'sample2.jpeg', image)


def save_model(sess):
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/model.ckpt")
    print("Model saved in file: %s" % save_path)


#Auxilary Util Functions


def softmax(x):
    # pprint.pprint(x)
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1)  # only difference


def flatten_image(image):
    image = np.array(image)
    flat_image = image[:, :, 1:]
    flat_image = flat_image.reshape(-1, flat_image.shape[2])
    return flat_image


def save_log_to_file(data):
    thefile = open('output_log.txt', 'w')
    for item in data:
        thefile.write("%s\n" % item)


def get_weighting_factor(alpha=1, gamma=0.5, prior_file='resources/prior_probs.npy'):
    prior_probs = np.load(prior_file)
    uni_probs = np.zeros_like(prior_probs)
    uni_probs[prior_probs != 0] = 1.
    uni_probs = uni_probs / np.sum(uni_probs)
    # convex combination of empirical prior and uniform distribution
    prior_mix = (1 - gamma) * prior_probs + gamma * uni_probs
    # set prior factor
    prior_factor = prior_mix ** -alpha
    prior_factor = prior_factor / np.sum(prior_probs * prior_factor)  # re-normalize
    return prior_factor


def encode(batch_y):
    height = 64
    width = 64
    NN = 10
    enc_dir = './resources/'
    sigma = 5
    color_space = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))

    compressed_y = []
    for i in range(len(batch_y)):
        temp = resize(batch_y[i],(height,width))
        compressed_y.append(temp)
    compressed_y = np.array(compressed_y)

    updated_batch_y = []
    updated_y = []
    for y in compressed_y:
        updated_y = []
        for i in range(height):
            row = []
            for j in range(width):
                a = y[i][j][1]
                b = y[i][j][2]
                cl = np.zeros(313)
                (dists, indices) = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(color_space).kneighbors(np.array([[a,b]]))
                dists = np.array(dists)
                wts = np.exp(dists / (2 * sigma ** 2))
                wts = wts / np.sum(wts)
                for index, w in zip(indices, wts):
                    cl[index] = w+10
                row.append(cl)
            updated_y.append(row)
        updated_batch_y.append(updated_y)
        updated_y = np.array(updated_y)
    updated_batch_y = np.array(updated_batch_y)
    return updated_batch_y
