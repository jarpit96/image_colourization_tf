import tensorflow as tf
import numpy as np
from Net import *
from dataset_loader import *
from utils import *
import time


batch_size = 3
test_percentage = 0.25
validation_percentage = 99.5
hm_epochs = 1
data_loader = dataset(batch_size=batch_size, test_percentage=test_percentage,
                      validation_percentage=validation_percentage)

X = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
Y = tf.placeholder(tf.float32, shape=[None, 64, 64, 313])


def test_network(epoch, dir_name="generatedPics/"):
    with tf.Session() as sess:
        image_l, image_lab = data_loader.getTestData()
        image_l = np.array(image_l, dtype=np.float32)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        encoded_img_val = sess.run(net.prediction, feed_dict={X:image_l})
        images_rgb = decode_batch(image_l, encoded_img_val, 2.63)
        i = 0
        for image_rgb in images_rgb:
            i += 1
            imsave(dir_name + 'boosted_' + str(epoch) + 'epoch_' + str(i) + 'Gen.jpeg', image_rgb)


def train_network():
    output_log = []
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./summaries/my_graph', sess.graph)
        saver = tf.train.Saver()
        for epoch in range(hm_epochs):
            epoch_loss = 0
            data_loader.shuffleTrainImages()
            for _ in range(int(data_loader.n_train_records / batch_size)):
                epoch_x, epoch_y = data_loader.getNextBatch()
                encoded_epoch_y = vectorized_encode(epoch_y)
                _, c = sess.run([net.optimizer, net.cost], feed_dict={X: epoch_x, Y: encoded_epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed  out of', hm_epochs, 'loss:', epoch_loss)
            output_log.append('Epoch: ' + str(epoch) + ' loss: ' + str(epoch_loss))
            if epoch != 0 and epoch % 10 == 0:
                save_path = saver.save(sess, "./model/model", global_step=net.global_step)
                print("Model saved in file: %s" % save_path)
                test_network(epoch)
        save_path = saver.save(sess, "./model/model", global_step=hm_epochs)
        print("Model saved in file: %s" % save_path)
        test_network('final')
    save_log_to_file(output_log)


print "start"
t1 = time.time()

net = Net(X, Y, batch_size)
train_network()

t2 = time.time()
print (t2-t1), "seconds"
