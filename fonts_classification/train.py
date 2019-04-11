from fonts_classification.networks import network, resnet18, alexnet, vgg16, googlenet
from fonts_classification.utils import random_read_batch
import tensorflow as tf
import scipy.io as sio
import numpy as np


BATCH_SIZE = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
EPSILON = 1e-10

def train():
    lr = tf.placeholder("float")
    inputs = tf.placeholder("float", [None, 64, 64, 1])
    labels = tf.placeholder(tf.int32, [None])
    labels_ = tf.one_hot(labels, 30)
    is_training = tf.placeholder("bool")
    prediction, _ = googlenet(inputs, is_training)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_, 1))
    accurancy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    loss = -tf.reduce_sum(labels_ * tf.log(prediction + EPSILON)) + tf.add_n(
        [tf.nn.l2_loss(var) for var in tf.trainable_variables()]) * WEIGHT_DECAY
    Opt = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    data = sio.loadmat("../data/dataset.mat")
    traindata = data["train"] / 127.5 - 1.0#np.reshape(data["train"], [2400, 64, 64, 1]) / 127.5 - 1.0
    trainlabel = data["trainlabels"]#data["train_label"]
    testdata = data["test"] / 127.5 - 1.0#np.reshape(data["test"], [800, 64, 64, 1]) / 127.5 - 1.0
    testlabel = data["testlabels"]#data["test_label"]
    max_test_acc = 0
    loss_list = []
    acc_list = []
    # saver.restore(sess, "./save_para/.\\model.ckpt")
    for i in range(11000):
        batch_data, label_data = random_read_batch(traindata, trainlabel, BATCH_SIZE)
        sess.run(Opt, feed_dict={inputs: batch_data, labels: label_data, is_training: True, lr: LEARNING_RATE})
        if i % 20 == 0:
            [LOSS, TRAIN_ACCURACY] = sess.run([loss, accurancy], feed_dict={inputs: batch_data, labels: label_data, is_training: False, lr: LEARNING_RATE})
            loss_list.append(LOSS)
            TEST_ACCURACY = 0
            for j in range(356):
                TEST_ACCURACY += sess.run(accurancy, feed_dict={inputs: testdata[j*50:j*50+50], labels: testlabel[0, j*50:j*50+50], is_training: False, lr: LEARNING_RATE})
            TEST_ACCURACY /= 356
            acc_list.append(TEST_ACCURACY)
            if TEST_ACCURACY > max_test_acc:
                max_test_acc = TEST_ACCURACY
            print("Step: %d, loss: %4g, training accuracy: %4g, testing accuracy: %4g, max testing accuracy: %4g"%(i, LOSS, TRAIN_ACCURACY, TEST_ACCURACY, max_test_acc))
        if i % 1000 == 0:
            np.savetxt("../data/loss_list.txt", loss_list)
            np.savetxt("../data/acc_list.txt", acc_list)
            saver.save(sess, "../save_para/model.ckpt")


if __name__ == "__main__":
    train()