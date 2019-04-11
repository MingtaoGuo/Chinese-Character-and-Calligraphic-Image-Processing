from networks import  googlenet
import tensorflow as tf
import scipy.io as sio
import numpy as np


BATCH_SIZE = 50


def get_predict_labels():
    inputs = tf.placeholder("float", [None, 64, 64, 1])
    is_training = tf.placeholder("bool")
    prediction, _ = googlenet(inputs, is_training)
    predict_labels = tf.argmax(prediction, 1)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    data = sio.loadmat("../data/dataset.mat")
    testdata = data["test"] / 127.5 - 1.0
    testlabel = data["testlabels"]
    saver.restore(sess, "../save_para/.\\model.ckpt")
    nums_test = testlabel.shape[1]
    PREDICT_LABELS = np.zeros([nums_test])
    for i in range(nums_test // BATCH_SIZE):
        PREDICT_LABELS[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE] = sess.run(predict_labels, feed_dict={inputs: testdata[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE], is_training: False})
    PREDICT_LABELS[(nums_test // BATCH_SIZE - 1) * BATCH_SIZE + BATCH_SIZE:] = sess.run(predict_labels, feed_dict={inputs: testdata[(nums_test // BATCH_SIZE - 1) * BATCH_SIZE + BATCH_SIZE:], is_training: False})
    np.savetxt("../data/predict_labels.txt", PREDICT_LABELS)

def construct_confusion_matrix(true_labels, predict_labels):
    con_mat = np.zeros([30, 30])
    for i in range(30):
        indx_true_labels = np.where(true_labels == i)[0]
        for j in range(30):
            con_mat[i, j] = np.sum(predict_labels[indx_true_labels] == j) / indx_true_labels.shape[0]
    return con_mat


if __name__ == "__main__":
    data = sio.loadmat("../data/dataset.mat")
    testlabels = data["testlabels"][0, :]
    predictlabels = np.loadtxt("../data/predict_labels.txt")
    con_mat = construct_confusion_matrix(testlabels, predictlabels)
    sio.savemat("../data/con_mat.mat", {"cm": con_mat})