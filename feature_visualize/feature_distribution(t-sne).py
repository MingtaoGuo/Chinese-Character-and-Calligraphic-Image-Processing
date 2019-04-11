from networks import  googlenet
import tensorflow as tf
import scipy.io as sio
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


BATCH_SIZE = 50


def get_feature():
    inputs = tf.placeholder("float", [None, 64, 64, 1])
    is_training = tf.placeholder("bool")
    _, feature = googlenet(inputs, is_training)
    feature = tf.squeeze(feature, [1, 2])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    data = sio.loadmat("../data/dataset.mat")
    testdata = data["test"] / 127.5 - 1.0
    testlabels = data["testlabels"]
    saver.restore(sess, "../save_para/.\\model.ckpt")
    nums_test = testdata.shape[0]
    FEATURE = np.zeros([nums_test, 1024])
    for i in range(nums_test // BATCH_SIZE):
        FEATURE[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE] = sess.run(feature, feed_dict={inputs: testdata[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE], is_training: False})
    FEATURE[(nums_test // BATCH_SIZE - 1) * BATCH_SIZE + BATCH_SIZE:] = sess.run(feature, feed_dict={inputs: testdata[(nums_test // BATCH_SIZE - 1) * BATCH_SIZE + BATCH_SIZE:], is_training: False})
    sio.savemat("../data/feature.mat", {"feature": FEATURE, "testlabels": testlabels})

def tsne():
    data = sio.loadmat("../data/feature.mat")
    feature_test = data["feature"]
    proj = TSNE().fit_transform(feature_test)
    sio.savemat("../data/proj.mat", {"proj": proj})

def plot_tsne():
    proj = sio.loadmat("../data/proj.mat")["proj"]
    color = ['darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise',
             'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite',
             'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray',
             'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavenderblush']
    data = sio.loadmat("../data/dataset.mat")
    labels = data["testlabels"][0, :]
    for i in range(30):
        plt.plot(proj[np.where(labels == i)[0], 0], proj[np.where(labels == i)[0], 1], ".", c=color[i], label="gmt")

    plt.show()



if __name__ == "__main__":
    plot_tsne()