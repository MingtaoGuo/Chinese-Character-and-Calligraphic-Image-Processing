from network import vggnet, transform
import tensorflow as tf
from ops import content_loss, style_loss, gram
from utils import random_batch, random_select_style
from PIL import Image
import numpy as np
import scipy.misc as misc


def Train(IMG_H = 256, IMG_W = 256, IMG_C = 3, STYLE_H=512, STYLE_W=512, C_NUMS = 10, batch_size = 2, learning_rate = 0.001, content_weight = 1.0, style_weight = 5.0, path_content = "./MSCOCO//", path_style = "./style_imgs//", model_path="./save_para//", vgg_path="./vgg_para//"):
    content = tf.placeholder(tf.float32, [batch_size, IMG_H, IMG_W, IMG_C])
    style = tf.placeholder(tf.float32, [batch_size, STYLE_H, STYLE_W, IMG_C])
    y = tf.placeholder(tf.float32, [1, C_NUMS])
    y_ = tf.zeros([1, C_NUMS])
    alpha = tf.constant([1.])
    target = transform(content, y, y_, alpha)
    Phi_T = vggnet(target, vgg_path)
    Phi_C = vggnet(content, vgg_path)
    Phi_S = vggnet(style, vgg_path)
    Loss = content_loss(Phi_C, Phi_T) * content_weight + style_loss(Phi_S, Phi_T) * style_weight
    Style_loss = style_loss(Phi_S, Phi_T)
    Content_loss = content_loss(Phi_C, Phi_T)
    Opt = tf.train.AdamOptimizer(learning_rate).minimize(Loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    for itr in range(40000):
        batch_content= random_batch(path_content, batch_size, [IMG_H, IMG_W, IMG_C])
        batch_style, y_labels = random_select_style(path_style, batch_size, [STYLE_H, STYLE_W, IMG_C], C_NUMS)
        sess.run(Opt, feed_dict={content: batch_content, style: batch_style, y: y_labels})
        if itr % 50 == 0:
            [loss, Target, CONTENT_LOSS, STYLE_LOSS] = sess.run([Loss, target, Content_loss, Style_loss], feed_dict={content: batch_content, style: batch_style, y: y_labels})
            save_img = np.concatenate((batch_content[0, :, :, :], misc.imresize(batch_style[0, :, :, :], [IMG_H, IMG_W]), Target[0, :, :, :]), axis=1)
            print("Iteration: %d, Loss: %e, Content_loss: %e, Style_loss: %e"%(itr, loss, CONTENT_LOSS, STYLE_LOSS))
            Image.fromarray(np.uint8(save_img)).save("./save_imgs//"+str(itr) + "_" + str(np.argmax(y_labels[0, :]))+".jpg")
        if itr % 500 == 0:
            saver.save(sess, model_path+"model.ckpt")

def Init(c_nums = 10, model_path = "./save_para//"):
    content = tf.placeholder(tf.float32, [1, None, None, 3])
    y1 = tf.placeholder(tf.float32, [1, c_nums])
    y2 = tf.placeholder(tf.float32, [1, c_nums])
    alpha = tf.placeholder(tf.float32)
    target = transform(content, y1, y2, alpha)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, model_path + ".\\model.ckpt")
    return target, sess, content, y1, y2, alpha

def stylize(img_path,result_path, label1, label2, alpha, target, sess, content_ph, y1_ph, y2_ph, alpha_ph):
    img = np.array(Image.open(img_path))
    h = img.shape[0]
    w = img.shape[1]
    img = misc.imresize(img, [h//5, w//5])
    Y1 = np.zeros([1, 10])
    Y2 = np.zeros([1, 10])
    Y1[0, label1] = 1
    Y2[0, label2] = 1
    img = sess.run(target, feed_dict={content_ph: img[np.newaxis, :, :, :], y1_ph: Y1, y2_ph: Y2, alpha_ph: alpha})
    Image.fromarray(np.uint8(img[0, :, :, :])).save(result_path + "result"+str(alpha)+".jpg")


if __name__ == "__main__":
    # Train()
    target, sess, content, y1, y2, alpha = Init()
    for alp in [0., 0.2, 0.4, 0.6, 0.8, 1.0]:
        stylize("C://Users//gmt//Desktop//content_dog.jpg", "./results//", 4, 5, alp, target, sess, content, y1, y2, alpha)


