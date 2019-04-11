import tensorflow as tf
import numpy as np
from PIL import Image
import os
import scipy.misc as misc
def read_data(path_perfect,  batch_size, img_size=256):
    def random_crop(img, crop_size=256):
        h = img.shape[0]
        w = img.shape[1]
        if h < crop_size or w < crop_size:
            if h < w:
                img = misc.imresize(img[:, :h], [crop_size, crop_size])
            else:
                img = misc.imresize(img[:w, :], [crop_size, crop_size])
            return img
        start_y = np.random.randint(0, h-crop_size+1)
        start_x = np.random.randint(0, w-crop_size+1)
        return img[start_y:start_y+crop_size, start_x:start_x+crop_size]

    filenames_perfect = os.listdir(path_perfect)
    data_perfect = np.zeros([batch_size, img_size, img_size, 3])
    rand_select_perfect = np.random.randint(0, filenames_perfect.__len__(), [batch_size])
    for i in range(batch_size):
        #read the perfect data
        img = np.array(Image.open(path_perfect+filenames_perfect[rand_select_perfect[i]]))
        shape_list = img.shape
        if shape_list.__len__() < 3:
            img = np.dstack((img, img, img))
        # h, w = shape_list[0], shape_list[1]
        # img = misc.imresize(img, [int(256 * h / w), 256])

        data_perfect[i] = random_crop(img[:,:,:3], crop_size=256)
    return data_perfect

def conv_layer_relu(x,W,b):
    return tf.nn.relu(tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b)

def conv_trainnet(x,kersize,chin,chout,stride,pad):
    W = tf.Variable(tf.truncated_normal([kersize,kersize,chin,chout],stddev=0.01))
    b = tf.Variable(tf.constant(0.01,shape=[chout]))
    return tf.nn.conv2d(x,W,[1,stride,stride,1],padding=pad)+b

def deconv(x,kersize,chin,chout,stride):
    W = tf.Variable(tf.truncated_normal([kersize, kersize, chout, chin], stddev=0.01))
    b = tf.Variable(tf.constant(0.01, shape=[chout]))
    batch,h,w = tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2]
    c = x.shape[-1]
    return tf.nn.conv2d_transpose(x,W,[batch,h*2,w*2,chout],[1,stride,stride,1],padding='SAME')+b

def instance_norm(x):
    mu,var = tf.nn.moments(x,axes=[1,2],keep_dims=True)
    c= mu.shape[-1]
    gama = tf.Variable(tf.constant(1.0,shape=[1,1,1,c]))
    beta = tf.Variable(tf.constant(0.0,shape=[1,1,1,c]))
    return gama*(x-mu)/tf.sqrt(var+1e-10)+beta

def residual_block(x):
    res_1 =conv_trainnet(x,3,128,128,1,'VALID')
    res_2 =tf.nn.relu(instance_norm(res_1))
    res_3 = conv_trainnet(res_2,3,128,128,1,"VALID")

    return instance_norm(res_3)+x[:, 2:-2, 2:-2, :]

def ave_pool(x):
    return tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')

def VGG(x,para):
    x = tf.reverse(x, [-1]) - np.array([103.939, 116.779, 123.68])
    conv1_1 = conv_layer_relu(x, para['conv1_1'][0], para['conv1_1'][1])
    conv1_2 = conv_layer_relu(conv1_1, para['conv1_2'][0], para['conv1_2'][1])
    conv1_2_ave = ave_pool(conv1_2)

    conv2_1 = conv_layer_relu(conv1_2_ave, para['conv2_1'][0], para['conv2_1'][1])
    conv2_2 = conv_layer_relu(conv2_1, para['conv2_2'][0], para['conv2_2'][1])
    conv2_2_ave = ave_pool(conv2_2)

    conv3_1 = conv_layer_relu(conv2_2_ave, para['conv3_1'][0], para['conv3_1'][1])
    conv3_2 = conv_layer_relu(conv3_1, para['conv3_2'][0], para['conv3_2'][1])
    conv3_3 = conv_layer_relu(conv3_2, para['conv3_3'][0], para['conv3_3'][1])
    conv3_3_ave = ave_pool(conv3_3)

    conv4_1 = conv_layer_relu(conv3_3_ave, para['conv4_1'][0], para['conv4_1'][1])
    conv4_2 = conv_layer_relu(conv4_1, para['conv4_2'][0], para['conv4_2'][1])
    conv4_3 = conv_layer_relu(conv4_2, para['conv4_3'][0], para['conv4_3'][1])
    f = {}
    f["conv1_2"] = conv1_2
    f["conv2_2"] = conv2_2
    f["conv3_3"] = conv3_3
    f["conv4_3"] = conv4_3
    return f

def style_training_net(x):
    pads = tf.constant([[0,0],[40,40],[40,40],[0,0]])
    x_ = tf.pad(x,pads,"REFLECT")
    x_ = tf.nn.relu(instance_norm(conv_trainnet(x_,9,3,32,1,'SAME')))
    x_ = tf.nn.relu(instance_norm(conv_trainnet(x_,3,32,64,2,'SAME')))
    x_ = tf.nn.relu(instance_norm(conv_trainnet(x_,3,64,128,2,'SAME')))
    for i in range(5):
        x_ = residual_block(x_)
    x_ = tf.nn.relu(instance_norm(deconv(x_,3,128,64,2)))
    x_ = tf.nn.relu(instance_norm(deconv(x_,3,64,32,2)))
    x_ = instance_norm(conv_trainnet(x_,9,32,3,1,'SAME'))
    return (tf.tanh(x_)+1)*127.5


def con_loss(f_con,f_yhat):
    h,w,dim = tf.shape(f_yhat['conv2_2'])[1],tf.shape(f_yhat['conv2_2'])[2],tf.shape(f_yhat['conv2_2'])[-1]
    temp = tf.cast(h*w*dim, dtype=tf.float32)
    return tf.nn.l2_loss((f_yhat['conv2_2']-f_con['conv2_2'])) / temp
def sty_loss(f_sty,f_yhat):
    loss = 0
    for name in ['conv1_2','conv2_2','conv3_3','conv4_3']:
        a = f_sty[name]
        batch,h_s,w_s,dim_s = tf.shape(a)[0],tf.shape(a)[1],tf.shape(a)[2],tf.shape(a)[3]
        a = tf.transpose(a,perm=[0,3,1,2])
        a = tf.reshape(a,[batch,dim_s,-1])
        GM_sty = tf.matmul(a,tf.transpose(a,perm=[0,2,1]))/tf.cast(dim_s*h_s*w_s,dtype=tf.float32)

        a = f_yhat[name]
        h_y,w_y,dim_y = tf.shape(a)[1],tf.shape(a)[2],tf.shape(a)[3]
        a = tf.transpose(a,perm=[0,3,1,2])
        a =tf.reshape(a,[batch,dim_y,-1])
        GM_yhat = tf.matmul(a,tf.transpose(a,perm=[0,2,1]))/tf.cast(dim_y*h_y*w_y,dtype=tf.float32)
        loss += tf.nn.l2_loss(GM_yhat-GM_sty) / tf.cast(tf.size(GM_sty), dtype=tf.float32)
    return loss
def mapping(img):
    return 255.0 * (img - np.min(img)) / (np.max(img) - np.min(img))

# def read_data(path, batch_size):
#     filenames = os.listdir(path)
#     filenames_len = filenames.__len__()
#     rand_select = np.random.randint(0, filenames_len, [batch_size])
#     batch_data = np.zeros([batch_size, 256, 256, 3])
#     for i in range(batch_size):
#         img = np.array(Image.open(path + filenames[rand_select[i]]).resize([256, 256]))
#         try:
#             if img.shape.__len__() == 3:
#                 batch_data[i, :, :, :] = img[:256, :256, :3]
#             else:
#                 batch_data[i, :, :, :] = np.dstack((img, img, img))[:256, :256, :]
#         except:
#             img = np.array(Image.open(path + filenames[0]))
#             batch_data[i, :, :, :] = img[:256, :256, :3]
#     return batch_data


if __name__=="__main__":

    cot_pic = tf.placeholder('float', [None, None, None, 3])

    y_hat = style_training_net(cot_pic)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, "C:/Users/gmt/Documents/Tencent Files/798714378/FileRecv/save_para(1)/save_para/model.ckpt")
        batch = np.array(Image.open("C:/Users/gmt/Desktop/zhang3.jpg"))
        h = batch.shape[0]
        w = batch.shape[1]
        batch = misc.imresize(batch, [h//5, w//5])[np.newaxis, :, :, :]
        img = sess.run(y_hat, feed_dict={cot_pic: batch})
        Image.fromarray(np.uint8(img[0])).save("./stylized.jpg")
