import tensorflow as tf
import tensorflow.contrib as contrib


def conv(inputs, num_out, ksize, strides):
    c = int(inputs.shape[-1])
    W = tf.get_variable("W", [ksize, ksize, c, num_out], initializer=contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [num_out], initializer=tf.constant_initializer([0.01]))
    return tf.nn.conv2d(inputs, W, [1, strides, strides, 1], "SAME") + b

def max_pooling(inputs, ksize, strides):
    return tf.nn.max_pool(inputs, [1, ksize, ksize, 1], [1, strides, strides, 1], padding="SAME")

def relu(inputs):
    return tf.nn.relu(inputs)

def fullycon(inputs, num_out):
    inputs = tf.layers.flatten(inputs)
    c = int(inputs.shape[-1])
    W = tf.get_variable("W", [c, num_out], initializer=contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [num_out], initializer=tf.constant_initializer([0.01]))
    return (tf.matmul(inputs, W) + b)

def global_avg_pool(inputs):
    w = int(inputs.shape[1])
    h = int(inputs.shape[2])
    return tf.nn.avg_pool(inputs, [1, w, h, 1], [1, 1, 1, 1], "VALID")

def SE_Block(inputs):
    #Squeeze-and-Excitation Networks
    #Hu J, Shen L, Sun G. Squeeze-and-Excitation Networks[J]. 2017.
    c = int(inputs.shape[-1])
    squeeze = tf.squeeze(global_avg_pool(inputs), [1, 2])
    with tf.variable_scope("FC1"):
        excitation = tf.nn.relu(fullycon(squeeze, int(c/16)))
    with tf.variable_scope("FC2"):
        excitation = tf.nn.sigmoid(fullycon(excitation, c))
    excitation = tf.reshape(excitation, [-1, 1, 1, c])
    scale = inputs * excitation
    return scale


def batchnorm(x, train_phase, scope_bn):
    # Batch Normalization
    # Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def haar_wavelet_block(x):
    #shape of x: Batch_size x feature_map_size
    x = tf.squeeze(global_avg_pool(x), [1, 2])
    feature_map_size = x.shape[-1]


    length = feature_map_size // 2
    temp = tf.reshape(x, [-1, length, 2])
    a = (temp[:, :, 0] + temp[:, :, 1]) / 2
    detail = (temp[:, :, 0] - temp[:, :, 1]) / 2
    length = length // 2
    while length != 16:  # 一级：32，acc：97.5， 二级：16，acc：97.875，三级：8, acc: 98.628, 四级：4，acc: 97.625, 五级：2，acc：97.5，六级：1，acc：97.375
        a = tf.reshape(a, [-1, length, 2])
        detail = tf.concat([(a[:, :, 0] - a[:, :, 1]) / 2, detail], axis=1)
        a = (a[:, :, 0] + a[:, :, 1]) / 2
        length = length // 2
    haar_info = tf.concat([a, detail], axis=1)
    return haar_info