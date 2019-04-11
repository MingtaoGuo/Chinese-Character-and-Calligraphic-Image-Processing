import tensorflow as tf



def conditional_instance_norm(x, scope_bn, y1=None, y2=None, alpha=1):
    mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    if y1==None:
        beta = tf.get_variable(name=scope_bn + 'beta', shape=[x.shape[-1]], initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
        gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[x.shape[-1]], initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
    else:
        beta = tf.get_variable(name=scope_bn+'beta', shape=[y1.shape[-1], x.shape[-1]], initializer=tf.constant_initializer([0.]), trainable=True) # label_nums x C
        gamma = tf.get_variable(name=scope_bn+'gamma', shape=[y1.shape[-1], x.shape[-1]], initializer=tf.constant_initializer([1.]), trainable=True) # label_nums x C
        beta1 = tf.matmul(y1, beta)
        gamma1 = tf.matmul(y1, gamma)
        beta2 = tf.matmul(y2, beta)
        gamma2 = tf.matmul(y2, gamma)
        beta = alpha * beta1 + (1. - alpha) * beta2
        gamma = alpha * gamma1 + (1. - alpha) * gamma2
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-10)
    return x


def conv(name, inputs, k_size, nums_in, nums_out, strides):
    pad_size = k_size // 2
    inputs = tf.pad(inputs, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode="REFLECT")
    kernel = tf.get_variable(name+"W", [k_size, k_size, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.01))
    bias = tf.get_variable(name+"B", [nums_out], initializer=tf.constant_initializer(0.))
    return tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], "VALID") + bias

def upsampling(name, inputs, nums_in, nums_out, y1, y2, alpha):
    inputs = tf.image.resize_nearest_neighbor(inputs, [tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])
    return conditional_instance_norm(conv(name, inputs, 3, nums_in, nums_out, 1), "cin"+name, y1, y2, alpha)

def relu(inputs):
    return tf.nn.relu(inputs)

def sigmoid(inputs):
    return tf.nn.sigmoid(inputs)

def ResBlock(name, inputs, k_size, nums_in, nums_out, y1, y2, alpha):
    temp = inputs * 1.0
    inputs = conditional_instance_norm(conv("conv1_" + name, inputs, k_size, nums_in, nums_out, 1), "cin1"+name, y1, y2, alpha)
    inputs = relu(inputs)
    inputs = conditional_instance_norm(conv("conv2_" + name, inputs, k_size, nums_in, nums_out, 1), "cin2"+name, y1, y2, alpha)
    return inputs + temp

def content_loss(phi_content, phi_target):
    return tf.nn.l2_loss(phi_content["conv2_2"] - phi_target["conv2_2"]) * 2 / tf.cast(tf.size(phi_content["conv2_2"]), dtype=tf.float32)

def style_loss(phi_style, phi_target):
    layers = ["conv1_2", "conv2_2", "conv3_3", "conv4_3"]
    loss = 0
    for layer in layers:
        s_maps = phi_style[layer]
        G_s = gram(s_maps)
        t_maps = phi_target[layer]
        G_t = gram(t_maps)
        loss += tf.nn.l2_loss(G_s - G_t) * 2 / tf.cast(tf.size(G_t), dtype=tf.float32)
    return loss

def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)
    return grams





