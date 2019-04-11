from fonts_classification.ops import *
from other_networks.resnet_v1 import resnet_v1_18
from other_networks.alexnet_v2 import alexnet_v2
from other_networks.vgg16 import vgg_16
from other_networks.inception_v2 import inception_v2
import tensorflow.contrib.slim.nets

def network(inputs, train_phase):
    with tf.variable_scope("HW_SE_NET"):
        with tf.variable_scope("conv1"):
            inputs = conv(inputs, 32, 5, 1)
            inputs = max_pooling(inputs, 3, 2)
            inputs = batchnorm(inputs, train_phase, "BN")
            inputs = relu(inputs)
        with tf.variable_scope("conv2"):
            inputs = conv(inputs, 32, 5, 1)
            inputs = max_pooling(inputs, 3, 2)
            inputs = batchnorm(inputs, train_phase, "BN")
            inputs = relu(inputs)
        with tf.variable_scope("conv3"):
            inputs = conv(inputs, 64, 5, 1)
            inputs = max_pooling(inputs, 3, 2)
            inputs = batchnorm(inputs, train_phase, "BN")
            inputs = relu(inputs)
            inputs = SE_Block(inputs)
        with tf.variable_scope("conv4"):
            inputs = conv(inputs, 128, 5, 1)
            inputs = max_pooling(inputs, 3, 2)
            inputs = batchnorm(inputs, train_phase, "BN")
            inputs = relu(inputs)
            inputs = SE_Block(inputs)
        with tf.variable_scope("haar_wavelet"):
            inputs = haar_wavelet_block(inputs)
        with tf.variable_scope("fully_connected"):
            logits = fullycon(inputs, 30)
            prediction = tf.nn.softmax(logits)
        return prediction

def resnet18(inputs, is_training):
    inputs = tf.image.resize_bicubic(inputs, [224, 224])
    inputs, _ = resnet_v1_18(inputs, num_classes=4, is_training=is_training)
    inputs = tf.nn.softmax(tf.squeeze(inputs, [1, 2]))
    return inputs

def alexnet(inputs, is_training):
    inputs = tf.image.resize_bicubic(inputs, [224, 224])
    inputs, _ = alexnet_v2(inputs, num_classes=4, is_training=is_training)
    inputs = tf.nn.softmax(inputs)
    return inputs

def vgg16(inputs, is_training):
    inputs = tf.image.resize_bicubic(inputs, [224, 224])
    inputs, _ = vgg_16(inputs, num_classes=4, is_training=is_training)
    inputs = tf.nn.softmax(inputs)
    return inputs

def googlenet(inputs, is_training):
    # inputs = tf.image.resize_bicubic(inputs, [224, 224])
    inputs, feature = inception_v2(inputs, num_classes=30, is_training=is_training)
    inputs = tf.nn.softmax(inputs)
    return inputs, feature