import numpy as np
from ops import *

def conv_(inputs, w, b):
    return tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME") + b

def max_pooling(inputs):
    return tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

def vggnet(inputs, vgg_path):
    inputs = tf.reverse(inputs, [-1]) - np.array([103.939, 116.779, 123.68])
    para = np.load(vgg_path+"vgg16.npy", encoding="latin1").item()
    F = {}
    inputs = relu(conv_(inputs, para["conv1_1"][0], para["conv1_1"][1]))
    inputs = relu(conv_(inputs, para["conv1_2"][0], para["conv1_2"][1]))
    F["conv1_2"] = inputs
    inputs = max_pooling(inputs)
    inputs = relu(conv_(inputs, para["conv2_1"][0], para["conv2_1"][1]))
    inputs = relu(conv_(inputs, para["conv2_2"][0], para["conv2_2"][1]))
    F["conv2_2"] = inputs
    inputs = max_pooling(inputs)
    inputs = relu(conv_(inputs, para["conv3_1"][0], para["conv3_1"][1]))
    inputs = relu(conv_(inputs, para["conv3_2"][0], para["conv3_2"][1]))
    inputs = relu(conv_(inputs, para["conv3_3"][0], para["conv3_3"][1]))
    F["conv3_3"] = inputs
    inputs = max_pooling(inputs)
    inputs = relu(conv_(inputs, para["conv4_1"][0], para["conv4_1"][1]))
    inputs = relu(conv_(inputs, para["conv4_2"][0], para["conv4_2"][1]))
    inputs = relu(conv_(inputs, para["conv4_3"][0], para["conv4_3"][1]))
    F["conv4_3"] = inputs
    return F

def transform(inputs, y1, y2, alpha):
    inputs = tf.reverse(inputs, [-1]) - np.array([103.939, 116.779, 123.68])
    inputs = relu(conditional_instance_norm(conv("conv1", inputs, 9, 3, 32, 1), "cin1", y1, y2, alpha))
    inputs = relu(conditional_instance_norm(conv("conv2", inputs, 3, 32, 64, 2), "cin2", y1, y2, alpha))
    inputs = relu(conditional_instance_norm(conv("conv3", inputs, 3, 64, 128, 2), "cin3", y1, y2, alpha))
    inputs = ResBlock("res1", inputs, 3, 128, 128, y1, y2, alpha)
    inputs = ResBlock("res2", inputs, 3, 128, 128, y1, y2, alpha)
    inputs = ResBlock("res3", inputs, 3, 128, 128, y1, y2, alpha)
    inputs = ResBlock("res4", inputs, 3, 128, 128, y1, y2, alpha)
    inputs = ResBlock("res5", inputs, 3, 128, 128, y1, y2, alpha)
    inputs = upsampling("up1", inputs, 128, 64, y1, y2, alpha)
    inputs = upsampling("up2", inputs, 64, 32, y1, y2, alpha)
    inputs = sigmoid(conditional_instance_norm(conv("last", inputs, 9, 32, 3, 1), "cinout", y1, y2, alpha)) * 255
    return inputs


