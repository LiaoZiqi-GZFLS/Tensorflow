import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import scipy.misc
import json
import heapq
import imageio
from skimage.transform import resize
 
VGG19_MAT_PATH = r"imagenet-vgg-verydeep-19.mat"
 
with open('./imagenet_1000_labels.json') as f:
    labels = json.load(f)
    labels = np.array(labels)
 
 
def _conv_layer(input, weight, bias):
    conv = tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)
 
 
def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
 
 
def preprocess(image, mean_pixel):
    '''简单预处理,全部图片减去平均值'''
    return image - mean_pixel
 
 
def net(in_x, data_path):
    """
    读取VGG模型参数,搭建VGG网络
    :param data_path: VGG模型文件位置
    :param input_image: 输入测试图像
    :return:
    """
    layers = (
        'conv1_1', 'conv1_2', 'pool1',
        'conv2_1', 'conv2_2', 'pool2',
        'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'pool3',
        'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool4',
        'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool5',
        'fc1', 'fc2', 'fc3',
        'softmax'
    )
    data = scipy.io.loadmat(data_path)
    # 数据预处理的均值
    mean = data["normalization"][0][0][0][0][0]
    net = {}
    current = in_x
    net["in_x"] = in_x  # 存储数据
    count = 0  # 计数存储
    for i in range(43):
        if str(data['layers'][0][i][0][0][0][0])[:4] == "relu":
            continue
        if str(data['layers'][0][i][0][0][0][0])[:4] == "pool":
            current = _pool_layer(current)
        elif str(data['layers'][0][i][0][0][0][0]) == "softmax":
            current = tf.nn.softmax(current)
        elif i == 37:
            shape = int(np.prod(current.get_shape()[1:]))
            current = tf.reshape(current, [-1, shape])
            kernels, bias = data['layers'][0][i][0][0][0][0]
            kernels = np.reshape(kernels, [-1, 4096])
            bias = bias.reshape(-1)
            current = tf.nn.relu(tf.add(tf.matmul(current, kernels), bias))
        elif i == 39:
            kernels, bias = data['layers'][0][i][0][0][0][0]
            kernels = np.reshape(kernels, [4096, 4096])
            bias = bias.reshape(-1)
            current = tf.nn.relu(tf.add(tf.matmul(current, kernels), bias))
        elif i == 41:
            kernels, bias = data['layers'][0][i][0][0][0][0]
            kernels = np.reshape(kernels, [4096, 1000])
            bias = bias.reshape(-1)
            current = tf.add(tf.matmul(current, kernels), bias)
        else:
            kernels, bias = data['layers'][0][i][0][0][0][0]
            # 注意VGG存储方式为[,]
            # kernels = np.transpose(kernels,[1,0,2,3])
            bias = bias.reshape(-1)  # 降低维度
            current = tf.nn.relu(_conv_layer(current, kernels, bias))
        net[layers[count]] = current  # 存储数据
        count += 1
    return net
 
 
# 返回 1 224 224 3 的数据
def read_image(path):
    #image = scipy.misc.imread(path, mode='RGB')
    image = imageio.v2.imread(path, mode='RGB')
    #image = scipy.misc.imresize(image, [224, 224])
    image = resize(image, [224, 224])
    image = np.expand_dims(image, 0).astype(np.float32)
    return image
 
 
def main():
    input_image = np.concatenate([
        read_image('./MIKU.jpg'),
        read_image('./p0.jpg'),
    ], axis=0)
    
    in_x = tf.placeholder(tf.float32, (None, 224, 224, 3))
    endpoints = net(in_x, VGG19_MAT_PATH)
 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logits_val = sess.run(endpoints, {
            in_x: input_image
        })["softmax"]
 
        # 取前三大概率的种类,由大到小
        labels_indexs = np.argsort(logits_val, axis=1)[:, -3:][:, ::-1]
        probs = np.sort(logits_val, axis=1)[:, -3:][:, ::-1]
        for label_index, prob in zip(labels_indexs, probs):
            print(labels[label_index], labels_indexs, prob)
 
 
if __name__ == '__main__':
    main()
