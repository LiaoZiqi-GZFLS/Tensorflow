import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os

IMAGE_W = int(input("img_w:"))
IMAGE_H = int(input("img_h:"))
CONTENT_IMG = input("content picture:")
STYLE_IMG = input("style picture:")
OUTPUT_DIR = './results'
OUTPUT_IMG = 'result.png'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
#Random sound point to content img
INI_NOISE_RATIO = 0.7
#Right of content img and style img
CONTENT_STRENGTH = 1
STYLE_STRENGTH = 500
ITERATION = 5000

n_in = tf.Variable(tf.random_normal([1,3,3,1]))

tf.nn.relu(tf.nn.conv2d(n_in, n_wb[0], strides=[1,1,1,1], padding='SAME')+ n_wb[1])

tf.nn.avg_pool(n_in, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

weights = vgg_layers[i][0][0][0][0][0]
weights = tf.constant(weights)
bias = vgg_layers[i][0][0][0][0][1]
bias = tf.constant(np.reshape(bias, (bias.size)))

#get_weight_bias函数作用是从vgg_layers中获取权重和偏置。


def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][0][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][0][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias

#build_net函数用来创建神经网络结构。


def build_net(net_type, inputs, weights_bias):
    weights, bias = weights_bias
    if net_type == 'conv':
        conv = tf.nn.conv2d(inputs, weights, strides=[1,1,1,1], padding='SAME')
        return tf.nn.relu(conv + bias)
    elif net_type == 'pool':
        return tf.nn.avg_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#接下来就是根据模版图片生成风格图片的代码，代码如下：


# 加载模型
path = os.path.join(os.getcwd(), VGG_MODEL)
vgg_rawnet = scipy.io.loadmat(path)
vgg_layers = vgg_rawnet['layers'][0]

# 定义网络结构
net = { }
net['input'] = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, 3)).astype('float32'))
net['conv1_1'] = build_net('conv',net['input'],get_weight_bias(vgg_layers,0))
net['conv1_2'] = build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2))
net['avgpool1'] = build_net('pool',net['conv1_2'],None)
net['conv2_1'] = build_net('conv',net['avgpool1'],get_weight_bias(vgg_layers,5))
net['conv2_2'] = build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7))
net['avgpool2'] = build_net('pool',net['conv2_2'],None)
net['conv3_1'] = build_net('conv',net['avgpool2'],get_weight_bias(vgg_layers,10))
net['conv3_2'] = build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12))
net['conv3_3'] = build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14))
net['conv3_4'] = build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16))
net['avgpool3'] = build_net('pool',net['conv3_4'],None)
net['conv4_1'] = build_net('conv',net['avgpool3'],get_weight_bias(vgg_layers,19))
net['conv4_2'] = build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21))
net['conv4_3'] = build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23))
net['conv4_4'] = build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25))
net['avgpool4'] = build_net('pool',net['conv4_4'],None)
net['conv5_1'] = build_net('conv',net['avgpool4'],get_weight_bias(vgg_layers,28))
net['conv5_2'] = build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30))
net['conv5_3'] = build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,32))
net['conv5_4'] = build_net('conv',net['conv5_3'],get_weight_bias(vgg_layers,34))
net['avgpool5'] = build_net('pool',net['conv5_4'],None)

# 加载内容图像
content_img = scipy.misc.imread(CONTENT_IMG)
content_img = scipy.misc.imresize(content_img,(IMAGE_H,IMAGE_W))
content_img = np.reshape(content_img, ((1,) + content_img.shape))

# 加载风格图像
style_img = scipy.misc.imread(STYLE_IMG)
style_img = scipy.misc.imresize(style_img,(IMAGE_H,IMAGE_W))
style_img = np.reshape(style_img, ((1,) + style_img.shape))

# 生成噪音图像
input_img = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, 3)).astype('float32')

# 定义内容图像的代价
content_loss = CONTENT_STRENGTH * (2 * tf.nn.l2_loss(
        net['conv4_2'] - content_features) / content_features.size)

# 定义风格图像的代价
style_loss = STYLE_STRENGTH * loss_style(net, style_layers)

# 定义总代价
total_loss = content_loss + style_loss

# 使用Adam算法进行优化
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(total_loss)

# 开始迭代
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(ITERATION):
    sess.run(train_step, feed_dict={net['input']: input_img})
    if i % 100 == 0:
        mix = sess.run(net['input'], feed_dict={net['input']: input_img})
        print("iteration " + str(i))
        print("cost " + str(sess.run(total_loss, feed_dict={net['input']: input_img})))
        scipy.misc.imsave(os.path.join(OUTPUT_DIR, OUTPUT_IMG), mix[0])
