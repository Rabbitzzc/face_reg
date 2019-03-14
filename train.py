"""
sklearn.model_selection.train_test_split随机划分训练集和测试集
官方文档链接：
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split

rain_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和testdata

https://blog.csdn.net/u011630575/article/details/78594012
sklearn工具包

TensorFlow是将复杂的数据结构传输至人工智能神经网中进行分析和处理过程的系统。
"""

import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

# 需要安装TensorFlow sklearn

# 定义个人脸目录 和 非个人(其他人)脸目录，个人为[0,1],其他人为[1,0],
# 纯下载照片训练模型
# my_faces_path = './data/my_faces'
other_faces_path = './data/other_faces'

# 重新使用视频获取的个人照片进行训练
my_faces_path = './data/my_faces_avi'

size = 64

imgs = []
labs = []

# 函数返回 分别表示四个方向上边界的长度
def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # /./表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

# 读取文件夹中的所有图片，把图片全部扩充为64*64的图片
def readData(path , h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top,bottom,left,right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分,增加原图像的边界
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)

readData(my_faces_path)
readData(other_faces_path)

# 将图片数据与标签转换成数组
imgs = np.array(imgs)
labs = np.array([[0,1] if lab == my_faces_path else [1,0] for lab in labs])

# 随机划分测试集与训练集，样本特征集，样本结果，样本占比，验证集占用5%
train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0,100))

# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# 将数据转换成小于1的数
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0

print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
# 图片块，每次取100张图片
batch_size = 100
num_batch = len(train_x) // batch_size

# 提供占位符，暂时储存变量
x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


# 权重
def weightVariable(shape):
    # 从截断的正态分布中输出随机值。 生成的值服从具有指定平均值和标准偏差的正态分布
    init = tf.random_normal(shape, stddev=0.01)
    # 可通过对其运行操作来改变其值的张量，tf.Variable 存在于单个 session.run 调用的上下文之外
    return tf.Variable(init)


def biasVariable(shape):
    init = tf.random_normal(shape)
    # 构造函数需要变量的初始值，它可以是任何类型和形状的Tensor
    return tf.Variable(init)

'''
分为SAME和VALID分别表示是否需要填充，
因为卷积完之后因为周围的像素没有卷积到，
因此一般是会出现卷积完的输出尺寸小于输入的现象的
'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    # 池化，池化的输入，池化创窗口大小，这里是2 2，返回一个Tensor
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def dropout(x, keep):
    # 训练数据，概率,使输入tensor中某些元素变为0，其它没变0的元素变为原来的1/keep_prob大小！
    return tf.nn.dropout(x, keep)

# 设计卷积神经层，三个卷积层和一个全连接层
def cnnLayer():
    # 第一层
    W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])

    # 卷积，tf.nn.relu()函数是将大于0的数保持不变，小于0的数置为0
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)

    # 池化
    pool1 = maxPool(conv1)

    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层，此时输入图片大小为30*30
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层
    Wf = weightVariable([8*16*32, 512])
    bf = biasVariable([512])
    # 函数的作用是将tensor变换为参数shape的形式，shape为一个列表形式
    drop3_flat = tf.reshape(drop3, [-1, 8*16*32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512,2])
    bout = weightVariable([2])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out


def cnnTrain():
    out = cnnLayer()

    # 按照 就是神经网络最后一层的输出，求平均值
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    # 
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))

    # 将loss与accuracy保存以供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 数据保存器的初始化，Saver类提供了向checkpoints文件保存和从checkpoints文件中恢复变量的相关方法，比如产品了checkoutpoint文件夹
    saver = tf.train.Saver()


    with tf.Session() as sess:
        # 在tensorflow中数据流图中的Op在得到执行之前,必须先创建Session对象,
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

        for n in range(10): # [0,1,2,...,9]
             # 每次取128(batch_size)张图片
            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x,y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75})
                summary_writer.add_summary(summary, n*num_batch+i)
                # 打印损失
                print(n*num_batch+i, loss)

                if (n*num_batch+i) % 100 == 0:
                    # 获取测试数据的准确率
                    acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
                    print(n*num_batch+i, acc)
                    # 准确率大于0.98时保存并退出
                    if acc > 0.98 and n > 2:
                        print("准备率大于0.98")
                        saver.save(sess, './model/train_faces.model', global_step=n*num_batch+i)
                        sys.exit(0)
        print('accuracy less 0.98, exited!')

cnnTrain()