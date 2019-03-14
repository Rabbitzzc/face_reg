import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

# my_faces_path = './data/my_faces'
other_faces_path = './data/other_faces'
size = 64

# 重新使用视频获取的个人照片进行训练
my_faces_path = './data/my_faces_avi'

imgs = []
labs = []

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readData(path , h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top,bottom,left,right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)

readData(my_faces_path)
readData(other_faces_path)
# 将图片数据与标签转换成数组
imgs = np.array(imgs)
labs = np.array([[0,1] if lab == my_faces_path else [1,0] for lab in labs])
# 随机划分测试集与训练集
train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0,100))
# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# 将数据转换成小于1的数
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0

print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
# 图片块，每次取128张图片
batch_size = 128
num_batch = len(train_x) // 128

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
    # 第一层
    W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
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
    drop3_flat = tf.reshape(drop3, [-1, 8*16*32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512,2])
    bout = biasVariable([2])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

output = cnnLayer()  
predict = tf.argmax(output, 1)  
   
# 导入使用tf训练的模型，创造网络
saver = tf.train.import_meta_graph('./model/train_faces.model-900.meta')  
sess = tf.Session()  
# import_meta_graph train_faces.model-400.meta

'''
记住，import_meta_graph将定义在.meta的网络导入到当前图中。
所以，这会替你创造图/网络，但我们仍然需要导入在这张图上训练好的参数。
'''
saver.restore(sess, tf.train.latest_checkpoint('./model'))  
   
def is_my_face(image):  
    sess.run(tf.global_variables_initializer())
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})  
    if res[0] == 1:  
        return True  
    else:  
        return False  

#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()



cam = cv2.VideoCapture(0)  

while True:  
    _, img = cam.read()  
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 0)
    
    # cv2.namedWindow("毕设-个人人脸检测",0)
    # cv2.resizeWindow("毕设-个人人脸检测", 640, 480)

    # 待会要显示在屏幕上的字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    '''
    添加一些说明
    1. 添加退出说明
    2. 添加是否为自己说明
    '''
    
    # img = cv2.putText(img, "Q: quit", (20, 450), font, 0.8, (238, 197, 145), 1, cv2.LINE_AA)

    # cv2.namedWindow('Graduation design of face recognition',0)
    if not len(dets):
        print('Can`t get face.')
        cv2.putText(img, "No Face: can't get the faces", (20, 50), font, 1, (238, 44 ,44), 1, cv2.LINE_AA)
        cv2.putText(img, "l: quit", (20, 450), font, 0.8, (238, 197, 145), 1, cv2.LINE_AA)
        cv2.imshow('img', img)
        if cv2.waitKey(300)&0xFF == ord('l'):
            # 摄像头关闭
            # cam.release()
            cv2.destroyWindow()
            # break
            # cv2.close()
    else:
        cv2.namedWindow('Graduation design of face recognition',0)
        cv2.putText(img, "Q: quit", (20, 450), font, 0.8, (238, 197, 145), 1, cv2.LINE_AA)
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            face = img[x1:y1,x2:y2]
            # 调整图片的尺寸
            face = cv2.resize(face, (size,size))
            # 命令行输入，也可以使用cv2对摄像头进行画图
            print('Is this my face? %s' % is_my_face(face))

            # 先标出人脸数
            cv2.putText(img, "Faces: "+str(len(dets)), (20,50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            if(is_my_face(face)):
                cv2.putText(img, "Identify Success, you are zhouzhechao~~" , (x2, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                                (0 ,255 ,0), 2, 4) # 绿色显示为成功
            else:
                cv2.putText(img, "Identify Failure, you are not zhouzhechao!!" , (x2, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                                (0 ,0 ,255), 2, 4) # 红色显示为不成功

            cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)
            
            cv2.imshow('image',img)

            if cv2.waitKey(300)&0xFF == ord('q'):
                # 按q退出程序
                cam.release()
                cv2.destroyAllWindows()
            # key = cv2.waitKey(30) & 0xff
            # if key == 27:
            #     sys.exit(0)


# 当所有事准备完成，关闭摄像头
sess.close()