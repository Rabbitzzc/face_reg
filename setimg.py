'''
从网上下下来的一些明星照片等等
就需要将灰度处理和人脸检测进行单独提取出来，对照片进行处理，然后保存
(视频则直接使用下面的函数)
'''
import sys
import os
import cv2
import dlib

detector = dlib.get_frontal_face_detector() #获取人脸分类器

# 设置训练集目录,定义输入目录，输出目录，

# 个人脸目录
my_input_dir = './data/face_recog/my_faces'
my_output_dir = './data/my_faces'

other_input_dir = './data/face_recog/other_faces'
other_output_dir = './data/other_faces'
size = 64

# 网上下载

if not os.path.exists(my_output_dir):
    os.makedirs(my_output_dir)

if not os.path.exists(other_output_dir):
    os.makedirs(other_output_dir)


# 预处理数据，对原图像进行处理，并保存到制定目录下

def shootImg(input_dir, output_dir): 
    index = 1
    # os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下
    for (path, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                print('Being processed picture %s' % index)
                img_path = path+'/'+filename
                # 从文件读取图片，使用imread必须检测output是否为空

                # print(img_path)
                img = cv2.imread(img_path)
                # 转为灰度图片
                # print(img)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 使用detector进行人脸检测 dets为返回的结果
                dets = detector(gray_img, 1)

                # 使用enumerate 函数遍历序列中的元素以及它们的下标
                # 下标i即为人脸序号
                # left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
                # top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    y1 = d.bottom() if d.bottom() > 0 else 0
                    x2 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0
                    # img[y:y+h,x:x+w]
                    face = img[x1:y1, x2:y2]
                    # 调整图片的尺寸
                    face = cv2.resize(face, (size, size))
                    cv2.imshow('image', face)
                    # 保存图片，固定名称格式
                    cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                index += 1




# 对个人头像进行处理
# shootImg(my_input_dir, my_output_dir)

# 对训练数据进行头像的处理
shootImg(other_input_dir, other_output_dir)


key = cv2.waitKey(30) & 0xff
if key == 27:
    sys.exit(0)

'''
sys.exit()的退出比较优雅，调用后会引发SystemExit异常，
可以捕获此异常做清理工作。os._exit()直接将python解释器退出，余下的语句不会执行。
'''
