# 更改我的脸的文件名，省的手动更改,并没有使用该模型

import os
import sys

# 循环修改文件名
def changeFleName(dir): 
    i = 0
    for path, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            print(os.path.join(path, filename))  # 输出rootdir路径下所有文件（包含子文件）信息
            newName="zzc_"+str(i)+".jpg"
            # print(path)
            os.rename(os.path.join(path, filename), os.path.join(path, newName))
            i=i+1


if __name__ == "__main__":
    dir = './data/face_recog/my_faces'
    changeFleName(dir)
