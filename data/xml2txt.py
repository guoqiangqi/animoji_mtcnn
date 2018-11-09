# -*- coding: utf-8 -*-

import xml.dom.minidom
import os
import cv2

save_dir = './training_data/landmarks_70/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
f = open(os.path.join(save_dir, 'ImageList.txt '), 'w')

#打开xml文档
DOMTree = xml.dom.minidom.parse(os.path.join(save_dir,'labels_ibug_all_w300W_pupil.xml'))
#获取xml文档对象
annotation = DOMTree.documentElement

#print(annotation.nodeName)

# filename = annotation.getElementsByTagName("images")[0]
# print(filename)
# print(filename.childNodes)

imgs=annotation.getElementsByTagName('image')
for img in imgs:
    #获取图片名
    filename=img.getAttribute('file')
    data_buffer=[filename]
    print(data_buffer[0])
    #图片中gt
    bbox=img.getElementsByTagName('box')[0]
    top=int(bbox.getAttribute('top'))
    left =int(bbox.getAttribute('left'))
    width = int(bbox.getAttribute('width'))
    height =int(bbox.getAttribute('height'))

    data_buffer.append(left)
    data_buffer.append(left+width)
    data_buffer.append(top)
    data_buffer.append(top+height)
    #print(data_buffer)
    #特征点（70）
    landmarks=bbox.getElementsByTagName('part')
    #print(len(landmarks))
    for landmark in landmarks:
        x=float(landmark.getAttribute('x'))
        y=float(landmark.getAttribute('y'))
        data_buffer.append(x)
        data_buffer.append(y)
    print(data_buffer)

    for i in range(len(data_buffer)):
        f.write(str(data_buffer[i])+' ')
    f.write('\t\n')

f.close()




#     loc = loc + [lefttopx, lefttopy, righttopx, righttopy, rightbottomx, rightbottomy, leftbottomx, leftbottomy]
#
# for i in range(len(loc)):
#     f.write(str(loc[i]) + ' ')
# f.write('\t\n')
# f.close()
'''
  test：top,left,w,h  and  left,right,top,bottom
'''

# pic1=cv2.imread('./1051618982_1.jpg')
# cv2.rectangle(pic1,(469,206),(469+216,206+216),(255,0,0))
# cv2.circle(pic1,(469,206),2,(0,0,255))
# cv2.imshow('pic1',pic1)
#
# pic2=cv2.imread('./Aaron_Guiel_0001.jpg')
# cv2.rectangle(pic2,(85,93),(172,181),(255,0,0))
# cv2.imshow('pic2',pic2)
#
# pic3=cv2.imread('./Aaron_Eckhart_0001.jpg')
# cv2.rectangle(pic3,(84,92),(161,169),(255,0,0))
# cv2.imshow('pic3',pic3)
#
# k=cv2.waitKey(0)
# if k ==ord('q'):
#     cv2.destroyAllWindows()
