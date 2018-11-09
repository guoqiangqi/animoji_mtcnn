# -*- coding: utf-8 -*-

import xml.dom.minidom
import os
import cv2

save_dir = './training_data/landmarks_70/'
# if not os.path.exists(save_dir):
    # os.mkdir(save_dir)
# f = open(os.path.join(save_dir, 'ImageList_3.txt'), 'w')

#打开xml文档
DOMTree = xml.dom.minidom.parse(os.path.join(save_dir, 'labels_ibug_all_w300W_pupil.xml'))
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
    image_tmp=cv2.imread(os.path.join('/home-ex/tclhk/guoqiang/Animoji/data/training_data/landmarks_70',filename))
    img_h,img_w,_=image_tmp.shape
    #图片中gt
    bbox=img.getElementsByTagName('box')[0]
    top=int(bbox.getAttribute('top'))
    left =int(bbox.getAttribute('left'))
    width = int(bbox.getAttribute('width'))
    height =int(bbox.getAttribute('height'))

    bbox_buffer=[]
    landmark_buffer=[]

    # data_buffer.append(left)
    # data_buffer.append(left+width)
    # data_buffer.append(top)
    # data_buffer.append(top+height)
    #print(data_buffer)
    #特征点（70）
    landmarks=bbox.getElementsByTagName('part')
    #print(len(landmarks))
    x_min=left+width
    x_max=left
    y_min=top+height
    y_max=top
    
    i=0
    for landmark in landmarks:
        x=float(landmark.getAttribute('x'))
        y=float(landmark.getAttribute('y'))

        # cv2.putText(image_tmp,i,(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
        cv2.putText(image_tmp, str(i), (int(x),int(y)), cv2.FONT_HERSHEY_TRIPLEX, 0.3,
                    color=(255, 0, 255))
        cv2.circle(image_tmp,(int(x),int(y)),1,(0,255,0))
        i=i+1
        
        x_min=min(x_min,x)
        x_max=max(x_max,x)
        y_min=min(y_min,y)
        y_max=max(y_max,y)

        landmark_buffer.append(x)
        landmark_buffer.append(y)
    # print(landmark_buffer)
    
    cv2.namedWindow(filename,cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename,image_tmp)
    k=cv2.waitKey(0)
    if k&0xff ==ord("q"):
        cv2.destroyAllWindows()
        exit()
    if k&0xff ==ord("n"):
        pass  
        
    
    bbox_buffer.append(min(max(int(x_min-5),0),img_w))
    bbox_buffer.append(min(max(int(x_max+5),0),img_w))
    bbox_buffer.append(min(max(int(y_min-5),0),img_h))
    bbox_buffer.append(min(max(int(y_max+5),0),img_h))

    # f.write(str(filename) + ' ')

    # for i in range(len(bbox_buffer)):
        # f.write(str(bbox_buffer[i]) + ' ')

    # for i in range(len(landmark_buffer)):
        # f.write(str(landmark_buffer[i])+' ')
    # f.write('\t\n')

# f.close()