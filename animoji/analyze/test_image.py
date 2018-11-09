from random import shuffle
import cv2
import numpy as np
import os

txt1 = 'rotated_image/faceImageList.txt'
txt2 = 'rotated_image/eyeImageList.txt'
txt3 = 'rotated_image/noseImageList.txt'
txt4 = 'rotated_image/mouseImageList.txt'

txts = [txt1,txt2,txt3,txt4]
for i, txt in enumerate(txts):
    lines = None
    with open(txt,'r') as f:
        lines = f.readlines()
    assert(lines is not None)
    shuffle(lines) 

    line = lines[0].strip().split(' ')
    img_path = line[0].replace('\\','/')

    # line[1:5] -> x1, x2, y1, y2
    bbox = (line[1], line[3], line[2], line[4]) # To -> x1, y1, x2, y2        
    bbox = [float(_) for _ in bbox]
    bbox = list(map(int,bbox))
    landmark = list(map(float, line[5:]))
    landmark = np.asarray(landmark,dtype=np.float32).reshape(-1,2)

    print(img_path)
    img_path = os.path.join('rotated_image', img_path)
    image = cv2.imread(img_path)
    cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))

    # idx = 5    
    for x,y in landmark:
        cv2.circle(image, (int(x),int(y)), 3, (0,0,255))
    cv2.imwrite('test{}.jpg'.format(i), image)


txt1 = 'rotated_image/faceRectangelList.txt'
txt2 = 'rotated_image/eyeRectangelList.txt'
txt3 = 'rotated_image/noseRectangelList.txt'
txt4 = 'rotated_image/mouseRectangelList.txt'

txts = [txt1,txt2,txt3,txt4]
for i, txt in enumerate(txts):
    lines = None
    with open(txt,'r') as f:
        lines = f.readlines()

    assert(lines is not None)
    shuffle(lines) 

    line = lines[0].strip().split(' ')
    img_path = line[0].replace('\\','/')
           
    _bbox = [float(x)+0.5 for x in line[1:5]]
    _bbox = list(map(int, _bbox))

    __bbox = None
    if len(line) > 9:
        assert(len(line) == 13)
        __bbox = [float(x)+0.5 for x in line[5:9]]
        __bbox = list(map(int, __bbox))
        
    bbox = [float(x)+0.5 for x in line[-4:]]
    bbox = list(map(int, bbox))

    print(img_path)
    img_path = os.path.join('rotated_image', img_path)
    image = cv2.imread(img_path)
    cv2.rectangle(image, (int(_bbox[0]),int(_bbox[1])),(int(_bbox[2]),int(_bbox[3])),(255,0,0))
    if __bbox is not None:
        cv2.rectangle(image, (int(__bbox[0]),int(__bbox[1])),(int(__bbox[2]),int(__bbox[3])),(255,0,0))
    cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))

    cv2.imwrite('rectangetest{}.jpg'.format(i), image)

