import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data'))
from data_utils import IoU

from collections import OrderedDict
import numpy as np
from numpy import array
import cv2
from utils import ratate
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math

left_eye_idx = [18,19,20,21,22,37,38,39,40,41,42,70]
right_eye_idx = [23,24,25,26,27,43,44,45,46,47,48,69]
nose_idx = [28,29,30,31,32,33,34,35,36]
mouse_idx = [49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68] 
face_idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

left_eye_idx = [x-1 for x in left_eye_idx]
right_eye_idx = [x-1 for x in right_eye_idx]
nose_idx = [x-1 for x in nose_idx]
mouse_idx = [x-1 for x in mouse_idx] 
face_idx = [x-1 for x in face_idx]

f = open('detect_result.txt','r')
out_result = eval(f.read())
f.close()

not_detect_face = []
error_detect_face = []
max_iou = []
dists = []
face_rate = []
for filepath, (filename,npz)  in out_result.items():
    Array = np.load(npz)
    bbox = Array['bbox']
    animoji = Array['landmark']
    boxes_pred = Array['boxes_pred']
    landmarks_pred = Array['landmarks_pred']
    animojis_pred = Array['animoji_pred']
    if len(boxes_pred.shape) == 0:
        assert(len(landmarks_pred.shape) == 0), str(landmarks_pred)
        assert(len(animojis_pred.shape) == 0), str(animojis_pred)
        not_detect_face.append(filepath)
        continue
        
    assert(len(bbox) == 4), str(bbox.shape)
    assert(len(animoji) == 140), str(animoji.shape)
    assert(boxes_pred.shape[0] == landmarks_pred.shape[0]), str(boxes_pred.shape)+str(landmarks_pred.shape)
    assert(boxes_pred.shape[0] == animojis_pred.shape[0]), str(boxes_pred.shape)+str(animojis_pred.shape)
    assert(boxes_pred.shape[1] == 5), str(boxes_pred.shape)
    assert(landmarks_pred.shape[1:] == (5,2)), str(landmarks_pred.shape)
    assert(animojis_pred.shape[1:] == (70,2)), str(animojis_pred.shape)
    assert(len(boxes_pred.shape) == 2), str(boxes_pred.shape)
    assert(len(landmarks_pred.shape) == 3), str(landmarks_pred.shape)
    assert(len(animojis_pred.shape) == 3), str(animojis_pred.shape)
    
    
    iou = IoU(bbox, boxes_pred)
    idx = np.argmax(iou) 
    
    if iou[idx] < 0.4:
        max_iou.append(iou[idx])
        error_detect_face.append(filepath)
        continue
        
    # img = cv2.imread(filepath)
    result = ratate(boxes_pred[idx], landmarks_pred[idx], points=[animoji.reshape(70,2), animojis_pred[idx]])
    landmark_pred_rotated, img_rotated, (animoji_rotated, animoji_pred_rotated) = result
    
    center = np.asarray([[(boxes_pred[idx][0]+boxes_pred[idx][2])/2, (boxes_pred[idx][1]+boxes_pred[idx][3])/2]],dtype=np.float32)
    # print(center.shape)
    left_eye_point = animoji_rotated[left_eye_idx]
    right_eye_point = animoji_rotated[right_eye_idx]
    nose_point = animoji_rotated[nose_idx]
    mouse_point = animoji_rotated[mouse_idx]
    face_point = animoji_rotated[face_idx]
    
    landmark_left_eye = landmark_pred_rotated[0]
    landmark_right_eye = landmark_pred_rotated[1]
    landmark_nose = landmark_pred_rotated[2]
    landmark_left_mouse = landmark_pred_rotated[3]
    landmark_right_mouse = landmark_pred_rotated[4]
    lx = landmark_right_eye[0] - landmark_left_eye[0] + 1
    ly = landmark_left_mouse[1] - landmark_left_eye[1] + 1 
    # print(lx, ly)
    assert(lx >0 and ly > 0), 'lx: {}, ly: {}'.format(lx, ly)
    
    p1 = np.min(nose_point,axis=0)
    p2 = np.max(nose_point,axis=0)
    
    dx1, dy1 = landmark_nose - p1
    dx2, dy2 = p2 - landmark_nose
 

   # p1 = np.min(face_point,axis=0)
    # p2 = np.max(face_point,axis=0)
    
    # p3 = boxes_pred[idx][0:2]
    # p4 = boxes_pred[idx][2:4]
    
    # dx1, dy1 = p3 - p1
    # dx2, dy2 = p2 - p4
    # Lx, Ly = p4 - p3 + 1

    face_rate.append((dx1/lx, dy1/ly, dx2/lx, dy2/ly))
 
 
num_miss = len(not_detect_face)
num_error = len(error_detect_face)
num_image = len(out_result)
print('Number not detect face: {}, error detect face: {},  detect image: {}, error rate: {:.2f}%'.format(\
        num_miss, num_error, num_image,((num_miss+num_error+0.0)/num_image*100)))
# # print('iou', max(max_iou), min(max_iou))
# print(max_iou)
# print(max(dists),'   ', min(dists))
face_rate = np.asarray(face_rate)
print(np.max(face_rate,axis=0))

xaxis = [x*0.01 for x in range(200)]
x1counts = [0 for _ in range(200)]
y1counts = [0 for _ in range(200)]
x2counts = [0 for _ in range(200)]
y2counts = [0 for _ in range(200)]

nc,na = 0, 0

for x1,y1,x2,y2 in face_rate:
    x1counts[int(x1*100)] += 1
    y1counts[int(y1*100)] += 1
    x2counts[int(x2*100)] += 1
    y2counts[int(y2*100)] += 1
    if x1 < 0.6 and y1 < 0.9 and x2 < 0.6 and y2 < 0.4:
        nc +=1
    na += 1
print(na, nc, nc/na)    
x1counts = np.asarray(x1counts, dtype=np.float32)
y1counts = np.asarray(y1counts, dtype=np.float32)
x2counts = np.asarray(x2counts, dtype=np.float32)
y2counts = np.asarray(y2counts, dtype=np.float32)   
# xcounts = np.log(xcounts+0.1)
# ycounts = np.log(ycounts+0.1)
plt.plot(xaxis, x1counts, 'r*')
plt.plot(xaxis, y1counts, 'b*')
plt.plot(xaxis, x2counts, 'r.')
plt.plot(xaxis, y2counts, 'b.')
plt.ylim(0,50)
plt.xlim(0,1)
plt.savefig('plot.png')
# print(x1counts)
# print(y1counts)
# print(x2counts)
# print(y2counts)

        

