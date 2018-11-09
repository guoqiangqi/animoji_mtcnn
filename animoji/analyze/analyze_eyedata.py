import sys
import os
from collections import OrderedDict
import numpy as np
from numpy import array
import cv2
from utils import ratate, Animoji, IoU, center_point
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math

f = open('../images/detect_result.txt','r')
out_result = eval(f.read())
f.close()

not_detect_face = []
error_detect_face = []
max_iou = []
dists = []
face_rate = []
_face_rate = []
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
    
    
    iou,_ = IoU(bbox, boxes_pred)
    idx = np.argmax(iou) 
    
    if iou[idx] < 0.4:
        max_iou.append(iou[idx])
        error_detect_face.append(filepath)
        continue
        
    ######
    result = ratate(boxes_pred[idx], landmarks_pred[idx], points=[animoji.reshape(70,2), landmarks_pred[idx]])
    img_rotated, (animoji_rotated, landmark_pred_rotated),_ = result
    
    animoji_rotated = Animoji(animoji_rotated)
    left_eye_point = animoji_rotated.left_eye
    right_eye_point = animoji_rotated.right_eye

    
    landmark_left_eye = landmark_pred_rotated[0]
    landmark_right_eye = landmark_pred_rotated[1]
    landmark_left_mouse = landmark_pred_rotated[3]
    lx = landmark_right_eye[0] - landmark_left_eye[0] + 1
    ly = landmark_left_mouse[1] - landmark_left_eye[1] + 1
    # print(lx, ly)
    assert(lx >0 and ly > 0), 'lx: {}, ly: {}'.format(lx, ly)
    
    
    lp1 = np.min(left_eye_point,axis=0)
    lp2 = np.max(left_eye_point,axis=0)
    rp1 = np.min(right_eye_point,axis=0)
    rp2 = np.max(right_eye_point,axis=0)
    
    ldx1, ldy1 = landmark_left_eye - lp1
    ldx2, ldy2 = lp2 - landmark_left_eye
    rdx2, rdy1 = landmark_right_eye - rp1
    rdx1, rdy2 = rp2 - landmark_right_eye

    face_rate.append((ldx1/lx, ldy1/ly, ldx2/lx, ldy2/ly, (lp1, lp2, lx, ly)))
    face_rate.append((rdx1/lx, rdy1/ly, rdx2/lx, rdy2/ly, (rp1, rp2, lx, ly)))
    
    #######
    animojis_class = Animoji(animojis_pred[idx])
    left_eye_center,_ = center_point(animojis_class.left_eye)
    right_eye_center,_ = center_point(animojis_class.right_eye)
    ratete_vector = [left_eye_center, right_eye_center]
    
    result = ratate(boxes_pred[idx], ratete_vector, points=[animoji.reshape(70,2), animojis_pred[idx]])
    img_rotated, (animoji_rotated, animoji_pred_rotated),_ = result
    animoji_rotated = Animoji(animoji_rotated)
    animoji_pred_rotated = Animoji(animoji_pred_rotated)
    
    left_eye_point = animoji_rotated.left_eye
    right_eye_point = animoji_rotated.right_eye
    
    left_eye_point_pred = animoji_pred_rotated.left_eye   
    right_eye_point_pred = animoji_pred_rotated.right_eye
    mouse_point_pred = animoji_pred_rotated.mouse
    
    left_eye_center,_ = center_point(left_eye_point_pred)
    right_eye_center,_ = center_point(right_eye_point_pred)
    mouse_center,_ = center_point(mouse_point_pred)
    
    lx = right_eye_center[0] - left_eye_center[0] + 1
    ly = mouse_center[1] - left_eye_center[1] + 1
    
    lp1 = np.min(left_eye_point,axis=0)
    lp2 = np.max(left_eye_point,axis=0)
    rp1 = np.min(right_eye_point,axis=0)
    rp2 = np.max(right_eye_point,axis=0)
    
    ldx1, ldy1 = left_eye_center - lp1
    ldx2, ldy2 = lp2 - left_eye_center
    rdx1, rdy1 = right_eye_center - rp1
    rdx2, rdy2 = rp2 - right_eye_center

    _face_rate.append((ldx1/lx, ldy1/ly, ldx2/lx, ldy2/ly, (lp1, lp2, lx, ly)))
    _face_rate.append((rdx1/lx, rdy1/ly, rdx2/lx, rdy2/ly, (rp1, rp2, lx, ly)))
    
    
num_miss = len(not_detect_face)
num_error = len(error_detect_face)
num_image = len(out_result)
print('Number not detect face: {}, error detect face: {},  detect image: {}, error rate: {:.2f}%'.format(\
        num_miss, num_error, num_image,((num_miss+num_error+0.0)/num_image*100)))

xaxis = [x*0.01 for x in range(200)]
x1counts = [0 for _ in range(200)]
y1counts = [0 for _ in range(200)]
x2counts = [0 for _ in range(200)]
y2counts = [0 for _ in range(200)]

nc,na = 0, 0

for x1,y1,x2,y2,_ in face_rate:
    idx = min(199, int(x1*100))
    x1counts[idx] += 1
    idx = min(199, int(y1*100))
    y1counts[idx] += 1
    idx = min(199, int(x2*100))
    x2counts[idx] += 1
    idx = min(199, int(y2*100))
    y2counts[idx] += 1
    if x1 < 0.8 and y1 < 0.7 and x2 < 0.8 and y2 < 0.4:
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

        

