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
    
    
    iou, _ = IoU(bbox, boxes_pred)
    idx = np.argmax(iou) 
    
    if iou[idx] < 0.4:
        max_iou.append(iou[idx])
        error_detect_face.append(filepath)
        continue

    #######
    result = ratate(boxes_pred[idx], landmarks_pred[idx][0:2], points=[animoji.reshape(70,2)])
    img_rotated, (animoji_rotated, ),_ = result
    animoji_rotated = Animoji(animoji_rotated)   
    face_point = animoji_rotated.face
   
    p1 = np.min(face_point,axis=0)
    p2 = np.max(face_point,axis=0)
    
    p3 = boxes_pred[idx][0:2]
    p4 = boxes_pred[idx][2:4]
    
    dx1, dy1 = p3 - p1
    dx2, dy2 = p2 - p4
    Lx, Ly = p4 - p3 + 1

    face_rate.append((dx1/Lx, dy1/Ly, dx2/Lx, dy2/Ly, (p1, p2, p3, p4)))
    
    #################
    animojis_class = Animoji(animojis_pred[idx])
    left_eye_center,_ = center_point(animojis_class.left_eye)
    right_eye_center,_ = center_point(animojis_class.right_eye)
    ratete_vector = [left_eye_center, right_eye_center]
    
    result = ratate(boxes_pred[idx], ratete_vector, points=[animoji.reshape(70,2), animojis_pred[idx]])
    img_rotated, (animoji_rotated, animoji_pred_rotated),_ = result
    animoji_rotated = Animoji(animoji_rotated)
    animoji_pred_rotated = Animoji(animoji_pred_rotated)
    
    face_point = animoji_rotated.face
    face_point_pred = animoji_pred_rotated.face
    
    p1 = np.min(face_point,axis=0)
    p2 = np.max(face_point,axis=0)
    _p3 = np.min(face_point_pred,axis=0)
    _p4 = np.max(face_point_pred,axis=0)
    
    dx1, dy1 = _p3 - p1
    dx2, dy2 = p2 - _p4
    Lx, Ly = _p4 - _p3 + 1

    _face_rate.append((dx1/Lx, dy1/Ly, dx2/Lx, dy2/Ly, (p1, p2, _p3, _p4)))
    
num_miss = len(not_detect_face)
num_error = len(error_detect_face)
num_image = len(out_result)
print('Number not detect face: {}, error detect face: {},  detect image: {}, error rate: {:.2f}%'.format(\
        num_miss, num_error, num_image,((num_miss+num_error+0.0)/num_image*100)))

# face_rate = np.asarray(face_rate)
# print(np.max(face_rate,axis=0))

# _face_rate = np.asarray(_face_rate)
# print(np.max(_face_rate,axis=0))

xaxis = [x*0.01 for x in range(200)]
x1counts = [0 for _ in range(200)]
y1counts = [0 for _ in range(200)]
x2counts = [0 for _ in range(200)]
y2counts = [0 for _ in range(200)]

nc,na = 0, 0
ovrs = []
for x1,y1,x2,y2,(p1, p2, p3, p4) in face_rate:
    x1counts[int(x1*100)] += 1
    y1counts[int((y1+0.5)*100)] += 1
    x2counts[int(x2*100)] += 1
    y2counts[int(y2*100)] += 1
    if x1 < 0.2 and y1 < -0.1 and x2 < 0.2 and y2 < 0.2:
        w, h = p2 - p1 + 1
        Lx, Ly = p4 - p3 + 1
        Lx = Lx*1.4
        Ly = Ly*1.1
        ovrs.append(((Lx*Ly)/(w*h)))
        nc +=1
    na += 1
    
x1counts = np.asarray(x1counts, dtype=np.float32)
y1counts = np.asarray(y1counts, dtype=np.float32)
x2counts = np.asarray(x2counts, dtype=np.float32)
y2counts = np.asarray(y2counts, dtype=np.float32)
ovrs = np.asarray(ovrs, dtype=np.float32)   
print(na, nc, nc/na)
print(np.mean(ovrs), np.std(ovrs))

plt.plot(xaxis, x1counts, 'r*', label='x1')
plt.plot(xaxis, y1counts, 'b*', label='y1')
plt.plot(xaxis, x2counts, 'r.', label='x2')
plt.plot(xaxis, y2counts, 'b.', label='y2')
plt.legend()
plt.ylim(0,50)
plt.xlim(0,0.5)
plt.savefig('face_plot1.png')
plt.clf()


xaxis = [x*0.01 for x in range(200)]
x1counts = [0 for _ in range(200)]
y1counts = [0 for _ in range(200)]
x2counts = [0 for _ in range(200)]
y2counts = [0 for _ in range(200)]

nc,na = 0, 0
ovrs = []
for x1,y1,x2,y2,(p1, p2, p3, p4) in _face_rate:
    x1counts[int(x1*100)] += 1
    y1counts[int(y1*100)] += 1
    x2counts[int(x2*100)] += 1
    y2counts[int(y2*100)] += 1
    if x1 < 0.3 and y1 < 0.3 and x2 < 0.3 and y2 < 0.3:
        w, h = p2 - p1 + 1
        Lx, Ly = p4 - p3 + 1
        Lx = Lx*1.6
        Ly = Ly*1.6
        ovrs.append(((Lx*Ly)/(w*h)))
        nc +=1
    na += 1
   
x1counts = np.asarray(x1counts, dtype=np.float32)
y1counts = np.asarray(y1counts, dtype=np.float32)
x2counts = np.asarray(x2counts, dtype=np.float32)
y2counts = np.asarray(y2counts, dtype=np.float32)   
ovrs = np.asarray(ovrs, dtype=np.float32)   
print(na, nc, nc/na)
print(np.mean(ovrs), np.std(ovrs))

plt.plot(xaxis, x1counts, 'r*', label='x1')
plt.plot(xaxis, y1counts, 'b*', label='y1')
plt.plot(xaxis, x2counts, 'r.', label='x2')
plt.plot(xaxis, y2counts, 'b.', label='y2')
plt.ylim(0,50)
plt.xlim(0,0.5)
plt.legend()
plt.savefig('face_plot2.png')

        

