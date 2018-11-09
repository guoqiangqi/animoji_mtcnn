import sys
import os
from collections import OrderedDict
import numpy as np
from numpy import array
import cv2
from utils import ratate, Animoji, IoU
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math


def center_point(points):
    p1 = np.min(points,axis=0)
    p2 = np.max(points,axis=0)
    center = (p1 + p2)/2
    w, h = p2 - p1 + 1
    return center.astype(np.int32),(w,h)

f = open('../images/detect_result.txt','r')
out_result = eval(f.read())
f.close()

not_detect_face = []
error_detect_face = []

diffs = []
for filepath, (filename,npz)  in out_result.items():
    # print(npz)
    Array = np.load(npz)
    bbox = Array['bbox']
    animoji = Array['landmark']
    boxes_pred = Array['boxes_pred']
    landmarks_pred = Array['landmarks_pred']
    animojis_pred = Array['animoji_pred']
    if len(boxes_pred.shape) == 0:
        assert(len(landmarks_pred.shape) == 0), str(landmarks_pred)
        assert(len(animojis_pred.shape) == 0), str(animojis_pred)
        not_detect_face.append(filename)
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
        error_detect_face.append(filename)
        continue
        
    box_pred = boxes_pred[idx][:-1]
    landmark_pred = landmarks_pred[idx]
    animoji_pred = animojis_pred[idx]
    animoji = animoji.reshape(70,2)
    animoji = Animoji(animoji)
    animoji_pred = Animoji(animoji_pred)
    
    left_eye_point = animoji.left_eye
    right_eye_point = animoji.right_eye
    nose_point = animoji.nose
    mouse_point = animoji.mouse
    face_point = animoji.face
    
    left_eye_point_pred = animoji_pred.left_eye
    right_eye_point_pred = animoji_pred.right_eye
    nose_point_pred = animoji_pred.nose
    mouse_point_pred = animoji_pred.mouse
    face_point_pred = animoji_pred.face
    
    landmark_left_eye = landmark_pred[0]
    landmark_right_eye = landmark_pred[1]
    landmark_nose = landmark_pred[2]
    landmark_left_mouse = landmark_pred[3]
    landmark_right_mouse = landmark_pred[4]
    landmark_mouse,_ = center_point(landmark_pred[3:])
    
    left_eye_center,_ = center_point(left_eye_point)
    right_eye_center,_ = center_point(right_eye_point)
    nose_center,_ = center_point(nose_point)
    mouse_center,_ = center_point(mouse_point)
    face_center,_ = center_point(face_point)
    
    left_eye_center_pred,_ = center_point(left_eye_point_pred)
    right_eye_center_pred,_ = center_point(right_eye_point_pred)
    nose_center_pred,_ = center_point(nose_point_pred)
    mouse_center_pred,(am,_) = center_point(mouse_point_pred)
    face_center_pred,(fw,fh) = center_point(face_point_pred)
    
    box_center_pred,(bw,bh) = center_point(box_pred.reshape(2,2))
    box_center,_ = center_point(bbox.reshape(2,2))
    
    lx = landmark_right_eye[0] - landmark_left_eye[0]
    lm = landmark_right_mouse[0] - landmark_left_mouse[0]
    ly = landmark_mouse[1] - landmark_left_eye[1]
    ax = right_eye_center_pred[0] - left_eye_center_pred[0]
    ay = mouse_center_pred[1] - left_eye_center_pred[1]
    # w1,h1 = face_point_pred
    lxy = np.asarray([lx, ly], dtype=np.float32)
    lmy = np.asarray([lm, ly], dtype=np.float32)
    axy = np.asarray([ax, ay], dtype=np.float32)
    amy = np.asarray([am, ay], dtype=np.float32)
    fwh = np.asarray([fw, fh], dtype=np.float32)
    bwh = np.asarray([bw, bh], dtype=np.float32)
    
    diff0 = (landmark_left_eye - left_eye_center)/lxy
    diff1 = (landmark_right_eye - right_eye_center)/lxy
    diff2 = (landmark_nose - nose_center)/lxy
    diff3 = (landmark_mouse - mouse_center)/ lmy 
    
    diff4 = (left_eye_center - left_eye_center_pred)/axy
    diff5 = (right_eye_center - right_eye_center_pred)/axy
    diff6 = (nose_center - nose_center_pred)/axy
    diff7 = (mouse_center - mouse_center_pred)/amy
       
    diff8 = (face_center - face_center_pred)/fwh
    diff9 = (box_center - face_center_pred)/fwh
    diff10 = (face_center - box_center_pred)/bwh
    diff11 = (box_center - box_center_pred)/bwh
    
    diffs.append((
                  np.sum(diff0*diff0),
                  np.sum(diff1*diff1),
                  np.sum(diff2*diff2),
                  np.sum(diff3*diff3),
                  np.sum(diff4*diff4),
                  np.sum(diff5*diff5),
                  np.sum(diff6*diff6),
                  np.sum(diff7*diff7),
                  np.sum(diff8*diff8),
                  np.sum(diff9*diff9),
                  np.sum(diff10*diff10),
                  np.sum(diff11*diff11)
                  ))
 
diffs = np.asarray(diffs,dtype=np.float32)
print(diffs.shape)
print(np.std(diffs,axis=0).reshape(3,4))    


num_miss = len(not_detect_face)
num_error = len(error_detect_face)
num_image = len(out_result)
print('Number not detect face: {}, error detect face: {},  detect image: {}, error rate: {:.2f}%'.format(\
        num_miss, num_error, num_image,((num_miss+num_error+0.0)/num_image*100)))




        

