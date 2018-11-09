import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data'))
from data_utils import IoU

from collections import OrderedDict
import numpy as np
from numpy import array
import cv2
from utils import ratate,rectangel, Animoji
import shutil

root_dir = os.path.dirname(os.path.realpath(__file__))

f = open(os.path.join(root_dir, 'images', 'detect_result.txt'),'r')
out_result = eval(f.read())
f.close()

face_data = []
eye_data = []
nose_data = []
mouse_data = []
face_rectangel = []
eye_rectangel = []
nose_rectangel = []
mouse_rectangel = []

output_image = os.path.join(root_dir, 'images', 'rotated_image')
if os.path.exists(output_image):
    shutil.rmtree(output_image)
os.mkdir(output_image)

num_of_images = len(out_result)
for i, (filepath, (filename,npz))  in enumerate(out_result.items()):
    if (i+1) % 100 == 0:
        print('{}/{} images done'.format(i+1, num_of_images))
    # if i > 10: break
    Array = np.load(npz)
    bbox = Array['bbox']
    animoji = Array['landmark']
    boxes_pred = Array['boxes_pred']
    landmarks_pred = Array['landmarks_pred']
    animojis_pred = Array['animoji_pred']
    if len(boxes_pred.shape) == 0:
        assert(len(landmarks_pred.shape) == 0), str(landmarks_pred)
        assert(len(animojis_pred.shape) == 0), str(animojis_pred)
        # not_detect_face.append(filepath)
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
        # max_iou.append(iou[idx])
        # error_detect_face.append(filepath)
        continue
        
    img = cv2.imread(filepath)
    ratate_vector = landmarks_pred[idx][0:2]
    out_points = [landmarks_pred[idx], animoji.reshape(70,2), animojis_pred[idx]]
    result = ratate(boxes_pred[idx], ratate_vector, points=out_points, img=img)
    img_rotated, (landmark_pred_rotated, animoji_rotated, animoji_pred_rotated),_ = result
    Animoji_rotated = Animoji(animoji_rotated)
    Animoji_pred_rotated = Animoji(animoji_pred_rotated)
    
    output_image_path = os.path.join(output_image, filename)
    if not os.path.exists(os.path.dirname(output_image_path)):
        os.makedirs(os.path.dirname(output_image_path))
    cv2.imwrite(output_image_path, img_rotated)
    imgh,imgw,imgc = img_rotated.shape
    face_flipped_by_x = cv2.flip(img_rotated, 1)
    
    name, ext = os.path.splitext(filename)
    flipped_name = name+'_filpped'+ext
    output_flipped_image_path = os.path.join(output_image, flipped_name)
    cv2.imwrite(output_flipped_image_path, face_flipped_by_x)
    
    left_eye_point = Animoji_rotated.left_eye
    right_eye_point = Animoji_rotated.right_eye
    eye_point = Animoji_rotated.eyes
    right_eye_filpped_point = Animoji_rotated.right_eye_filpped
    nose_point = Animoji_rotated.nose
    mouse_point = Animoji_rotated.mouse
    face_point = Animoji_rotated.face
    
    landmark_left_eye = landmark_pred_rotated[0]
    landmark_right_eye = landmark_pred_rotated[1]
    landmark_nose = landmark_pred_rotated[2]
    landmark_left_mouse = landmark_pred_rotated[3]
    landmark_right_mouse = landmark_pred_rotated[4]
    Lxe = landmark_right_eye[0] - landmark_left_eye[0] + 1
    Lxm = landmark_right_mouse[0] - landmark_left_mouse[0] + 1
    Ly = landmark_left_mouse[1] - landmark_left_eye[1] + 1
    W, H = boxes_pred[idx][2:4] - boxes_pred[idx][0:2] + 1
    assert(Lxe > 0 and Lxm > 0 and Ly > 0 and W > 0 and H > 0), 'lxe: {}, lxm: {}, ly: {}, w: {}, h: {}'.format(Lxe,Lxm,Ly,W,H)

    # face data    
    x1,y1,x2,y2,_ =  boxes_pred[idx]
    # if x1 < 0.2 and y1 < 0.0 and x2 < 0.2 and y2 < 0.2:
    dx1, dy1, dx2, dy2 = W*0.2, -H*0.1, W*0.2, H*0.2
    x1 = max(0, x1 - dx1)
    y1 = max(0, y1 - dy1)
    x2 = min(imgw-1, x2 + dx2)
    y2 = min(imgh-1, y2 + dy2)
    face_box = (x1,y1,x2,y2)
    face_data.append((filename, face_box, face_point))
    
    _face_box = rectangel(face_point, imgw, imgh)
    face_rectangel.append((filename, _face_box, bbox.tolist()))
    
    # eye data  
    lx, ly =  landmark_left_eye
    rx, ry =  landmark_right_eye
    # if x1 < 0.8 and y1 < 0.7 and x2 < 0.8 and y2 < 0.6:
    dx1, dy1, dx2, dy2 = Lxe*0.8, Ly*0.7, Lxe*0.8, Ly*0.6
    lx1 = max(0, lx - dx1)
    ly1 = max(0, ly - dy1)
    lx2 = min(imgw-1, lx + dx2)
    ly2 = min(imgh-1, ly + dy2)

    rx = imgw - rx
    rx1 = max(0, rx - dx1)
    ry1 = max(0, ry - dy1)
    rx2 = min(imgw-1, rx + dx2)
    ry2 = min(imgh-1, ry + dy2)
    
    eye_left_box = (lx1, ly1, lx2, ly2)
    eye_right_box = (rx1, ry1, rx2, ry2)  
    eye_data.append((filename, eye_left_box, left_eye_point))
    right_eye_filpped_point[:,0] = imgw - right_eye_filpped_point[:,0]
    eye_data.append((flipped_name, eye_right_box, right_eye_filpped_point))
    
    _left_eye_box = rectangel(left_eye_point, imgw, imgh) 
    _right_eye_box = rectangel(right_eye_point, imgw, imgh)    
    eye_rectangel.append((filename, _left_eye_box, _right_eye_box, bbox.tolist()))
    
    # nose data
    cx, cy = landmark_nose
    # if x1 < 0.6 and y1 < 0.9 and x2 < 0.6 and y2 < 0.4:
    dx1,dy1,dx2,dy2 = Lxe*0.6, Ly*0.9, Lxe*0.6, Ly*0.4
    x1 = max(0, cx - dx1)
    y1 = max(0, cy - dy1)
    x2 = min(imgw-1, cx + dx2)
    y2 = min(imgh-1, cy + dy2)
    nose_box = (x1,y1,x2,y2)
    nose_data.append((filename, nose_box, nose_point))
    
    _nose_box = rectangel(nose_point, imgw, imgh)  
    nose_rectangel.append((filename, _nose_box, bbox.tolist()))
    
    # mouse data
    landmark_mouse = landmark_pred_rotated[3:]
    x1,y1 = np.min(landmark_mouse,axis=0)
    x2,y2 = np.max(landmark_mouse,axis=0)
    # if x1 < 0.35 and y1 < 0.45 and x2 < 0.35 and y2 < 0.6:
    dx1,dy1,dx2,dy2 = Lxm*0.35, Ly*0.45, Lxm*0.35, Ly*0.6
    
    x1 = max(0, x1 - dx1)
    y1 = max(0, y1 - dy1)
    x2 = min(imgw-1, x2 + dx2)
    y2 = min(imgh-1, y2 + dy2)

    mouse_box = (x1,y1,x2,y2)
    mouse_data.append((filename, mouse_box, mouse_point))
    
    _mouse_box = rectangel(mouse_point, imgw, imgh)  
    mouse_rectangel.append((filename, _mouse_box, bbox.tolist()))

def saveImageList(savename, data, assert_shape=None):      
    f = open(savename,'w')
    for filename, box, point in data:
        x1,y1,x2,y2 = box
        if assert_shape is not None:
            assert(point.shape == assert_shape)
        point_list = point.reshape(-1).tolist()
        point_str = map(str, point_list)
        point_str = ' '.join(point_str)
        f.write('{} {} {} {} {} {}\n'.format(filename, int(x1+0.5), int(x2+0.5), int(y1+0.5), int(y2+0.5), point_str))
    f.close()
    
def saveRectangelList(savename, data):      
    f = open(savename,'w')
    for filename, bbox_, bbox in data:
        assert(len(bbox_) == 4)
        assert(len(bbox) == 4)
        bbox_ = [int(x+0.5) for x in bbox_]
        bbox_str_ = map(str, bbox_)
        bbox_str_ = ' '.join(bbox_str_)
        
        bbox = [int(x+0.5) for x in bbox]
        bbox_str = map(str, bbox)
        bbox_str = ' '.join(bbox_str)
        f.write('{} {} {}\n'.format(filename, bbox_str_, bbox_str))
    f.close()
    
def saveEyeRectangelList(savename, data):      
    f = open(savename,'w')
    for filename, left_bbox_, right_bbox_, bbox in data:
        assert(len(left_bbox_) == 4)
        assert(len(right_bbox_) == 4)
        assert(len(bbox) == 4)
        left_bbox_ = [int(x+0.5) for x in left_bbox_]
        left_bbox_str_ = map(str, left_bbox_)
        left_bbox_str_ = ' '.join(left_bbox_str_)
        
        right_bbox_ = [int(x+0.5) for x in right_bbox_]
        right_bbox_str_ = map(str, right_bbox_)
        right_bbox_str_ = ' '.join(right_bbox_str_)
        
        bbox = [int(x+0.5) for x in bbox]
        bbox_str = map(str, bbox)
        bbox_str = ' '.join(bbox_str)
        f.write('{} {} {} {}\n'.format(filename, left_bbox_str_, right_bbox_str_, bbox_str))
    f.close()
    
saveImageList(os.path.join(output_image, 'faceImageList.txt'), face_data, (17,2))
saveImageList(os.path.join(output_image, 'eyeImageList.txt'), eye_data, (12,2))
saveImageList(os.path.join(output_image, 'noseImageList.txt'), nose_data, (9,2))
saveImageList(os.path.join(output_image, 'mouseImageList.txt'), mouse_data, (20,2))

saveRectangelList(os.path.join(output_image, 'faceRectangelList.txt'), face_rectangel)
saveEyeRectangelList(os.path.join(output_image, 'eyeRectangelList.txt'), eye_rectangel)
saveRectangelList(os.path.join(output_image, 'noseRectangelList.txt'), nose_rectangel)
saveRectangelList(os.path.join(output_image, 'mouseRectangelList.txt'), mouse_rectangel)

print('-------------   end   ---------------')
