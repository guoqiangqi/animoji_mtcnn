import sys
import os
# from utils import IoU
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../core'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
from collections import OrderedDict
from detection import detect_face
import shutil

def IoU(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h + 0.0
    ovr_Union = inter / (box_area + area - inter)
    ovr_Minimum = inter / np.minimum(box_area, area)
    return ovr_Union, ovr_Minimum   
    
if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = [os.path.join(root_dir, '../models/PNet/PNet-22'), 
                  os.path.join(root_dir, '../models/RNet/RNet-18'),
                  os.path.join(root_dir, '../models/LONet/LONet_expand-580')]
    image_txt = os.path.join(root_dir, '../data/300w/ImageList.txt')              
                  
    facenet = detect_face(model_path, True)
    
    print('read image list.')
    lines = None
    with open(image_txt,'r') as f:
        lines = f.readlines()
  
    num_images = len(lines)
    image_dir = os.path.dirname(image_txt)
    out_result = OrderedDict()
    out_data_dir = os.path.join(root_dir,'images','tmp')
    if os.path.exists(out_data_dir):
        shutil.rmtree(out_data_dir)
    os.mkdir(out_data_dir)
    
    not_detection_face = 0
    error_detection_face = 0
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        image_filename = os.path.join(image_dir, line[0])
        bbox = (line[1], line[3], line[2], line[4]) # To -> x1, y1, x2, y2        
        bbox = [float(_) for _ in bbox]
        bbox = list(map(int,bbox))
        w = bbox[2] - bbox[0] + 1
        h = bbox[3] - bbox[1] + 1
        
        landmark = list(map(float, line[5:]))
        assert(len(landmark) == 140)
        image = cv2.imread(image_filename)
        # pboxes,rboxes,boxes, landmarks, animojis,reg
        _, _, boxes_pred, landmarks_pred, animoji_pred, _ = facenet.predict(image)
        if boxes_pred is None:
            print('not detection:',image_filename)
            not_detection_face += 1
            detection_flag = False
        else:
            assert(len(bbox) == 4), str(len(bbox))
            assert(len(landmark) == 140), str(len(landmark))
            assert(boxes_pred.shape[0] == landmarks_pred.shape[0]), str(boxes_pred.shape)+str(landmarks_pred.shape)
            assert(boxes_pred.shape[0] == animoji_pred.shape[0]), str(boxes_pred.shape)+str(animoji_pred.shape)
            assert(boxes_pred.shape[1] == 5), str(boxes_pred.shape)
            assert(landmarks_pred.shape[1:] == (5,2)), str(landmarks_pred.shape)
            assert(animoji_pred.shape[1:] == (70,2)), str(animoji_pred.shape)
            assert(len(boxes_pred.shape) == 2), str(boxes_pred.shape)
            assert(len(landmarks_pred.shape) == 3), str(landmarks_pred.shape)
            assert(len(animoji_pred.shape) == 3), str(animoji_pred.shape)
            detection_flag = True
        
        bbox = np.asarray(bbox,dtype=np.float32)
        landmark = np.asarray(landmark,dtype=np.float32)
        boxes_pred = np.asarray(boxes_pred,dtype=np.float32)
        landmarks_pred = np.asarray(landmarks_pred,dtype=np.float32)
        animoji_pred = np.asarray(animoji_pred,dtype=np.float32)
        savez_path = os.path.join(out_data_dir, '{}.npz'.format(i))
        np.savez(savez_path,bbox=bbox,landmark=landmark,boxes_pred=boxes_pred,landmarks_pred=landmarks_pred,animoji_pred=animoji_pred)
        out_result[image_filename] = (line[0], savez_path)
        
        if detection_flag:
            iou,_ = IoU(bbox, boxes_pred)
            idx = np.argmax(iou)        
            if iou[idx] < 0.4:
                print('error: ', image_filename)
                error_detection_face += 1
        
        if (i+1) % 100 == 0:
            print('{}/{}  -- not detection: {}({})   error detection: {}({})'.format(i+1, num_images, \
               not_detection_face, not_detection_face/(i+1), error_detection_face, error_detection_face/(i+1)))
            # break
            
    print('Save detect result.')
    f = open(os.path.join(root_dir,'images','detect_result.txt'), 'w')
    f.write(str(out_result))
    f.close()
    # print(not_detection_face, num_images, not_detection_face/num_images)
    print('num_images: {}  -- not detection: {}({})   error detection: {}({})'.format(num_images, \
               not_detection_face, not_detection_face/num_images, error_detection_face, error_detection_face/num_images))
    print('-----------------   end   ------------------')