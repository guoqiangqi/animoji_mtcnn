import numpy as np
from numpy import array
import cv2
import tensorflow as tf

_left_eye_idx = [18,19,20,21,22,37,38,39,40,41,42,70]
_right_eye_idx = [23,24,25,26,27,43,44,45,46,47,48,69]
_right_eye_filpped_idx = [27,26,25,24,23,46,45,44,43,48,47,69]
_nose_idx = [28,29,30,31,32,33,34,35,36]
_mouse_idx = [49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68] 
_face_idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

_left_eye_idx = [x-1 for x in _left_eye_idx]
_right_eye_idx = [x-1 for x in _right_eye_idx]
_right_eye_filpped_idx = [x-1 for x in _right_eye_filpped_idx]
_nose_idx = [x-1 for x in _nose_idx]
_mouse_idx = [x-1 for x in _mouse_idx] 
_face_idx = [x-1 for x in _face_idx]

class Animoji():
    def __init__(self, animoji=None):
        if animoji is not None:
            assert(animoji.shape == (70,2)), str(animoji.shape)
            self.animoji = animoji
        else:
            self.animoji = np.zeros((70,2),dtype=np.float32)
        self.left_eye = self.animoji[_left_eye_idx]
        self.right_eye = self.animoji[_right_eye_idx]
        self.eyes = self.animoji[_left_eye_idx + _right_eye_idx]
        self.right_eye_filpped = self.animoji[_right_eye_filpped_idx]
        self.nose = self.animoji[_nose_idx]
        self.mouse = self.animoji[_mouse_idx]
        self.face = self.animoji[_face_idx]
  
    def get_animoji(self):
        self.animoji[_left_eye_idx] = self.left_eye
        self.animoji[_right_eye_idx] = self.right_eye
        self.animoji[_face_idx] = self.face
        self.animoji[_nose_idx] = self.nose
        self.animoji[_mouse_idx] = self.mouse
        return self.animoji

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _convert_to_example_simple(image_example, image_buffer):
    class_label = image_example['label']
    landmark = image_example['landmark'] 
      
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _int64_feature(class_label),
        'image/landmark': _float_feature(landmark),
    }))
    return example

def _process_image_withoutcoder(filename,size):
    image = cv2.imread(filename)
    image = cv2.resize(image, (size,size))
    image_data = image.tostring()
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width
    
def add_to_tfrecord(filename, image_example, tfrecord_writer, size):
    image_data, height, width = _process_image_withoutcoder(filename, size)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())

def get_dataset(imagelist, landmark_length):
    dataset = []
    for line in imagelist:
        info = line.strip().split(' ')
        data_example = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])
        if len(info) > 2:
            assert(len(info) == (landmark_length+2))
            data_example['landmark'] = list(map(float, info[2:]))
        else:
            data_example['landmark'] = [-1 for _ in range(landmark_length)]
        
        dataset.append(data_example)

    return dataset

def ratate(bbox, ratate_vector, img=None, points=None):
    left_eye = ratate_vector[0]
    right_eye = ratate_vector[1]
    x_dis, y_dist = right_eye - left_eye
    assert(x_dis > 0)
    angle = np.arctan2(y_dist, x_dis)*180/np.pi
    
    center = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    if img is not None:
        img_rotated_by_alpha = cv2.warpAffine(img.copy(), rot_mat,(img.shape[1],img.shape[0]))
    else:
        img_rotated_by_alpha = None
    
    if points is None:
        new_points_ = None
    else:
        new_points_ = []
        for points in points:
            points_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in points])
            new_points_.append(points_)
                 
    return (img_rotated_by_alpha, new_points_, (angle, center))

def rectangel(points, imgw, imgh):
    _x1,_y1 = np.min(points,axis=0)
    _x2,_y2 = np.max(points,axis=0)
    _w = _x2 - _x1 + 1
    _h = _y2 - _y1 + 1
    
    _x1 = max(0, _x1 - _w*0.1)
    _y1 = max(0, _y1 - _h*0.1)
    _x2 = min(imgw-1, _x2 + _w*0.1)
    _y2 = min(imgh-1, _y2 + _h*0.1)
    return _x1, _y1, _x2, _y2
   
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
    
def convert_to_square(bbox):
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox
    
def center_point(points):
    p1 = np.min(points,axis=0)
    p2 = np.max(points,axis=0)
    center = (p1 + p2)/2
    w, h = p2 - p1 + 1
    return center.astype(np.int32),(w,h)
    
if __name__ == '__main__':
    image = cv2.imread('../test03.jpg')
    landmark = [[341.01495817, 225.30015886],
                [410.52078259, 268.54534593],
                [350.76140836, 285.87877005],
                [323.4578793,  315.43765283],
                [390.45477861, 328.14139211]]
    landmark = np.asarray(landmark,dtype=np.float32)
    box = np.asarray([300.34860645, 173.70040955, 473.25332832, 380.84998456],dtype=np.float32)

    cv2.rectangle(image, (int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255))
            
    for x,y in landmark:
        cv2.circle(image, (int(x+0.5),int(y+0.5)), 3, (0,0,255))
        
    cv2.imwrite('test.jpg',image)
    
    img, landmark_, _ = ratate(image, box, landmark)
    
    cv2.rectangle(img, (int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255))
            
    for x,y in landmark_:
        cv2.circle(img, (int(x+0.5),int(y+0.5)), 3, (0,0,255))
        
    cv2.imwrite('test2.jpg',img)