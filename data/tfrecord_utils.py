import tensorflow as tf
import cv2
from PIL import Image

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
    bbox = image_example['bbox']
    roi = bbox['roi']
    landmark = bbox['landmark']
    animoji = bbox['animoji']  
      
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _int64_feature(class_label),
        'image/roi': _float_feature(roi),
        'image/landmark': _float_feature(landmark),
        'image/animoji':_float_feature(animoji)
    }))
    return example

def _process_image_withoutcoder(filename):
    image = cv2.imread(filename)
    image_data = image.tostring()
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width
    
def add_to_tfrecord(filename, image_example, tfrecord_writer):
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())

def get_dataset(imagelist):
    dataset = []
    for line in imagelist:
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])
        
        if len(info) == 6:
            bbox['roi'] = list(map(float, info[2:6]))
        else:
            bbox['roi'] = [0 for _ in range(4)]
            
        if len(info)== 12:
            bbox['landmark'] = list(map(float, info[2:12]))
        else:
            bbox['landmark'] = [0 for _ in range(10)]
            
        if len(info) == 142:
            bbox['animoji'] = list(map(float, info[2:142]))
        else:
            bbox['animoji'] = [0 for _ in range(140)]

        data_example['bbox'] = bbox
        dataset.append(data_example)

    return dataset

