import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import random
import sys
import time
import shutil
import tensorflow as tf
from utils import add_to_tfrecord, get_dataset
from gen_face_data import gen_face_bbox_data
from gen_eye_data import gen_eye_bbox_data
from gen_mouse_data import gen_mouse_bbox_data
from gen_nose_data import gen_nose_bbox_data
from gen_landmark_data import GenLandmarkData
    
def run(imagelist, tf_filename, landmark_length, size, shuffling=False):

    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    dataset = get_dataset(imagelist, landmark_length)
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        random.shuffle(dataset)

    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if (i+1) % 5000 == 0:
                print('\r>> Converting image {}/{}'.format(i + 1, len(dataset)))
            filename = image_example['filename']
            add_to_tfrecord(filename, image_example, tfrecord_writer, size)
    print('\nFinished converting dataset!')

    
def gen_tfrecords(
    gen_bbox_data,
    neg_list_file,
    pos_list_file,
    landmark_list_file,
    landmark_length,
    save_dir, 
    net,
    size,
    debug=False):
    
    tfrecords_output_dir = os.path.join(save_dir, net)
    image_output_dir = os.path.join(tfrecords_output_dir, 'image')
    if not os.path.exists(tfrecords_output_dir):
        os.makedirs(tfrecords_output_dir)
        
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
    
    files = gen_bbox_data(neg_list_file, pos_list_file, image_output_dir, debug=debug)
    _,_,landmark_list_file = GenLandmarkData(landmark_list_file, net, image_output_dir, \
                                             argument=True,debug=debug)

    with open(files[0], 'r') as f:
        pos = f.readlines()

    with open(files[1], 'r') as f:
        neg = f.readlines()


    with open(landmark_list_file, 'r') as f:
        landmark = f.readlines()
    #write all data
    imageLists = [pos, neg, landmark]
    if not os.path.exists(tfrecords_output_dir):
        os.mkdir(tfrecords_output_dir)
        
    with open(os.path.join(tfrecords_output_dir, "train_{}.txt".format(net)), "w") as f:
        print('Number positive: {}, negative: {}, landmark: {}'.format(len(pos), len(neg), len(landmark)))
        for i in np.arange(len(pos)):
            f.write(pos[i])
        for i in np.arange(len(neg)):
            f.write(neg[i])
        for i in np.arange(len(landmark)):
            f.write(landmark[i])

            
    tf_filenames = [
        os.path.join(tfrecords_output_dir,'pos_landmark.tfrecord'),
        os.path.join(tfrecords_output_dir,'neg_landmark.tfrecord'),
        os.path.join(tfrecords_output_dir,'landmark_landmark.tfrecord'),
    ]    
    
    for imgs, files in zip(imageLists, tf_filenames):
        run(imgs, files, landmark_length, size, shuffling=True)
        
    
if __name__ == '__main__': 
    debug = True
    debug = False
    
    size = 38
    save_dir = './train_data_debug' if debug else './train_data'        
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    print('save_dir: ', save_dir)
        
    print('\n### face')
    gen_bbox_data = gen_face_bbox_data
    neg_list_file = 'face_detection/neg.txt'
    pos_list_file = 'rotated_image/faceRectangelList.txt'
    landmark_list_file = 'rotated_image/faceImageList.txt'
    landmark_length = 34
    files = gen_tfrecords(gen_bbox_data, neg_list_file, pos_list_file, landmark_list_file, landmark_length, save_dir, 'face', size, debug=debug)
    
    print('\n### eye')
    gen_bbox_data = gen_eye_bbox_data
    pos_list_file = 'rotated_image/eyeRectangelList.txt'
    landmark_list_file = 'rotated_image/eyeImageList.txt'
    landmark_length = 24
    files = gen_tfrecords(gen_bbox_data, neg_list_file, pos_list_file, landmark_list_file, landmark_length, save_dir, 'eye', size, debug=debug)
    
    print('\n### nose')
    gen_bbox_data = gen_nose_bbox_data
    pos_list_file = 'rotated_image/noseRectangelList.txt'
    landmark_list_file = 'rotated_image/noseImageList.txt'
    landmark_length = 18
    files = gen_tfrecords(gen_bbox_data, neg_list_file, pos_list_file, landmark_list_file, landmark_length, save_dir, 'nose', size, debug=debug)
    
    print('\n### mouse')
    gen_bbox_data = gen_mouse_bbox_data
    pos_list_file = 'rotated_image/mouseRectangelList.txt'
    landmark_list_file = 'rotated_image/mouseImageList.txt'
    landmark_length = 40
    files = gen_tfrecords(gen_bbox_data, neg_list_file, pos_list_file, landmark_list_file, landmark_length, save_dir, 'mouse', size, debug=debug)