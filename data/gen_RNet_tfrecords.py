import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import random
import sys
import time
import shutil
import tensorflow as tf
from tfrecord_utils import add_to_tfrecord, get_dataset
from gen_24net_data import gen_RNet_bbox_data
from generateLandmark import GenLandmarkData
    
def run(imagelist, tf_filename, shuffling=False):
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    dataset = get_dataset(imagelist)
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        random.shuffle(dataset)

    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if (i+1) % 1000 == 0:
                print('\r>> Converting image {}/{}'.format(i + 1, len(dataset)))
            filename = image_example['filename']
            add_to_tfrecord(filename, image_example, tfrecord_writer)
    print('\nFinished converting the MTCNN dataset!')

    
def gen_RNet_tfrecords(
    bbox_anno_file,
    bbox_im_dir,
    save_dir,
    landmark_anno_file,
    tfrecords_output_dir,
    model_path,
    debug=False):
    
    size = 24
    net = 'RNet'
    
    # pos_list_file, neg_list_file, part_list_file = files
    files = gen_RNet_bbox_data(bbox_anno_file, bbox_im_dir, save_dir, model_path, debug=debug)
    _,_,landmark_list_file = GenLandmarkData(landmark_anno_file, net, size, save_dir, \
                                             argument=True,debug=debug)
    
    with open(files[0], 'r') as f:
        pos = f.readlines()

    with open(files[1], 'r') as f:
        neg = f.readlines()

    with open(files[2], 'r') as f:
        part = f.readlines()

    with open(landmark_list_file, 'r') as f:
        landmark = f.readlines()
    
    if not os.path.exists(tfrecords_output_dir):
        os.mkdir(tfrecords_output_dir)    
    #write all data
    with open(os.path.join(tfrecords_output_dir, "train_{}.txt".format(net)), "w") as f:
        print(len(neg))
        print(len(pos))
        print(len(part))
        print(len(landmark))
        for i in np.arange(len(pos)):
            f.write(pos[i])
        for i in np.arange(len(neg)):
            f.write(neg[i])
        for i in np.arange(len(part)):
            f.write(part[i])
        for i in np.arange(len(landmark)):
            f.write(landmark[i])

        
    imageLists = [pos, neg, part, landmark]
    
    tf_filenames = [
        os.path.join(tfrecords_output_dir,'pos_landmark.tfrecord'),
        os.path.join(tfrecords_output_dir,'part_landmark.tfrecord'),
        os.path.join(tfrecords_output_dir,'neg_landmark.tfrecord'),
        os.path.join(tfrecords_output_dir,'landmark_landmark.tfrecord'),
    ]    
    
    for imgs, files in zip(imageLists, tf_filenames):
        run(imgs, files, shuffling=True)
        
    
if __name__ == '__main__': 
    root_dir = os.path.dirname(os.path.realpath(__file__)) 
    anno_file = os.path.join(root_dir, 'widerface/wider_face_train_bbx_gt.txt')
    im_dir = os.path.join(root_dir, 'widerface/WIDER_train/images')
    save_dir = os.path.join(root_dir, '24')
    train_txt = os.path.join(root_dir, 'lfw/trainImageList.txt')
    tfrecords_output_dir = os.path.join(root_dir, 'imglists/RNet')
    model_path = os.path.join(root_dir, '../models/PNet/PNet-26')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    if os.path.exists(tfrecords_output_dir):
        shutil.rmtree(tfrecords_output_dir)
    
    gen_RNet_tfrecords(anno_file,       \
                       im_dir,           \
                       save_dir,          \
                       train_txt,          \
                       tfrecords_output_dir,\
                       model_path,           \
                       debug=False)