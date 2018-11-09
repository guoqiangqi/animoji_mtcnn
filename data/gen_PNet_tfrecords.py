import os
import numpy as np
import random
import sys
import time
import tensorflow as tf
from tfrecord_utils import add_to_tfrecord, get_dataset
from gen_12net_data import gen_PNet_bbox_data
from generateLandmark import GenLandmarkData

def run(imagelist, net, output_dir, shuffling=False):
    tf_filename = os.path.join(output_dir, 'train_{}.tfrecord'.format(net))
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # GET Dataset, and shuffling.
    dataset = get_dataset(imagelist)
    num = len(dataset)

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
    
def gen_PNet_tfrecords(
    bbox_anno_file,
    bbox_im_dir,
    save_dir,
    landmark_anno_file,
    output_directory,
    debug=False):
    
    size = 12
    net = 'PNet'
    
    # pos_list_file, neg_list_file, part_list_file = files
    files = gen_PNet_bbox_data(bbox_anno_file, bbox_im_dir, save_dir, debug=debug)
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
        
    nums = [len(neg), len(pos), len(part)]
    # ratio = [3, 1, 1]
    base_num = 250000
    base_num = min([int(len(neg)/3),len(pos), len(part), base_num])
    print(len(neg), len(pos), len(part), base_num)
    if len(neg) > base_num * 3:
        neg_keep = np.random.choice(len(neg), size=base_num * 3, replace=True)
    else:
        neg_keep = np.random.choice(len(neg), size=len(neg), replace=True)
    pos_keep = np.random.choice(len(pos), size=base_num, replace=True)
    part_keep = np.random.choice(len(part), size=base_num, replace=True)
    print(len(neg_keep), len(pos_keep), len(part_keep))
    imagelist = []
    for i in pos_keep:
        imagelist.append(pos[i])
    for i in neg_keep:
        imagelist.append(neg[i])
    for i in part_keep:
        imagelist.append(part[i])
    for item in landmark:
        imagelist.append(item)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    f = open(os.path.join(output_directory,'train_{}.txt'.format(net)),'w')
    f.writelines(imagelist)
    f.close()
        
    run(imagelist, net, output_directory, shuffling=True)
    
if __name__ == '__main__': 
    root_dir = os.path.dirname(os.path.realpath(__file__)) 
    anno_file = os.path.join(root_dir, 'widerface/wider_face_train.txt')
    im_dir = os.path.join(root_dir, 'widerface/WIDER_train/images')
    save_dir = os.path.join(root_dir, '12')
    train_txt = os.path.join(root_dir, 'lfw/trainImageList.txt')
    output_directory = os.path.join(root_dir, 'imglists/PNet')
    gen_PNet_tfrecords(anno_file,   \
                       im_dir,       \
                       save_dir,      \
                       train_txt,      \
                       output_directory,\
                       debug=False)
    
