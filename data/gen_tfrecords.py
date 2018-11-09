import os
import numpy as np
import random
import sys
import time
import shutil
import tensorflow as tf
import argparse
from tfrecord_utils import add_to_tfrecord, get_dataset
from gen_data import gen_PNet_bbox_data, gen_RNet_bbox_data, gen_ONet_bbox_data
from generateLandmark import GenLandmarkData


# def run(imagelist, net, output_dir, shuffling=False):
    # tf_filename = os.path.join(output_dir, 'train_{}.tfrecord'.format(net))
    # if tf.gfile.Exists(tf_filename):
        # print('Dataset files already exist. Exiting without re-creating them.')
        # return
    # # GET Dataset, and shuffling.
    # dataset = get_dataset(imagelist)
    # num = len(dataset)

    # if shuffling:
        # tf_filename = tf_filename + '_shuffle'
        # random.shuffle(dataset)

    # with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        # for i, image_example in enumerate(dataset):
            # if (i+1) % 1000 == 0:
                # print '\r>> Converting image {}/{}'.format(i + 1, len(dataset))
            # filename = image_example['filename']
            # _add_to_tfrecord(filename, image_example, tfrecord_writer)
    # print('\nFinished converting the MTCNN dataset!')
    
    # def run(imagelist, tf_filename, shuffling=False):

    # if tf.gfile.Exists(tf_filename):
        # print('Dataset files already exist. Exiting without re-creating them.')
        # return

    # dataset = get_dataset(imagelist)
    # if shuffling:
        # tf_filename = tf_filename + '_shuffle'
        # random.shuffle(dataset)

    # with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        # for i, image_example in enumerate(dataset):
            # if (i+1) % 5000 == 0:
                # print '\r>> Converting image {}/{}'.format(i + 1, len(dataset))
            # filename = image_example['filename']
            # _add_to_tfrecord(filename, image_example, tfrecord_writer)
    # print('\nFinished converting the MTCNN dataset!')

   
    
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
                print '\r>> Converting image {}/{}'.format(i + 1, len(dataset))
            filename = image_example['filename']
            add_to_tfrecord(filename, image_example, tfrecord_writer)
    print('\nFinished converting the MTCNN dataset!')
    
def gen_PNet_tfrecords(
    bbox_anno_file,
    bbox_im_dir,
    save_dir,
    landmark_anno_file,
    landmark_im_dir,
    tfrecords_output_dir,
    debug=False):
    
    size = 12
    net = "PNet"
    
    # pos_list_file, neg_list_file, part_list_file = files
    files = gen_PNet_bbox_data(bbox_anno_file, bbox_im_dir, save_dir, debug=debug)
    _,_,landmark_list_file = GenLandmarkData(landmark_anno_file, landmark_im_dir, \
                            net, save_dir, argument=True,debug=debug)

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
    
    if not os.path.exists(tfrecords_output_dir):
        os.makedirs(tfrecords_output_dir)
        
    f = open(os.path.join(tfrecords_output_dir,'train_{}.txt'.format(net)),'w')
    f.writelines(imagelist)
    f.close()
    
    tf_filename = os.path.join(tfrecords_output_dir, 'train_{}.tfrecord'.format(net))    
    run(imagelist, tf_filename, shuffling=True)
    
def write_tfrecords(files, tfrecords_output_dir, net):
    with open(files[0], 'r') as f:
        pos = f.readlines()

    with open(files[1], 'r') as f:
        neg = f.readlines()

    with open(files[2], 'r') as f:
        part = f.readlines()

    with open(files[3], 'r') as f:
        landmark = f.readlines()
               
    #write all data
    imageLists = [pos, neg, part, landmark]
    if not os.path.exists(tfrecords_output_dir):
        os.mkdir(tfrecords_output_dir)
        
    with open(os.path.join(tfrecords_output_dir, "train_{}.txt".format(net)), "w") as f:
        print len(neg)
        print len(pos)
        print len(part)
        print len(landmark)
        for i in np.arange(len(pos)):
            f.write(pos[i])
        for i in np.arange(len(neg)):
            f.write(neg[i])
        for i in np.arange(len(part)):
            f.write(part[i])
        for i in np.arange(len(landmark)):
            f.write(landmark[i])
    tf_filenames = [
        os.path.join(tfrecords_output_dir,'pos_landmark.tfrecord'),
        os.path.join(tfrecords_output_dir,'part_landmark.tfrecord'),
        os.path.join(tfrecords_output_dir,'neg_landmark.tfrecord'),
        os.path.join(tfrecords_output_dir,'landmark_landmark.tfrecord'),
    ]    
    
    for imgs, files in zip(imageLists, tf_filenames):
        run(imgs, files, shuffling=True)
    
def gen_RNet_tfrecords(
    bbox_anno_file,
    bbox_im_dir,
    save_dir,
    landmark_anno_file,
    landmark_im_dir,
    tfrecords_output_dir,
    model_path,
    debug=False):
    
    size = 24
    net = "RNet"
    
    # pos_list_file, neg_list_file, part_list_file = files
    files = gen_RNet_bbox_data(bbox_anno_file, bbox_im_dir, save_dir, model_path, debug=debug)
    _,_,landmark_list_file = GenLandmarkData(landmark_anno_file, landmark_im_dir, \
                            net, save_dir, argument=True,debug=debug)
    files.append(landmark_list_file)
    write_tfrecords(files, tfrecords_output_dir, net)
    
    # with open(files[0], 'r') as f:
        # pos = f.readlines()

    # with open(files[1], 'r') as f:
        # neg = f.readlines()

    # with open(files[2], 'r') as f:
        # part = f.readlines()

    # with open(landmark_list_file, 'r') as f:
        # landmark = f.readlines()
               
    # #write all data
    # imageLists = [pos, neg, part, landmark]
    # if not os.path.exists(tfrecords_output_dir):
        # os.mkdir(tfrecords_output_dir)
        
    # with open(os.path.join(tfrecords_output_dir, "train_{}.txt".format(net)), "w") as f:
        # print len(neg)
        # print len(pos)
        # print len(part)
        # print len(landmark)
        # for i in np.arange(len(pos)):
            # f.write(pos[i])
        # for i in np.arange(len(neg)):
            # f.write(neg[i])
        # for i in np.arange(len(part)):
            # f.write(part[i])
        # for i in np.arange(len(landmark)):
            # f.write(landmark[i])
    # tf_filenames = [
        # os.path.join(tfrecords_output_dir,'pos_landmark.tfrecord'),
        # os.path.join(tfrecords_output_dir,'part_landmark.tfrecord'),
        # os.path.join(tfrecords_output_dir,'neg_landmark.tfrecord'),
        # os.path.join(tfrecords_output_dir,'landmark_landmark.tfrecord'),
    # ]    
    
    # for imgs, files in zip(imageLists, tf_filenames):
        # run(imgs, files, shuffling=True)
    
def gen_ONet_tfrecords(
    bbox_anno_file,
    bbox_im_dir,
    save_dir,
    landmark_anno_file,
    landmark_im_dir,
    tfrecords_output_dir,
    model_path,
    debug=False):
    
    size = 48
    net = "ONet"
    
    # pos_list_file, neg_list_file, part_list_file = files
    files = gen_ONet_bbox_data(bbox_anno_file, bbox_im_dir, save_dir, model_path, debug=debug)
    _,_,landmark_list_file = GenLandmarkData(landmark_anno_file, landmark_im_dir, \
                            net, save_dir, argument=True,debug=debug)
    
    files.append(landmark_list_file)
    write_tfrecords(files, tfrecords_output_dir, net)
    # with open(files[0], 'r') as f:
        # pos = f.readlines()

    # with open(files[1], 'r') as f:
        # neg = f.readlines()

    # with open(files[2], 'r') as f:
        # part = f.readlines()

    # with open(landmark_list_file, 'r') as f:
        # landmark = f.readlines()
        
    # #write all data
    # imageLists = [pos, neg, part, landmark]
    # if not os.path.exists(tfrecords_output_dir):
        # os.mkdir(tfrecords_output_dir)
        
    # with open(os.path.join(tfrecords_output_dir, "train_{}.txt".format(net)), "w") as f:
        # print len(neg)
        # print len(pos)
        # print len(part)
        # print len(landmark)
        # for i in np.arange(len(pos)):
            # f.write(pos[i])
        # for i in np.arange(len(neg)):
            # f.write(neg[i])
        # for i in np.arange(len(part)):
            # f.write(part[i])
        # for i in np.arange(len(landmark)):
            # f.write(landmark[i])

    # tf_filenames = [
        # os.path.join(tfrecords_output_dir,'pos_landmark.tfrecord'),
        # os.path.join(tfrecords_output_dir,'part_landmark.tfrecord'),
        # os.path.join(tfrecords_output_dir,'neg_landmark.tfrecord'),
        # os.path.join(tfrecords_output_dir,'landmark_landmark.tfrecord'),
    # ]    
    
    # for imgs, files in zip(imageLists, tf_filenames):
        # run(imgs, files, shuffling=True)
        
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', 
                        nargs="+", default=-1, type=int)
    parser.add_argument('--mode', dest='mode', help='net type, can be pnet, rnet or onet', type=str)
    parser.add_argument('--debug', dest='debug', action='store_true')
    args = parser.parse_args()
    return args   
    
if __name__ == '__main__': 
    # debug = False
    
    args = parse_args()
    root_dir = os.path.dirname(os.path.realpath(__file__)) 
    im_dir = os.path.join(root_dir, 'widerface/WIDER_train/images')
    train_txt = os.path.join(root_dir, 'lfw/trainImageList.txt')
    train_root = os.path.join(root_dir, '')
    debug = args.debug
    
    assert(args.mode in ['PNet', 'RNet', 'ONet']), 'mode: ({}), can be pnet, rnet or onet.'.format(args.mode)
    print args.mode, args.epoch, args.debug
    if args.mode == 'PNet':
        anno_file = os.path.join(root_dir, 'widerface/wider_face_train.txt')
        save_dir = os.path.join(root_dir, '12')
        tfrecords_output_dir = os.path.join(root_dir, 'imglists/PNet')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(tfrecords_output_dir):
            shutil.rmtree(tfrecords_output_dir)
        gen_PNet_tfrecords(anno_file,      \
                           im_dir,          \
                           save_dir,         \
                           train_txt,         \
                           train_root,         \
                           tfrecords_output_dir,\
                           debug=debug)
                           
    elif args.mode == 'RNet':  
        # root_dir = os.path.dirname(os.path.realpath(__file__)) 
        anno_file = os.path.join(root_dir, 'widerface/wider_face_train_bbx_gt.txt')
        # im_dir = os.path.join(root_dir, 'widerface/WIDER_train/images')
        save_dir = os.path.join(root_dir, '24')
        # train_txt = os.path.join(root_dir, 'lfw/trainImageList.txt')
        # train_root = os.path.join(root_dir, '')
        tfrecords_output_dir = os.path.join(root_dir, 'imglists/RNet')
        assert(len(args.epoch) == 1)
        model_path = os.path.join(root_dir, '../models/PNet/PNet-{}'.format(args.epoch[0]))
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(tfrecords_output_dir):
            shutil.rmtree(tfrecords_output_dir)
        
        gen_RNet_tfrecords(anno_file,      \
                           im_dir,          \
                           save_dir,         \
                           train_txt,         \
                           train_root,         \
                           tfrecords_output_dir,\
                           model_path,           \
                           debug=debug)
                           
    elif args.mode == 'ONet':                   
        # root_dir = os.path.dirname(os.path.realpath(__file__)) 
        anno_file = os.path.join(root_dir, 'widerface/wider_face_train_bbx_gt.txt')
        # im_dir = os.path.join(root_dir, 'widerface/WIDER_train/images')
        save_dir = os.path.join(root_dir, '48')
        # train_txt = os.path.join(root_dir, 'lfw/trainImageList.txt')
        # train_root = os.path.join(root_dir, '')
        tfrecords_output_dir = os.path.join(root_dir, 'imglists/ONet')
        assert(len(args.epoch) == 2)
        model_path = [os.path.join(root_dir, '../models/PNet/PNet-{}'.format(args.epoch[0])),
                      os.path.join(root_dir, '../models/RNet/RNet-{}'.format(args.epoch[1]))]
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(tfrecords_output_dir):
            shutil.rmtree(tfrecords_output_dir)
        
        gen_ONet_tfrecords(anno_file,      \
                           im_dir,          \
                           save_dir,         \
                           train_txt,         \
                           train_root,         \
                           tfrecords_output_dir,\
                           model_path,           \
                           debug=debug)
    print 'End'