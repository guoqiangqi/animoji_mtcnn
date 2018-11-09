import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import random
import cv2
num_keep_radio = 0.7

def prelu(inputs):
    alphas = tf.get_variable('alphas', shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg

def cls_ohem(cls_prob, label):
    zeros = tf.zeros_like(label)
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int = tf.cast(label_filter_invalid,tf.int32)
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    row = tf.range(num_row)*2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    valid_inds = tf.where(label < zeros,zeros,ones)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    loss = loss * valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)

def landmark_ohem(landmark_pred,landmark_target,label):
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label,-1),ones,zeros)
    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
    
def cal_accuracy(cls_prob,label):
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(label,tf.int64)
    cond = tf.where(tf.greater_equal(label_int,0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    
    cond = tf.where(tf.greater(label_picked,0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_picked,picked)
    pred_picked = tf.gather(pred_picked,picked)
    recall_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op,recall_op

    
def Net(inputs,landmark_dim, reuse = None, label=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        # weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(5e-5),
                        # weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        print('Net network shape')
        if reuse is None:
            reuse = ''
        print(inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope=reuse+'conv1')
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope=reuse+'pool1', padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope=reuse+'conv2')
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope=reuse+'pool2', padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=128,kernel_size=[3,3],stride=1,scope=reuse+'conv3')
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope=reuse+'pool3', padding='SAME')
        print(net.get_shape())
        # net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope=reuse+'conv4')
        # print(net.get_shape())
        # net = tf.transpose(net, perm=[0,3,1,2]) 
        # net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope=reuse+'avg_pool')
        # print(net.get_shape())        
        fc_flatten = slim.flatten(net)
        print(fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=100,scope=reuse+'fc1', activation_fn=prelu)
        print(fc1.get_shape())

        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope=reuse+'cls_fc',activation_fn=tf.nn.softmax)
        print(cls_prob.get_shape())

        landmark_pred = slim.fully_connected(fc1,num_outputs=landmark_dim,scope=reuse+'landmark_fc',activation_fn=None)
        print(landmark_pred.get_shape())        

        if training:
            cls_loss = cls_ohem(cls_prob,label)
            accuracy, recall = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss,landmark_loss,L2_loss,accuracy, recall
        else:
            return cls_prob,landmark_pred

            
def train_model(base_lr, loss, data_num, batch_size, lr_epoch):
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    boundaries = [int(epoch * data_num / batch_size) for epoch in lr_epoch]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(lr_epoch) + 1)]
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    # optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    optimizer = tf.train.AdamOptimizer(lr_op)
    train_op = optimizer.minimize(loss, global_step)

    return train_op, lr_op       
        
def read_single_tfrecord(tfrecord_file, landmark_length, batch_size, image_size, gray=False):
    filename_queue = tf.train.string_input_producer([tfrecord_file],shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    image_features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),#one image  one record
            'image/label': tf.FixedLenFeature([], tf.int64),
            'image/landmark': tf.FixedLenFeature([landmark_length],tf.float32)
        }
    )

    image = tf.decode_raw(image_features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [image_size, image_size, 3])
    if gray:
        image = tf.image.rgb_to_grayscale(image)
    image = (tf.cast(image, tf.float32)-127.5) / 128
    
    label = tf.cast(image_features['image/label'], tf.float32)
    landmark = tf.cast(image_features['image/landmark'],tf.float32)
    image, label,landmark = tf.train.batch(
        [image, label, landmark],
        batch_size=batch_size,
        num_threads=2,
        capacity=1 * batch_size
    )
    label = tf.reshape(label, [batch_size])
    landmark = tf.reshape(landmark,[batch_size,landmark_length])
    return image, label, landmark

def read_multi_tfrecords(tfrecord_files, landmark_length, batch_sizes, image_size, gray=False):
    pos_dir,neg_dir,landmark_dir = tfrecord_files
    pos_batch_size,neg_batch_size,landmark_batch_size = batch_sizes
    
    pos_image,pos_label,pos_landmark = read_single_tfrecord(pos_dir, landmark_length, pos_batch_size, image_size, gray)
    print(pos_image.get_shape())

    neg_image,neg_label,neg_landmark = read_single_tfrecord(neg_dir, landmark_length, neg_batch_size, image_size, gray)
    print(neg_image.get_shape())
    
    landmark_image,landmark_label,landmark_landmark = read_single_tfrecord(landmark_dir, landmark_length, landmark_batch_size, image_size, gray)
    print(landmark_image.get_shape())    

    images = tf.concat([pos_image,neg_image,landmark_image], 0, name='concat/image')
    print(images.get_shape())
    labels = tf.concat([pos_label,neg_label,landmark_label],0,name='concat/label')
    print(labels.get_shape())
    landmarks = tf.concat([pos_landmark,neg_landmark,landmark_landmark],0,name='concat/landmark')
    print(landmarks.get_shape())
    return images,labels,landmarks