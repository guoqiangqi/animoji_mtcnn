#coding:utf-8
import numpy as np
import os
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import tensorflow as tf
from datetime import datetime
from mtcnn_model import O_Net,L_O_Net,read_multi_tfrecords,train_model,random_flip_images

pre_model_path = None
root_dir = os.path.dirname(os.path.realpath(__file__)) 
base_dir = os.path.join(root_dir, '../data/imglists/LONet_expand')
model_path = os.path.join(root_dir, '../models/LONet/LONet_expand')
logs_dir = os.path.join(root_dir, '../models/logs/LONet_expand')
# pre_model_path = os.path.join(root_dir, '../models/LONet_pre_0805/LONet2-22')
print('root_dir: ', root_dir)
print('base_dir: ', base_dir)
print('model_path: ', model_path)
print('logs_dir: ', logs_dir)
        
prefix = model_path
end_epoch = 20000
display = 500
base_lr = 0.001
BATCH_SIZE = 405
image_size = 48
radio_cls_loss = 1.0
radio_bbox_loss = 0.5
radio_landmark_loss = 1.0
radio_animoji_loss = 1.0
# LR_EPOCH = [6,14,20,30,50,80,120]
LR_EPOCH = [10,40,100,300,600,1000,2000,5000,10000]

net = 'LONet'
label_file = os.path.join(base_dir,'train_{}.txt'.format(net))
print(label_file) 
f = open(label_file, 'r')
num = len(f.readlines())
print('Total datasets is: ', num)
print(prefix)
gray = False
if gray:
    image_channel = 1
else:
    image_channel = 3
print('iamge channel: ', image_channel)
pos_dir = os.path.join(base_dir,'pos_landmark.tfrecord_shuffle')
part_dir = os.path.join(base_dir,'part_landmark.tfrecord_shuffle')
neg_dir = os.path.join(base_dir,'neg_landmark.tfrecord_shuffle')
landmark_dir = os.path.join(base_dir,'landmark_landmark.tfrecord_shuffle')
animoji_dir = os.path.join(base_dir,'animoji_landmark.tfrecord_shuffle')
dataset_dirs = [pos_dir,part_dir,neg_dir,landmark_dir,animoji_dir]
pos_radio = 1.0/9;part_radio = 1.0/9;landmark_radio=2.0/9;neg_radio=3.0/9;animoji_radio=2.0/9
pos_batch_size = int(np.ceil(BATCH_SIZE*pos_radio))
assert pos_batch_size != 0,'Batch Size Error '
part_batch_size = int(np.ceil(BATCH_SIZE*part_radio))
assert part_batch_size != 0,'Batch Size Error '        
neg_batch_size = int(np.ceil(BATCH_SIZE*neg_radio))
assert neg_batch_size != 0,'Batch Size Error '
landmark_batch_size = int(np.ceil(BATCH_SIZE*landmark_radio))
assert landmark_batch_size != 0,'Batch Size Error '
animoji_batch_size = int(np.ceil(BATCH_SIZE*animoji_radio))
assert animoji_batch_size != 0,'Batch Size Error '
batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size,animoji_batch_size]
image_batch, label_batch, bbox_batch,landmark_batch,animoji_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, image_size, gray)

input_image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, image_size, image_size, image_channel], name='input_image')
label = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='label')
bbox_target = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 4], name='bbox_target')
landmark_target = tf.placeholder(tf.float32,shape=[BATCH_SIZE,10],name='landmark_target')
animoji_target = tf.placeholder(tf.float32,shape=[BATCH_SIZE,140],name='animoji_target')
cls_loss_op,bbox_loss_op,landmark_loss_op,animoji_loss_op,L2_loss_op,accuracy_op,recall_op = L_O_Net(input_image, label, bbox_target,landmark_target,animoji_target,training=True)

loss = radio_cls_loss*cls_loss_op + radio_bbox_loss*bbox_loss_op + radio_landmark_loss*landmark_loss_op + animoji_loss_op*radio_animoji_loss + L2_loss_op
train_op, lr_op = train_model(base_lr, loss, num, BATCH_SIZE, LR_EPOCH)

init = tf.global_variables_initializer()
sess = tf.Session()

saver = tf.train.Saver(max_to_keep=0)
sess.run(init)

tf.summary.scalar('cls_loss',cls_loss_op)
tf.summary.scalar('bbox_loss',bbox_loss_op)
tf.summary.scalar('landmark_loss',landmark_loss_op)
tf.summary.scalar('animoji_loss',animoji_loss_op)
tf.summary.scalar('cls_accuracy',accuracy_op)
tf.summary.scalar('cls_recall',recall_op)
summary_op = tf.summary.merge_all()

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
writer = tf.summary.FileWriter(logs_dir,sess.graph)
#begin 
coord = tf.train.Coordinator()
#begin enqueue thread
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
i = 0
#total steps
MAX_STEP = int(num / BATCH_SIZE + 1) * end_epoch
epoch = 0
sess.graph.finalize()    
try:
    if pre_model_path is not None:
        print('pre_training model: ', pre_model_path)
        saver.restore(sess, pre_model_path)
    for step in range(MAX_STEP):
        i = i + 1
        if coord.should_stop():
            break
        image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array,animoji_batch_array= \
                        sess.run([image_batch, label_batch, bbox_batch,landmark_batch,animoji_batch])
                        
        # for img_idx, img in enumerate(image_batch_array):
            # img = img *128 + 127.5
            # img = img.astype(np.uint8)
            # cv2.imwrite('outimg/{}.jpg'.format(img_idx), img)
        # exit()
        #random flip
        image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
        
        _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array,\
                                            bbox_target: bbox_batch_array,landmark_target:landmark_batch_array, \
                                            animoji_target: animoji_batch_array})
        
        if (step+1) % display == 0:
            cls_loss, bbox_loss,landmark_loss,animoji_loss,L2_loss,lr,acc,recall = \
                sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,animoji_loss_op,L2_loss_op,lr_op,accuracy_op,recall_op],
                feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: \
                bbox_batch_array, landmark_target: landmark_batch_array, \
                animoji_target: animoji_batch_array})
                
            print('{}: Step: {}, acc: {:.3}, recall: {:.3}, cls loss: {:.4}, bbox loss: {:.4}, landmark loss: {:.4}, animoji loss: {:.4}, L2 loss: {:.4}, lr:{:.4} '.format(
                datetime.now(), step+1, acc, recall, cls_loss, bbox_loss, landmark_loss, animoji_loss, L2_loss, lr))

        if i * BATCH_SIZE > num*2:
            epoch = epoch + 2
            i = 0
            cls_loss, bbox_loss,landmark_loss,animoji_loss,L2_loss,lr,acc,recall = \
                sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,animoji_loss_op,L2_loss_op,lr_op,accuracy_op,recall_op],
                feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: \
                bbox_batch_array, landmark_target: landmark_batch_array, animoji_target: animoji_batch_array})
            print('{}: {}, Step: {}, acc: {:.3}, recall: {:.3}, cls loss: {:.4}, bbox loss: {:.4}, landmark loss: {:.4}, animoji loss: {:.4}, L2 loss: {:.4}, lr:{:.4} '.format( \
                'Save model epoch', epoch, step+1, acc, recall, cls_loss, bbox_loss, landmark_loss, animoji_loss, L2_loss, lr))
            saver.save(sess, prefix, global_step=epoch)
        writer.add_summary(summary,global_step=step)
except tf.errors.OutOfRangeError:
    print('finish.')
finally:
    coord.request_stop()
    writer.close()
coord.join(threads)
sess.close()
print('finish.')
