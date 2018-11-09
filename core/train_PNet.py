#coding:utf-8
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
from datetime import datetime
from mtcnn_model import P_Net,read_single_tfrecord,train_model,random_flip_images

root_dir = os.path.dirname(os.path.realpath(__file__)) 
base_dir = os.path.join(root_dir, '../data/imglists/PNet')
model_path = os.path.join(root_dir, '../models/PNet/PNet')
logs_dir = os.path.join(root_dir, '../models/logs/PNet')
        
prefix = model_path
end_epoch = 30
display = 100
base_lr = 0.01
BATCH_SIZE = 384
image_size = 12
radio_cls_loss = 1.0
radio_bbox_loss = 0.5
radio_landmark_loss = 0.5
LR_EPOCH = [6,14,20]

net = 'PNet'
image_size = 12
label_file = os.path.join(base_dir,'train_{}.txt'.format(net))
print(label_file) 
f = open(label_file, 'r')
num = len(f.readlines())
print('Total datasets is: ', num)
print(prefix)

dataset_dir = os.path.join(base_dir,'train_%s.tfrecord_shuffle' % net)
print(dataset_dir)
image_batch, label_batch, bbox_batch,landmark_batch,_ = read_single_tfrecord(dataset_dir, BATCH_SIZE, image_size)

input_image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, image_size, image_size, 3], name='input_image')
label = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='label')
bbox_target = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 4], name='bbox_target')
landmark_target = tf.placeholder(tf.float32,shape=[BATCH_SIZE,10],name='landmark_target')
cls_loss_op,bbox_loss_op,landmark_loss_op,L2_loss_op,accuracy_op, recall_op = P_Net(input_image, label, bbox_target,landmark_target,training=True)

loss = radio_cls_loss*cls_loss_op + radio_bbox_loss*bbox_loss_op + radio_landmark_loss*landmark_loss_op + L2_loss_op
train_op, lr_op = train_model(base_lr, loss, num, BATCH_SIZE, LR_EPOCH)

init = tf.global_variables_initializer()
sess = tf.Session()

saver = tf.train.Saver(max_to_keep=0)
sess.run(init)

tf.summary.scalar('cls_loss',cls_loss_op)
tf.summary.scalar('bbox_loss',bbox_loss_op)
tf.summary.scalar('landmark_loss',landmark_loss_op)
tf.summary.scalar('cls_accuracy',accuracy_op)
tf.summary.scalar('cls_accuracy',recall_op)
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
    for step in range(MAX_STEP):
        i = i + 1
        if coord.should_stop():
            break
        image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = \
                        sess.run([image_batch, label_batch, bbox_batch,landmark_batch])
        #random flip
        image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
        
        _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array,\
                                            bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})
        
        if (step+1) % display == 0:
            cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc,recall = \
                sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op,recall_op],
                feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: \
                bbox_batch_array, landmark_target: landmark_batch_array})
                
            print('{}: Step: {}, accuracy: {:.3}, recall: {:.3}, cls loss: {:.4}, bbox loss: {:.4}, landmark loss: {:.4}, L2 loss: {:.4}, lr:{} '.format(
                datetime.now(), step+1, acc, recall, cls_loss, bbox_loss, landmark_loss, L2_loss, lr))

        if i * BATCH_SIZE > num*2:
            epoch = epoch + 2
            i = 0
            cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc,recall = \
                sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op,recall_op],
                feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: \
                bbox_batch_array, landmark_target: landmark_batch_array})
            print('Save model epoch: {}, Step: {}, accuracy: {:.3}, recall: {:.3}, cls loss: {:.4}, bbox loss: {:.4}, landmark loss: {:.4}, L2 loss: {:.4}, lr:{} '.format( \
                epoch, step+1, acc, recall, cls_loss, bbox_loss, landmark_loss, L2_loss, lr))
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
