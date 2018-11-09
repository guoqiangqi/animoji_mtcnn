#coding:utf-8
import numpy as np
import os
import cv2
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import tensorflow as tf
from datetime import datetime
from model import Net,read_multi_tfrecords,train_model

def run(dataset_dirs, landmark_length, model_path, num_class, train_args):           
    prefix = model_path+'/'
    end_epoch = train_args.get('end_epoch',20000)
    display = train_args.get('display',100)
    base_lr = train_args.get('base_lr',0.001)
    BATCH_SIZE = train_args.get('BATCH_SIZE',385)
    image_size = train_args.get('image_size',38)
    # radio_cls_loss = train_args.get('radio_cls_loss',1.0)
    # radio_landmark_loss = train_args.get('radio_landmark_loss',0.1)
    LR_EPOCH = train_args.get('LR_EPOCH',[10,40,100,300,600,1000,2000,5000,10000])

    num = num_class
    print('Total datasets is: ', num)
    print(prefix)

    pos_radio = 1.0/5;landmark_radio=3.0/5;neg_radio=1.0/5
    pos_batch_size = int(np.ceil(BATCH_SIZE*pos_radio))
    assert pos_batch_size != 0,'Batch Size Error '        
    neg_batch_size = int(np.ceil(BATCH_SIZE*neg_radio))
    assert neg_batch_size != 0,'Batch Size Error '
    landmark_batch_size = int(np.ceil(BATCH_SIZE*landmark_radio))
    assert landmark_batch_size != 0,'Batch Size Error '
    batch_sizes = [pos_batch_size,neg_batch_size,landmark_batch_size]
    image_batch, label_batch,landmark_batch = read_multi_tfrecords(dataset_dirs,landmark_length,batch_sizes,image_size)

    input_image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='label')
    landmark_target = tf.placeholder(tf.float32,shape=[BATCH_SIZE,landmark_length],name='landmark_target')
    radio_landmark_loss = tf.placeholder(tf.float32, name='radio_landmark_loss')
    cls_loss_op,landmark_loss_op,L2_loss_op,accuracy_op,recall_op = Net(input_image, landmark_length, label,landmark_target,training=True)

    loss = (1-radio_landmark_loss)*cls_loss_op + radio_landmark_loss*landmark_loss_op + L2_loss_op
    train_op, lr_op = train_model(base_lr, loss, num, BATCH_SIZE, LR_EPOCH)

    init = tf.global_variables_initializer()
    sess = tf.Session()

    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)

    tf.summary.scalar('cls_loss',cls_loss_op)
    tf.summary.scalar('landmark_loss',landmark_loss_op)
    tf.summary.scalar('cls_accuracy',accuracy_op)
    tf.summary.scalar('cls_recall',recall_op)
    summary_op = tf.summary.merge_all()

    logs_dir = os.path.join(model_path,'logs')
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
                
            radio_loss = epoch/30.0 + 0.7
            radio_loss = min(radio_loss, 0.99)
            # radio_loss = 0.95
            
            image_batch_array, label_batch_array,landmark_batch_array = \
                            sess.run([image_batch, label_batch,landmark_batch])
            # print(landmark_batch_array[-1])
            
            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array,\
                                                landmark_target:landmark_batch_array,radio_landmark_loss: radio_loss})
            
            if (step+1) % display == 0:
                cls_loss,landmark_loss,L2_loss,lr,acc,recall,rl = \
                    sess.run([cls_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op,recall_op,radio_landmark_loss],
                    feed_dict={input_image: image_batch_array, label: label_batch_array, landmark_target: landmark_batch_array,\
                    radio_landmark_loss: radio_loss})
                    
                print('{}: Step: {}, accuracy: {:.3}, recall: {:.3} cls loss: {:.4}, landmark loss: {:.4}, L2 loss: {:.4}, lr:{:.2}, radio_landmark_loss:{:0.2} '.format(
                    datetime.now(), step+1, acc, recall, cls_loss, landmark_loss, L2_loss, lr, rl))

            if i * BATCH_SIZE > num*2:
                epoch = epoch + 2
                i = 0
                cls_loss,landmark_loss,L2_loss,lr,acc,recall,rl = \
                    sess.run([cls_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op,recall_op,radio_landmark_loss],
                    feed_dict={input_image: image_batch_array, label: label_batch_array, landmark_target: landmark_batch_array,\
                    radio_landmark_loss: radio_loss})
                print('Save model epoch: {}, Step: {}, accuracy: {:.3}, recall: {:.3}, cls loss: {:.4}, landmark loss: {:.4}, L2 loss: {:.4}, lr:{:.2}, radio_landmark_loss:{:.2} '.format( \
                    epoch, step+1, acc, recall, cls_loss, landmark_loss, L2_loss, lr,rl))
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
    
def train_net(net):
    print('\n####  ', net)
    root_dir = os.path.dirname(os.path.realpath(__file__)) 
    base_dir = os.path.join(root_dir, 'train_data', net)
    model_path = os.path.join(root_dir, 'models', net)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    print(base_dir, model_path)
    dataset_dirs = [os.path.join(base_dir, 'pos_landmark.tfrecord_shuffle'),
                    os.path.join(base_dir, 'neg_landmark.tfrecord_shuffle'),
                    os.path.join(base_dir, 'landmark_landmark.tfrecord_shuffle')]
                    
    print(dataset_dirs)
    label_file = os.path.join(base_dir,'train_{}.txt'.format(net))
    print(label_file) 
    f = open(label_file, 'r')
    lines = f.readlines()
    f.close()
    num_class = len(lines)

    landmark_length = None
    for line in lines:
        line = line.strip().split()
        if len(line) > 2:
            landmark_length = len(line) - 2
            break
    assert(landmark_length is not None)
    train_args ={}
    run(dataset_dirs, landmark_length, model_path, num_class, train_args)
    
if __name__ == '__main__':

    import sys
    net = sys.argv[1]
    assert(net in ['face', 'eye', 'nose', 'mouse'])
    train_net(net)
    # train_net('eye')
    # train_net('nose')
    # train_net('mouse')
    