import os
import cv2
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from mtcnn_model import  P_Net, R_Net, O_Net, L_O_Net
from face_utils import py_nms, calibrate_box, convert_to_square, pad
from numpys import pnet_numpys, rnet_numpys, onet_numpys
import glob

    
class FcnDetector(object):
    def __init__(self, net_factory, model_path):
        graph = tf.Graph()
        print(model_path)
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            self.cls_prob, self.bbox_pred, _ = net_factory(image_reshape, training=False)
            
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                                    gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            # print model_path
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            # print "restore models' param"
            saver.restore(self.sess, model_path)
            self.params = tf.trainable_variables()
    def predict(self, databatch):
        height, width, _ = databatch.shape
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                       feed_dict={self.image_op: databatch, self.width_op: width,
                                                  self.height_op: height})
        return cls_prob, bbox_pred
        
    def get_params(self):  
        params_dict = OrderedDict()
        for v in self.params:
            params_dict[v.name] = self.sess.run(v.name)
        # print params_dict.keys()
        return params_dict

class Detector(object):
    def __init__(self, net_factory, data_size, batch_size, model_path):
        graph = tf.Graph()
        print(model_path)
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, 3], name='input_image')            
            self.cls_prob, self.bbox_pred, self.landmark_pred, self.animoji= net_factory(self.image_op, training=False)
            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            # print "restore models' param"
            saver.restore(self.sess, model_path)
            self.params = tf.trainable_variables()            

        self.data_size = data_size
        self.batch_size = batch_size
        
    def predict(self, databatch):
        scores = []
        batch_size = self.batch_size

        minibatch = []
        cur = 0
        n = databatch.shape[0]
        while cur < n:
            minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size
        cls_prob_list = []
        bbox_pred_list = []
        landmark_pred_list = []
        animoji_pred_list = []
        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size 
            if m < batch_size:
                keep_inds = np.arange(m)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m

          
            if self.animoji is None:
                cls_prob, bbox_pred,landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred,self.landmark_pred], feed_dict={self.image_op: data})
            else:
                cls_prob, bbox_pred,landmark_pred, animoji = self.sess.run([self.cls_prob, self.bbox_pred,self.landmark_pred, self.animoji], feed_dict={self.image_op: data})
                animoji_pred_list.append(animoji[:real_size])
            cls_prob_list.append(cls_prob[:real_size])
            bbox_pred_list.append(bbox_pred[:real_size])
            landmark_pred_list.append(landmark_pred[:real_size])
            
        if self.animoji is None:
            animoji_pred_list = None
        else:
            animoji_pred_list = np.concatenate(animoji_pred_list, axis=0)
        return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0),animoji_pred_list

    def get_params(self):
        params_dict = OrderedDict()
        for v in self.params:
            params_dict[v.name] = self.sess.run(v.name)
        # print params_dict.keys()
        return params_dict
        
class detect_pnet(object):
    def __init__(self, model_path, min_face_size=25, stride=2, threshold=0.6, scale_factor = 0.79):
        # print model_path
        self.PNet = FcnDetector(P_Net, model_path)
        self.min_face_size = min_face_size
        self.stride = stride
        self.threshold = threshold
        self.scale_factor = scale_factor
        # h, w, c = im.shape
        self.net_size = 12
            
        self.current_scale = float(self.net_size) / self.min_face_size  # find initial scale
        # print("current_scale", net_size, self.min_face_size, current_scale)
        
    def processed_image(self, img, scale):
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
        img_resized = (img_resized - 127.5) / 128
        return img_resized
        
    def generate_bbox(self, cls_map, reg, scale):
        stride = self.stride
        cellsize = self.net_size

        t_index = np.where(cls_map > self.threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        #offset
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])

        return boundingbox.T
        
    def predict(self, im):
        current_scale = self.current_scale
        im_resized = self.processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape
        # fcn
        all_boxes = list()
        # print('pnet')
        # isprint = True
        
        while min(current_height, current_width) > self.net_size:    
            cls_cls_map, reg = self.PNet.predict(im_resized)
            boxes = self.generate_bbox(cls_cls_map[:, :,1], reg, current_scale)

            current_scale *= self.scale_factor
            im_resized = self.processed_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape
            # if isprint:
                # print(cls_cls_map.shape)
                # print(cls_cls_map)
                # print(reg.shape)
                # print(reg)
                # print(im_resized.shape)
                # print(im_resized)
                # isprint = False

            if boxes.size == 0:
                continue
            keep = py_nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None, None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]

        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # refine the boxes
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return boxes, boxes_c, None
    
    def get_params(self):
        return self.PNet.get_params()
        
        
class detect_rnet(object):
    def __init__(self, model_path, threshold=0.6, batch_size=256):
        self.net_size = 24
        self.batch_size = batch_size
        self.threshold = threshold
        self.RNet = Detector(R_Net, self.net_size, batch_size, model_path)
        
    def predict(self, im, dets):
        h, w, c = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, self.net_size, self.net_size, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (self.net_size, self.net_size))-127.5) / 128
        cls_scores, reg, _, _ = self.RNet.predict(cropped_ims)
        cls_scores = cls_scores[:,1]
        keep_inds = np.where(cls_scores > self.threshold)[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None, None
        
        
        keep = py_nms(boxes, 0.6)
        boxes = boxes[keep]
        boxes_c = calibrate_box(boxes, reg[keep])
        return boxes, boxes_c, None
    def get_params(self):
        return self.RNet.get_params()

class detect_onet(object):
    def __init__(self, model_path, threshold=0.7, batch_size=16):
        self.net_size = 96
        self.batch_size = batch_size
        self.threshold = threshold
        self.ONet = Detector(O_Net, self.net_size, batch_size, model_path)
        
    def predict(self, im, dets):
        
        h, w, c = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, self.net_size, self.net_size, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (self.net_size, self.net_size))-127.5) / 128
            
        cls_scores, reg,landmark,_ = self.ONet.predict(cropped_ims)
        #prob belongs to face
        cls_scores = cls_scores[:,1]        
        keep_inds = np.where(cls_scores > self.threshold)[0]        
        if len(keep_inds) > 0:
            #pickout filtered box
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None, None, None
        
        wh = boxes[:,2:4] - boxes[:,0:2] + 1
        xy = boxes[:,0:2]
        landmark = landmark.reshape(-1,5,2)*wh[:,None,:] + xy[:,None,:]       
        boxes_c = calibrate_box(boxes, reg)
        
        
        boxes = boxes[py_nms(boxes, 0.6, "Minimum")]
        keep = py_nms(boxes_c, 0.6, "Minimum")
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes, boxes_c, landmark, None
    
    def get_params(self):
        return self.ONet.get_params()

class detect_L_onet(object):
    def __init__(self, model_path, threshold=0.7, batch_size=16):
        self.net_size = 96
        self.batch_size = batch_size
        self.threshold = threshold
        self.expand = 0.3
        self.L_ONet = Detector(L_O_Net, self.net_size, batch_size, model_path)
        print('L onet expand: ', self.expand)
        
    def predict(self, im, dets):
        
        h, w, c = im.shape
        dets = convert_to_square(dets, self.expand)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, self.net_size, self.net_size, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (self.net_size, self.net_size))-127.5) / 128
            
        # print(cropped_ims.shape)
        # cropped_ims.tofile('test_img.bin')
        cls_scores, reg, landmark, animoji = self.L_ONet.predict(cropped_ims)
        # print(reg.shape)
        # print(reg)
        #prob belongs to face
        cls_scores = cls_scores[:,1]        
        keep_inds = np.where(cls_scores > self.threshold)[0]        
        if len(keep_inds) > 0:
            #pickout filtered box
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
            animoji = animoji[keep_inds]
        else:
            return None, None, None, None, None
        
        wh = boxes[:,2:4] - boxes[:,0:2] + 1
        xy = boxes[:,0:2]
        landmark = landmark.reshape(-1,5,2)*wh[:,None,:] + xy[:,None,:]
        animoji = animoji.reshape(-1,70,2)*wh[:,None,:] + xy[:,None,:]       
        boxes_c = calibrate_box(boxes, reg)
        
        
        boxes = boxes[py_nms(boxes, 0.6, "Minimum")]
        keep = py_nms(boxes_c, 0.6, "Minimum")
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        animoji = animoji[keep]
        return boxes, boxes_c, landmark, animoji, reg
    
    def get_params(self):
        return self.L_ONet.get_params()
        
class  detect_face(object):
    def __init__(self, model_paths, Used_L_ONet=False):
        assert(len(model_paths) == 3)
        self.pnet = detect_pnet(model_paths[0])
        self.rnet = detect_rnet(model_paths[1])
        if Used_L_ONet:
            self.onet = detect_L_onet(model_paths[2])
        else:
            self.onet = detect_onet(model_paths[2])
    
    def predict(self,image):        
        _, pboxes_c, _ = self.pnet.predict(image)
        # if pboxes_c is not None:
            # print("pboxes_c")
            # print(pboxes_c.shape)
            # print(pboxes_c)
        # exit()

        if pboxes_c is None:
            return None, None, None, None, None, None
        _, rboxes_c, landmarks = self.rnet.predict(image, pboxes_c)
        # print('rboxes_c')
        # print(rboxes_c.shape)
        # print(rboxes_c)
        if rboxes_c is None:
            return None, None, None, None, None, None
        _, oboxes_c, landmarks, animoji,reg = self.onet.predict(image, rboxes_c)
        # if oboxes_c is not None:
            # print("oboxes_c")
            # print(oboxes_c.shape)
            # print(oboxes_c)
        return pboxes_c, rboxes_c, oboxes_c, landmarks, animoji, reg
    
    def get_params(self):
        params_dict = OrderedDict()
        pnet_param = self.pnet.get_params()
        rnet_param = self.rnet.get_params()
        onet_param = self.onet.get_params()
        
        for name, param in pnet_param.items():
            params_dict['pnet_'+name] = param
        for name, param in rnet_param.items():
            params_dict['rnet_'+name] = param
        for name, param in onet_param.items():
            params_dict['onet_'+name] = param
        return params_dict
            
    def save_model_bin(self, model_path, model_info=None):
        if model_info is None:
            model_info = os.path.splitext(model_path)[0] + '.txt'
        pnet_param = self.pnet.get_params()
        rnet_param = self.rnet.get_params()
        onet_param = self.onet.get_params()
        
        fbin = open(model_path, 'wb')
        ftxt = open(model_info, 'w')
        params = [('pnet param', pnet_param), 
                  ('rnet param', rnet_param),
                  ('onet param', onet_param)]
        for name, param in params:
            ftxt.write('### {} \n'.format(name))
            for key in param.keys():
                data = param[key].astype(np.float32)
                if len(data.shape) == 2:
                    data = data.transpose(1,0)
                elif len(data.shape) == 4:
                    data = data.transpose(3,2,0,1)
                shape = data.shape
                ftxt.write('{:<50} {}\n'.format(key, ' '.join(map(str, shape))))
                data.tofile(fbin)
            ftxt.write('\n')
        fbin.close()
        ftxt.close()
            
if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = [os.path.join(root_dir, '../models/PNet_1/PNet-60'), 
                  os.path.join(root_dir, '../models/RNet_1/RNet-42'),
                  os.path.join(root_dir, '../models/LONet_mobilenet1/LONet-126399'),
                  # os.path.join(root_dir, '../models/LONet_0805/LONet-340'),
                  ]
    #imagepath = os.path.join(root_dir, '../image_test/001.jpg')
    #image = cv2.imread(imagepath)
    # h,w,_ = image.shape
    # image = cv2.resize(image, (w//2, h//2))
    
    facenet = detect_face(model_path, True)
    # pronet_param = [facenet.pnet.get_params(), facenet.rnet.get_params(), facenet.onet.get_params()]
    
    # pronet_params = []
    # for net_param in pronet_param:
        # numpy_params = OrderedDict()
        # for name, param in net_param.items():
            # if len(param.shape) == 2:
                # param = param.transpose(1,0)
            # elif len(param.shape) == 4:
                # param = param.transpose(3,2,0,1)
                
            # layer, n = name.split('/')
            # if layer not in numpy_params.keys():
                # numpy_params[layer] = [None, None, None]
            # if 'weights:0' == n:
                # numpy_params[layer][0] = param
            # elif 'biases:0' == n:
                # numpy_params[layer][1] = param
            # elif 'alphas:0' == n:
                # numpy_params[layer][2] = param
                
        # net_numpy = []
        # for l, p in numpy_params.items():
            # w,b,alphas = p
            # # print(l)
            # # print(w.shape)
            # # print(b.shape)
            # if alphas is not None:
                # # print(alphas.shape)
                # if len(w.shape) == 4:
                    # alphas = alphas[:,None,None]
                # net_numpy.append((w,b,alphas))
            # else:
                # net_numpy.append((w,b))
        
        # pronet_params.append(net_numpy)
    # exit()
    
    #save_model
    #facenet.save_model_bin(os.path.join(root_dir, '../numpys/face_deepermodel.bin'))    
    
    #pboxes,rboxes,boxes, landmarks, animojis,reg = facenet.predict(image.copy())
    
    # onet_params = pronet_params[2]
    # f = open('Onet.bin','wb')
    # for ps in onet_params:
        # for p in ps:
            # p.astype(np.float32).tofile(f)
    # f.close()

    # pnet = pnet_numpys(param=pronet_params[0])
    # _, npboxes, _ = pnet.predict(image.copy())
    # rnet = rnet_numpys(param=pronet_params[1])
    # _, npboxes, _ = rnet.predict(image.copy(),npboxes)

    # onet = onet_numpys(param=pronet_params[2])
    # _, npboxes, nplandmark, npanimoji,npreg = onet.predict(image.copy(), npboxes)
    
    
    # print(boxes.shape[0])
    # print(npboxes.shape[0])
    # print(boxes)
    # print(npboxes)
    # print(np.abs(npreg - reg))
    # print(np.abs(npreg - reg) < 5e-5)
    # print((np.abs(npreg - reg) < 5e-5).all())
    # print(boxes[:,2:4] - boxes[:,0:2] + 1)
    # print(boxes) 
    # print(landmarks)
    image_path = os.path.join(root_dir, '../image_part_train/')
    for f in glob.glob(os.path.join(image_path, '*.jpg')):
        image=cv2.imread(f)
        pboxes,rboxes,boxes, landmarks, animojis,reg = facenet.predict(image.copy())
        if boxes is not None:
            for bbox in boxes:
                cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
                cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
        if landmarks is not None:    
            for landmark in landmarks:
                for x,y in landmark:
                    cv2.circle(image, (int(x+0.5),int(y+0.5)), 3, (0,0,255))
        
        if animojis is not None:
            for animoji in animojis:
                for x,y in animoji:
                    cv2.circle(image, (int(x+0.5),int(y+0.5)), 1, (255,0,0))    
        # ImageName = imagepath.split('/')[-1]
        # imagedir = os.path.
        p,g=os.path.split(f)
        cv2.imshow(g,image)
    k =cv2.waitKey(0)
    if k==ord('q'):
        cv2.destroyAllWindows()
