import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../core'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data'))
from data_utils import IoU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
from collections import OrderedDict
from detection import detect_face
import shutil
from utils import ratate,rectangel,Animoji
from model import Net
import tensorflow as tf
from random import shuffle
import dlib
    
class Detector(object):
    def __init__(self, data_size, out_size, model_path):
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[1, data_size, data_size, 3], name='input_image')            
            _,self.landmark_pred= Net(self.image_op, out_size, training=False)
            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print(model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            # print "restore models' param"
            saver.restore(self.sess, model_path)
            self.params = tf.trainable_variables()            

        self.data_size = data_size
        
    def predict(self, img):
        if len(img.shape) == 3:
            img = img[None,:,:,:]
        else:
            assert(len(img.shape) == 4), str(img.shape)
            assert(img.shape[0] == 1), str(img.shape)
        landmark_pred = self.sess.run(self.landmark_pred, feed_dict={self.image_op: img})
            
        return landmark_pred
        
    def get_params(self):
        params_dict = OrderedDict()
        for v in self.params:
            params_dict[v.name] = self.sess.run(v.name)
        # print params_dict.keys()
        return params_dict
   
class detect_face_net(object):
    def __init__(self, model_path):
        self.net_size = 38
        output_dim = 34
        self.net = Detector(self.net_size, output_dim, model_path)
        
    def predict(self, im, boxes, landmarks):
        
        img_rotated, _,(angle, center) = ratate(boxes, landmarks,img=im)
        
        image = img_rotated.copy()
        # bbox = boxes
        # cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
        # cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
        # cv2.imwrite('rotated.jpg', image)
        
        h, w, c = img_rotated.shape
        W, H = boxes[2:4] - boxes[0:2] + 1
        # print(angle)
        rate = 1.0
        # if angle > 10:
            # rate = 1.2
        # elif angle > 15:
            # rate = 2
                
        # face data    
        x1,y1,x2,y2,_ =  boxes
        dx1, dy1, dx2, dy2 = W*0.2*rate, (0.1*rate-0.2)*H, W*0.2*rate, H*0.2*rate
        x1 = max(0, int(x1 - dx1 + 0.5))
        y1 = max(0, int(y1 - dy1 + 0.5))
        x2 = min(w-1, int(x2 + dx2 + 0.5))
        y2 = min(h-1, int(y2 + dy2 + 0.5))
        # cv2.rectangle(image, (int(x1),int(y1)),(int(x2),int(y2)),(255,0,0))
        # cv2.imwrite('rotated.jpg', image)
        
        # face_box = (x1,y1,x2,y2)
        cropped_im = img_rotated[y1:y2, x1:x2, :]
        # cropped_im = cv2.resize(cropped_im, (38,38))

        cropped_ims = np.zeros((1, self.net_size, self.net_size, 3), dtype=np.float32)
        cropped_ims[0, :, :, :] = (cv2.resize(cropped_im, (self.net_size, self.net_size))-127.5) / 128
            
        landmark = self.net.predict(cropped_ims)

        wh = np.asarray([[x2-x1+1, y2-y1+1]],dtype=np.float32)
        xy =np.asarray([[x1,y1]],dtype=np.float32)
        landmark = landmark.reshape(-1,2)*wh + xy 

        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1)        
        landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark]) 

        return landmark
    
    def get_params(self):
        return self.net.get_params()
        
class detect_eye_net(object):
    def __init__(self, model_path):
        self.net_size = 38
        output_dim = 24
        self.net = Detector(self.net_size, output_dim, model_path)
        
    def predict(self, im, boxes, landmarks):
        
        img_rotated, (landmark_pred_rotated,),(angle, center) = ratate(boxes, landmarks, points = [landmarks], img=im)
        h, w, c = img_rotated.shape
        
        landmark_left_eye = landmark_pred_rotated[0]
        landmark_right_eye = landmark_pred_rotated[1]
        landmark_left_mouse = landmark_pred_rotated[3]
        W = landmark_right_eye[0] - landmark_left_eye[0] + 1
        H = landmark_left_mouse[1] - landmark_left_eye[1] + 1

        dx1, dy1, dx2, dy2 = W*0.8, H*0.7, W*0.8, H*0.4
        lx,ly = landmark_left_eye
        lx1 = max(0, int(lx - dx1 + 0.5))
        ly1 = max(0, int(ly - dy1 + 0.5))
        lx2 = min(w-1, int(lx + dx2 + 0.5))
        ly2 = min(h-1, int(ly + dy2 + 0.5))
        Llx = lx2 - lx1 + 1
        Lly = ly2 - ly1 + 1
        
        rx,ry = landmark_right_eye
        rx1 = max(0, int(rx - dx2 + 0.5))
        ry1 = max(0, int(ry - dy1 + 0.5))
        rx2 = min(w-1, int(rx + dx1 + 0.5))
        ry2 = min(h-1, int(ry + dy2 + 0.5))
        Lrx = rx2 - rx1 + 1
        Lry = ry2 - ry1 + 1
        
        left_im = img_rotated[ly1:ly2, lx1:lx2, :]
        right_im = img_rotated[ry1:ry2, rx1:rx2, :]
        right_im = cv2.flip(right_im, 1)
        # cv2.imwrite('left.jpg', left_im)
        # cv2.imwrite('right.jpg', right_im)

        cropped_im1 = np.zeros((1, self.net_size, self.net_size, 3), dtype=np.float32)
        cropped_im1[0, :, :, :] = (cv2.resize(left_im, (self.net_size, self.net_size))-127.5) / 128
        
        cropped_im2 = np.zeros((1, self.net_size, self.net_size, 3), dtype=np.float32)
        cropped_im2[0, :, :, :] = (cv2.resize(right_im, (self.net_size, self.net_size))-127.5) / 128
            
        landmark1 = self.net.predict(cropped_im1)
        landmark2 = self.net.predict(cropped_im2)
        
        landmark1 = landmark1.reshape(-1,2)
        landmark2 = landmark2.reshape(-1,2) 
        
        # print(landmark2.shape)
        idx = [4,3,2,1,0,8,7,6,5,10,9,11]
        landmark2 = landmark2[idx,:]
        landmark2[:,0] = 1 - landmark2[:,0]

        lwh = np.asarray([[Llx, Lly]],dtype=np.float32)
        lxy = np.asarray([[lx1,ly1]],dtype=np.float32)
        rwh = np.asarray([[Lrx, Lry]],dtype=np.float32)
        rxy = np.asarray([[rx1,ry1]],dtype=np.float32)
        landmark1 = landmark1*lwh + lxy
        landmark2 = landmark2*rwh + rxy        

        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1)        
        landmark1_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark1]) 
        landmark2_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark2])

        return landmark1_, landmark2_
    
    def get_params(self):
        return self.net.get_params()
        
class detect_mouse_net(object):
    def __init__(self, model_path):
        self.net_size = 38
        output_dim = 40
        self.net = Detector(self.net_size, output_dim, model_path)
        
    def predict(self, im, boxes, landmarks):
        
        img_rotated, (landmark_pred_rotated,),(angle, center) = ratate(boxes, landmarks, points = [landmarks], img=im)
        h, w, c = img_rotated.shape
        
        landmark_left_eye = landmark_pred_rotated[0]
        landmark_left_mouse = landmark_pred_rotated[3]
        landmark_right_mouse = landmark_pred_rotated[4]
        
        W = landmark_right_mouse[0] - landmark_left_mouse[0] + 1
        H = landmark_left_mouse[1] - landmark_left_eye[1] + 1

        dx1, dy1, dx2, dy2 = W*0.35, H*0.45, W*0.35, H*0.6
        # if x1 < 0.35 and y1 < 0.45 and x2 < 0.35 and y2 < 0.6:
        x1_, y1_ = landmark_left_mouse
        x2_, y2_ = landmark_right_mouse
        x1 = min(x1_, x2_)
        x2 = max(x1_, x2_)
        y1 = min(y1_, y2_)
        y2 = max(y1_, y2_)
        
        x1 = max(0, int(x1 - dx1 + 0.5))
        y1 = max(0, int(y1 - dy1 + 0.5))
        x2 = min(w-1, int(x2 + dx2 + 0.5))
        y2 = min(h-1, int(y2 + dy2 + 0.5))
        cropped_im = img_rotated[y1:y2, x1:x2, :]
        # cropped_im = cv2.resize(cropped_im, (38,38))

        cropped_ims = np.zeros((1, self.net_size, self.net_size, 3), dtype=np.float32)
        cropped_ims[0, :, :, :] = (cv2.resize(cropped_im, (self.net_size, self.net_size))-127.5) / 128
            
        landmark = self.net.predict(cropped_ims)

        wh = np.asarray([[x2-x1+1, y2-y1+1]],dtype=np.float32)
        xy =np.asarray([[x1,y1]],dtype=np.float32)
        landmark = landmark.reshape(-1,2)*wh + xy 

        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1)        
        landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark]) 

        # print('landmark shape:', landmark_.shape)
        return landmark_
    
    def get_params(self):
        return self.net.get_params()
        
class detect_nose_net(object):
    def __init__(self, model_path):
        self.net_size = 38
        output_dim = 18
        self.net = Detector(self.net_size, output_dim, model_path)
        
    def predict(self, im, boxes, landmarks):
        
        img_rotated, (landmark_pred_rotated,),(angle, center) = ratate(boxes, landmarks, points = [landmarks], img=im)
        h, w, c = img_rotated.shape
        
        landmark_left_eye = landmark_pred_rotated[0]
        landmark_right_eye = landmark_pred_rotated[1]
        landmark_left_mouse = landmark_pred_rotated[3]
        W = landmark_right_eye[0] - landmark_left_eye[0] + 1
        H = landmark_left_mouse[1] - landmark_left_eye[1] + 1


        dx1, dy1, dx2, dy2 = W*0.6, H*0.9, W*0.6, H*0.4
        # if x1 < 0.6 and y1 < 0.9 and x2 < 0.6 and y2 < 0.4:
        nx,ny = landmark_pred_rotated[2]
        
        x1 = max(0, int(nx - dx1 + 0.5))
        y1 = max(0, int(ny - dy1 + 0.5))
        x2 = min(w-1, int(nx + dx2 + 0.5))
        y2 = min(h-1, int(ny + dy2 + 0.5))
        cropped_im = img_rotated[y1:y2, x1:x2, :]
        # cropped_im = cv2.resize(cropped_im, (38,38))

        cropped_ims = np.zeros((1, self.net_size, self.net_size, 3), dtype=np.float32)
        cropped_ims[0, :, :, :] = (cv2.resize(cropped_im, (self.net_size, self.net_size))-127.5) / 128
            
        landmark = self.net.predict(cropped_ims)

        wh = np.asarray([[x2-x1+1, y2-y1+1]],dtype=np.float32)
        xy =np.asarray([[x1,y1]],dtype=np.float32)
        landmark = landmark.reshape(-1,2)*wh + xy 

        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1)        
        landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark]) 

        # print('landmark shape:', landmark_.shape)
        return landmark_
    
    def get_params(self):
        return self.net.get_params()
        
class detect_all_net(object):
    def __init__(self, model_paths):
        assert(len(model_paths) == 4)
        face_model_path, eye_model_path, mouse_model_path, nose_model_path = model_paths
        
        self.face_net = detect_face_net(face_model_path)
        self.eyes_net = detect_eye_net(eye_model_path)
        self.mouse_net = detect_mouse_net(mouse_model_path)
        self.nose_net = detect_nose_net(nose_model_path)
        
    def predict(self, im, boxes, landmarks):
        Olandmark = []
        for idx in range(len(boxes)):
            animoji_class = Animoji()
            box = boxes[idx]
            landmark = landmarks[idx]
            animoji = animojis[idx]
            animoji_class.face = self.face_net.predict(image.copy(),box,landmark)
            animoji_class.left_eye, animoji_class.right_eye = self.eyes_net.predict(image.copy(),box,landmark)
            animoji_class.mouse = self.mouse_net.predict(image.copy(),box,landmark)
            animoji_class.nose = self.nose_net.predict(image.copy(),box,landmark)
            Olandmark.append(animoji_class.get_animoji())
        return np.asarray(Olandmark,dtype=np.float32)
    
    def get_params(self):
        params = OrderedDict()
               
        for name, p in self.face_net.get_params().items():
            params['face_'+name] = p
                   
        for name, p in self.eyes_net.get_params().items():   
            params['eye_'+name] = p

        for name, p in self.mouse_net.get_params().items():
            params['mouse_'+name] = p

        for name, p in self.nose_net.get_params().items():
            params['nose_'+name] = p
            
        return params
        
if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = [os.path.join(root_dir, '../models/PNet/PNet-22'), 
                  os.path.join(root_dir, '../models/RNet/RNet-18'),
                  os.path.join(root_dir, '../models/LONet/LONet_expand-580'),
                  # os.path.join(root_dir, '../models/LONet_0805/LONet-340')
                  ]             
                  
    face_model_path = [os.path.join(root_dir, 'models/face/-1716'),
                       os.path.join(root_dir, 'models/eye/-682'),
                       os.path.join(root_dir, 'models/mouse/-714'),
                       os.path.join(root_dir, 'models/nose/-1058')]
    
    facenet = detect_face(model_path, True)
    landmarknet = detect_all_net(face_model_path)
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 
    
    face_params = facenet.get_params()
    landmark_params = landmarknet.get_params()
    remove_names = ['pnet_conv4_3', 'rnet_landmark_fc', 'face_cls_fc', 'eye_cls_fc', 'mouse_cls_fc', 'nose_cls_fc']
    f1 = open('face.bin', 'wb')
    f3 = open('face3.bin', 'wb')
    fonet = open('onet.bin','wb')
    for name, param in face_params.items():
        param = param.astype(np.float32)
        if len(param.shape) == 2:
            param = param.transpose(1,0)
        elif len(param.shape) == 4:
            param = param.transpose(3,2,0,1)
            
        if 'onet' in name:
            param.tofile(fonet)
    
        Isremove = False
        for re in remove_names:
            if re in name:
                Isremove = True
                break
        if not Isremove:
            print(name)
            
            param.tofile(f1)
            param.tofile(f3)
    f1.close()
    fonet.close()
    f2 = open('face2.bin','wb')
    for name, param in landmark_params.items():
        param = param.astype(np.float32)
        if len(param.shape) == 2:
            param = param.transpose(1,0)
        elif len(param.shape) == 4:
            param = param.transpose(3,2,0,1)
            
        Isremove = False
        for re in remove_names:
            if re in name:
                Isremove = True
                break
        if not Isremove:
            print(name)                
            param.tofile(f2)
            param.tofile(f3)
    f2.close()
    f3.close()
    # exit()
    
    imagepath = os.path.join(root_dir, '../test02.jpg')
    image = cv2.imread(imagepath)
    h,w,_ = image.shape
    if min(h,w) > 1000:
        image = cv2.resize(image, (w//2, h//2))
    # pboxes,rboxes,boxes, landmarks, animojis,reg
    _, rboxes, boxes, landmarks, animojis, _ = facenet.predict(image.copy())
    animojis_landmarks = landmarknet.predict(image.copy(), boxes, landmarks)
    # print(animojis_landmarks)

    for idx in range(boxes.shape[0]):
        box = boxes[idx]
        landmark = landmarks[idx]
        animoji = animojis[idx]
        animoji_landmark = animojis_landmarks[idx]
        cv2.putText(image,str(np.round(box[4],2)),(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
        cv2.rectangle(image, (int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255))

        for x,y in animoji_landmark:
            cv2.circle(image, (int(x+0.5),int(y+0.5)), 1, (0,255,0))
        for x,y in landmark:
            cv2.circle(image, (int(x+0.5),int(y+0.5)), 1, (0,0,255))
        for x,y in animoji:
            cv2.circle(image, (int(x+0.5),int(y+0.5)), 1, (255,0,0))    
    
    p,f=os.path.split(imagepath)
    cv2.imwrite(os.path.join(p,'result_{}'.format(f)),image) 
    # exit()
    
    print('test image.')
    test_path = '/home/zhifeng/data/chinese'
    test_image_paths = []
    exts = []
    for name in os.listdir(test_path):
        pt = os.path.join(test_path, name)
        images = os.listdir(pt)
        if len(images) == 1:
            pt = os.path.join(pt,images[0])
            images = os.listdir(pt)
        # print(len(images))
        for image in images:
            _, ext = os.path.splitext(image)
            if ext not in exts:
                exts.append(ext)
            if ext in ['.jpg', '.JPG']:
                test_image_paths.append(os.path.join(pt, image))
    print(exts)
    shuffle(test_image_paths)
    image_idx = 0
    test_miss = 0
    for i, filepath in enumerate(test_image_paths):
        
        # print(filepath)
        image = cv2.imread(filepath)
        h,w,_ = image.shape
        line_w = int(max(h,w)/500+1)
        _, _, boxes, landmarks, animojis,_ = facenet.predict(image.copy())
        if boxes is None or landmark is None or animojis is None:
            test_miss += 1
            continue
        if image_idx >= 100:
            continue
        animojis_landmarks = landmarknet.predict(image.copy(), boxes, landmarks)        

        for box, animoji, animoji_landmark in zip(boxes, animojis, animojis_landmarks):
            Animoji_landmark = Animoji(animoji_landmark)
            Animoji_pred = Animoji(animoji)
            Animoji_landmark.face = Animoji_pred.face
            animoji_landmark = Animoji_landmark.get_animoji()
            
            cv2.putText(image,str(np.round(box[4],2)),(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
            cv2.rectangle(image, (int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255))
            for x,y in animoji_landmark:
                cv2.circle(image, (int(x+0.5),int(y+0.5)), line_w, (0,0,255))

            x1,y1,x2,y2,_ = box
            face = dlib.rectangle(int(x1+0.5), int(y1+0.5), int(x2+0.5), int(y2+0.5))
            shape = predictor(image, face)
            for pt in shape.parts():
                pt_pos = (pt.x, pt.y)
                cv2.circle(image, pt_pos, line_w+1, (0, 255, 0), 1)
                
        cv2.imwrite('test_landmark_images/test_landmark_{}.jpg'.format(image_idx), image)
        image_idx += 1
    print('Detection failure rate: {}  ({}/{})'.format((test_miss+0.0)/len(test_image_paths), test_miss, len(test_image_paths)))
    
    
    print('precision analysis.')
    f = open('images/detect_result.txt','r')
    out_result = eval(f.read())
    f.close()

    num_of_images = len(out_result)
    names = ['face', 'left_eye', 'right_eye', 'nose', 'mouse', 'animoji']
    dists1,dists2, dists3 = {}, {}, {}
    dlib_dists = []
    dists_mean = []
    for name in names:
        dists1[name] = []
        dists2[name] = []
        dists3[name] = []
        
    filepaths = list(out_result.keys())
    shuffle(filepaths)
    image_idx = 0
    for i, filepath  in enumerate(filepaths):
        filename, npz = out_result[filepath]
        if (i+1) % 100 == 0:
            print('{}/{} images done'.format(i+1, num_of_images))
            for name in names:
                dists1_ = np.asarray(dists1[name])
                dists2_ = np.asarray(dists2[name])
                dists3_ = np.asarray(dists3[name])
                print('{:<16}{:<8.4f}{:<8.4f}{:<8.4f} --- {:<8.4f}{:<8.4f}{:<8.4f}'.format(\
                        name+':', np.mean(dists1_), np.mean(dists2_), np.mean(dists3_), \
                        np.std(dists1_), np.std(dists2_), np.std(dists3_)))
                        
            dlib_dists_ = np.asarray(dlib_dists)
            print('{:<16}{:<8.4f} --- {:<8.4f}'.format('dlib:', np.mean(dlib_dists_), np.std(dlib_dists_)))
            print('')
            # break
        # if i > 10: break
        Array = np.load(npz)
        bbox = Array['bbox']
        wh = bbox[2:4] - bbox[0:2] + 1
        wh = wh[None,:]
        animoji = Array['landmark']
        animoji = animoji.reshape(70,2)
        boxes_pred = Array['boxes_pred']
        landmarks_pred = Array['landmarks_pred']
        animojis_pred = Array['animoji_pred']
        if len(boxes_pred.shape) == 0:
            continue
        
        iou = IoU(bbox, boxes_pred)
        idx = np.argmax(iou) 
        
        if iou[idx] < 0.4:
            continue

        image = cv2.imread(filepath)
        image2 = image.copy()
        h,w,_ = image2.shape
        line_w = int(max(h,w)/500+1)
        
        boxes = boxes_pred[idx]
        landmarks = landmarks_pred[idx]
        
        animoji_pred = animojis_pred[idx]
        animojis_landmark = landmarknet.predict(image.copy(), [boxes], [landmarks])[0]
        Animojis_landmark = Animoji(animojis_landmark)
        Animoji_pred = Animoji(animoji_pred)
        Animojis_landmark.face = Animoji_pred.face
        
        x1,y1,x2,y2,_ = boxes
        face = dlib.rectangle(int(x1+0.5), int(y1+0.5), int(x2+0.5), int(y2+0.5))
        shape = predictor(image.copy(), face)
        pt_poses = []
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            pt_poses.append(pt_pos)
            cv2.circle(image2, pt_pos, line_w+1, (0, 255, 0), 1)
        pt_poses = np.asarray(pt_poses, dtype=np.float32)
        
        for x,y in landmarks:
            cv2.circle(image, (int(x),int(y)), 1, (255,0,255))
        for x,y in animoji:
            cv2.circle(image, (int(x),int(y)), 1, (255,0,0))
        for x,y in animojis_landmark:
            cv2.circle(image, (int(x),int(y)), 1, (0,255,0))
        for x,y in animoji_pred:
            cv2.circle(image, (int(x),int(y)), 1, (0,0,255))

        
        if image_idx < 100 and np.min(wh) > 90:
            cv2.imwrite('test_landmark_images/tmp_landmark_{}.jpg'.format(image_idx), image)
            for x,y in Animojis_landmark.get_animoji():
                cv2.circle(image2, (int(x+0.5),int(y+0.5)), int(line_w*1.2), (0,0,255))
            for x,y in animoji:
                cv2.circle(image2, (int(x+0.5),int(y+0.5)), line_w, (255,0,0))
            cv2.imwrite('test_landmark_images/out_landmark_{}.jpg'.format(image_idx), image2)
            image_idx += 1
        
        
        dlib_dist = np.sum(np.square(animoji[:-2,:]/wh - pt_poses/wh))
        dlib_dists.append(dlib_dist)
        
        animoji = Animoji(animoji)
        animoji_pred = Animoji(animoji_pred)
        animojis_landmark = Animoji(animojis_landmark)
                
        for name in names:
            animoji_ = getattr(animoji, name)
            animoji_pred_ = getattr(animoji_pred, name)
            animojis_landmark_ = getattr(animojis_landmark, name)
            mean_ = (animoji_pred_ + animojis_landmark_)*0.5
        
            dist1 = np.sum(np.square(animoji_/wh - animoji_pred_/wh))
            dist2 = np.sum(np.square(animoji_/wh - animojis_landmark_/wh))
            dist3 = np.sum(np.square(animoji_/wh - mean_/wh))
            dists1[name].append(dist1)
            dists2[name].append(dist2)
            dists3[name].append(dist3)
    
    for name in names:
        dists1_ = np.asarray(dists1[name])
        dists2_ = np.asarray(dists2[name])
        dists3_ = np.asarray(dists3[name])
        print('{:<16}{:<8.4f}{:<8.4f}{:<8.4f} --- {:<8.4f}{:<8.4f}{:<8.4f}'.format(\
                        name+':', np.mean(dists1_), np.mean(dists2_), np.mean(dists3_), \
                        np.std(dists1_), np.std(dists2_), np.std(dists3_)))
    dlib_dists_ = np.asarray(dlib_dists)
    print('{:<16}{:<8.4f} --- {:<8.4f}'.format('dlib:', np.mean(dlib_dists_), np.std(dlib_dists_)))
    print('-----------------   end   ------------------')
        # break