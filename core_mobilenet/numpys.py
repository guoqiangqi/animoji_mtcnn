import numpy as np
from face_utils import convert_to_square, pad, py_nms,calibrate_box
import cv2

def read_file(file, shape):
    num = 1
    for s in shape:
        num *= s
    data = np.fromfile(file, dtype=np.float32, count=num)
    return data.reshape(shape)
    
def conv_forward_naive(x, w, b, conv_param):
  out = None
  C,H,W = x.shape
  F,I,HH,WW = w.shape
  assert(I == C ), '{}/{}'.format(I,C)
  S = conv_param['stride']
  P = conv_param['pad']
  Ho = 1 + (H + 2 * P - HH) // S
  Wo = 1 + (W + 2 * P - WW) // S
  x_pad = np.zeros((C,H+2*P,W+2*P))
  x_pad[:,P:P+H,P:P+W]=x
  out = np.zeros((F,Ho,Wo),dtype=np.float32)

  for f in range(F):
    for i in range(Ho):
      for j in range(Wo):
        out[f,i,j] = np.sum(x_pad[:, i*S : i*S+HH, j*S : j*S+WW] * w[f, :, :, :]) 

    out[f,:,:]+=b[f]
  return out
  
def cnn_maxpooling(out_feature, size=2, stride= 2, type_pooling='same'): 
    k,row,col=np.shape(out_feature)  
    if type_pooling == 'same':
            pad_height = ((row - 1) // stride)*stride + size - row;

            pad_weight = ((col - 1) // stride)*stride + size - col;
            pad_height = max(0, pad_height);
            pad_weight = max(0, pad_weight);
            pad0 = pad_height // 2;
            pad1 = pad_height - pad0;
            pad2 = pad_weight // 2;
            pad3 = pad_weight - pad2;
    else:
        assert(type_pooling == 'valid')
        pad_height = ((row - size + 1) // stride)*stride + size - row
        pad_weight = ((col - size + 1) // stride)*stride + size - col
        pad_height = max([0, pad_height])
        pad_weight = max([0, pad_weight])
        pad0 = pad_height // 2
        pad1 = pad_height - pad0
        pad2 = pad_weight // 2
        pad3 = pad_weight - pad2

        

    # pad_height = ((row - size + 1) // stride)*stride + size - row
    # pad_weight = ((col - size + 1) // stride)*stride + size - col
    # pad_height = max([0, pad_height])
    # pad_weight = max([0, pad_weight])
    # pad0 = pad_height // 2
    # pad1 = pad_height - pad0
    # pad2 = pad_weight // 2
    # pad3 = pad_weight - pad2
    
    out_row=int((row + pad_height - size) // stride) + 1  
    out_col=int((col + pad_weight - size) // stride) + 1  
    out_pooling=np.zeros((k,out_row,out_col),dtype=np.float32)
    
    
    for k_idx in range(0,k):  
        for r_idx in range(0,out_row):  
            for c_idx in range(0,out_col): 
                h_start = max([0,stride*r_idx-pad0])
                h_end = min([stride*r_idx+size-pad0, row])
                w_start = max([0, stride*c_idx-pad2])
                w_end = min([stride*c_idx+size-pad2, col])
                # print h_start, h_end, w_start, w_end
                temp_matrix=out_feature[k_idx, h_start:h_end, w_start:w_end]  
                out_pooling[k_idx,r_idx,c_idx]=np.amax(temp_matrix)    
    return out_pooling    
        
class onet_numpys(object):
    def __init__(self, file=None, param=None):
        if file is not None:
            params = []    
            w = read_file(file, (32,3,3,3))
            b = read_file(file, (32,))
            alphas = read_file(file, (32,1,1))
            params.append((w,b,alphas))
            
            w = read_file(file, (64,32,3,3))
            b = read_file(file, (64,))
            alphas = read_file(file, (64,1,1))
            params.append((w,b,alphas))
            
            w = read_file(file, (64,64,3,3))
            b = read_file(file, (64,))
            alphas = read_file(file, (64,1,1))
            params.append((w,b,alphas))
            
            w = read_file(file, (128,64,2,2))
            b = read_file(file, (128,))
            alphas = read_file(file, (128,1,1))
            params.append((w,b,alphas))
            
            w = read_file(file, (256,128*3*3))
            b = read_file(file, (256,))
            alphas = read_file(file, (256,))
            params.append((w,b,alphas))
            
            w = read_file(file, (2,256))
            b = read_file(file, (2,))
            params.append((w,b))
            
            w = read_file(file, (4,256))
            b = read_file(file, (4,))
            params.append((w,b))
            
            w = read_file(file, (10,256))
            b = read_file(file, (10,))
            params.append((w,b))
            
            w = read_file(file, (140,256))
            b = read_file(file, (140,))
            params.append((w,b))
        if param is not None:
            self.params = param
        else:
            self.params = params
     
    def forward(self, img, f):
        conv_param = {'stride':1, 'pad':0}
        x = img
        # x.astype(np.float32).tofile(f)
        w,b,alphas = self.params[0]
        x = conv_forward_naive(x, w, b, conv_param)
        # x.tofile(f)
        x = np.where(x<0,x*alphas,x)
        # x.tofile(f)
        x = cnn_maxpooling(x, size=3, stride=2, type_pooling='same')
        # x.tofile(f)
        
        w,b,alphas = self.params[1]
        x = conv_forward_naive(x, w, b, conv_param)
        # x.tofile(f)
        x = np.where(x<0,x*alphas,x)
        # x.tofile(f)
        x = cnn_maxpooling(x, size=3, stride=2, type_pooling='valid')
        # x.tofile(f)
        
        w,b,alphas = self.params[2]
        x = conv_forward_naive(x, w, b, conv_param)
        # x.tofile(f)
        x = np.where(x<0,x*alphas,x)
        # x.tofile(f)
        x = cnn_maxpooling(x, size=2, stride=2, type_pooling='same')
        # x.tofile(f)
        
        w,b,alphas = self.params[3]
        x = conv_forward_naive(x, w, b, conv_param)
        # x.tofile(f)
        x = np.where(x<0,x*alphas,x)
        # x.tofile(f)
        
        # x = x.transpose(1,2,0).reshape(-1)
        x = x.reshape(-1)
        w,b,alphas = self.params[4]
        x = np.dot(w,x) + b
        # x.tofile(f)
        x = np.where(x<0,x*alphas,x)
        # x.tofile(f)
        
        w,b = self.params[5]
        cls_prob = np.dot(w,x) +b
        # cls_prob.tofile(f)
        cls_prob = np.exp(cls_prob)/np.sum(np.exp(cls_prob),axis=0)
        # cls_prob = softmax(cls_prob)
        # cls_prob.tofile(f)
        
        w,b = self.params[6]
        bbox_pred = np.dot(w,x) +b
        # bbox_pred.tofile(f)
        
        w,b = self.params[7]
        landmark_pred  = np.dot(w,x) +b
        # landmark_pred.tofile(f)
        
        w,b = self.params[8]
        animoji_pred = np.dot(w,x) +b
        # animoji_pred.tofile(f)

        return cls_prob, bbox_pred, landmark_pred, animoji_pred
        
    def predict(self, im, dets_):
        dets = dets_.copy()
        net_size = 48
        threshold = 0.7
        expand = 0.3
        
        h, w, c = im.shape
        dets = convert_to_square(dets, expand)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, net_size, net_size, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (net_size, net_size))-127.5) / 128
        
        cls_scores = []
        reg = []
        landmark = []
        animoji = []
        
        for img_ in cropped_ims:
            cls_scores_i, reg_i,landmark_i, animoji_i = self.forward(img_.transpose(2,0,1),None)
            cls_scores.append(cls_scores_i.reshape(-1))
            reg.append(reg_i.reshape(-1))
            landmark.append(landmark_i.reshape(-1))
            animoji.append(animoji_i.reshape(-1))
            
        cls_scores = np.asarray(cls_scores)
        reg = np.asarray(reg)
        landmark = np.asarray(landmark)
        animoji = np.asarray(animoji)
        # print(reg.shape)
        # print(reg)
        # f.close()
        #prob belongs to face
        cls_scores = cls_scores[:,1]        
        keep_inds = np.where(cls_scores > threshold)[0]        
        if len(keep_inds) > 0:
            #pickout filtered box
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
            animoji = animoji[keep_inds]
        else:
            return None, None, None
        
        wh = boxes[:,2:4] - boxes[:,0:2] + 1
        xy = boxes[:,0:2]
        landmark = landmark.reshape(-1,5,2)*wh[:,None,:] + xy[:,None,:]  
        animoji = animoji.reshape(-1,70,2)*wh[:,None,:] + xy[:,None,:]         
        boxes_c = calibrate_box(boxes, reg)
        
        
        boxes = boxes[py_nms(boxes, 0.6, "Minimum")]
        keep = py_nms(boxes_c, 0.6, "Minimum")
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        
        return boxes, boxes_c, landmark, animoji,reg
        
class rnet_numpys(object):
    def __init__(self, file=None, param=None):
        if file is not None:
            params = []    
            w = read_file(file, (28,3,3,3))
            b = read_file(file, (28,))
            alphas = read_file(file, (28,1,1))
            params.append((w,b,alphas))
            
            w = read_file(file, (48,28,3,3))
            b = read_file(file, (48,))
            alphas = read_file(file, (48,1,1))
            params.append((w,b,alphas))
            
            w = read_file(file, (64,48,2,2))
            b = read_file(file, (64,))
            alphas = read_file(file, (64,1,1))
            params.append((w,b,alphas))
            
            w = read_file(file, (128,64*3*3))
            b = read_file(file, (128,))
            alphas = read_file(file, (128,))
            params.append((w,b,alphas))
            
            w = read_file(file, (2,128))
            b = read_file(file, (2,))
            params.append((w,b))
            
            w = read_file(file, (4,128))
            b = read_file(file, (4,))
            params.append((w,b))
            
            w = read_file(file, (10,128))
            b = read_file(file, (10,))
            params.append((w,b))
        if param is not None:
            self.params = param
        else:
            self.params = params
     
    def forward(self, img, f):
        conv_param = {'stride':1, 'pad':0}
        x = img
        # print(self.params[0])
        # conv1
        w,b,alphas = self.params[0]
        x = conv_forward_naive(x, w, b, conv_param)
        x = np.where(x<0,x*alphas,x)

        x = cnn_maxpooling(x, size=3, stride=2, type_pooling='same')
        
        # conv2
        w,b,alphas = self.params[1]
        x = conv_forward_naive(x, w, b, conv_param)
        x = np.where(x<0,x*alphas,x)

        x = cnn_maxpooling(x, size=3, stride=2, type_pooling='valid')

        # conv3
        w,b,alphas = self.params[2]
        x = conv_forward_naive(x, w, b, conv_param)
        x = np.where(x<0,x*alphas,x)
        
        # fc     
        x = x.reshape(-1)
        w,b,alphas = self.params[3]
        x = np.dot(w,x) + b
        x = np.where(x<0,x*alphas,x)

        # cls_prob
        w,b = self.params[4]
        cls_prob = np.dot(w,x) +b
        cls_prob = np.exp(cls_prob)/np.sum(np.exp(cls_prob),axis=0)
        
        # bbox_pred
        w,b = self.params[5]
        bbox_pred = np.dot(w,x) +b
        
        # landmark_pred
        w,b = self.params[6]
        landmark_pred  = np.dot(w,x) +b


        return cls_prob, bbox_pred, landmark_pred
        
        
    def predict(self, im, dets):
        threshold=0.6
        net_size = 24
    
        h, w, c = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, net_size, net_size, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (net_size, net_size))-127.5) / 128
            
            
        cls_scores = []
        reg = []
        
        for img_ in cropped_ims:
            cls_scores_i, reg_i,_ = self.forward(img_.transpose(2,0,1),None)
            cls_scores.append(cls_scores_i.reshape(-1))
            reg.append(reg_i.reshape(-1))
            
        cls_scores = np.asarray(cls_scores)
        reg = np.asarray(reg)
        cls_scores = cls_scores[:,1]
        keep_inds = np.where(cls_scores > threshold)[0]
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
        
class pnet_numpys(object):
    def __init__(self, file=None, param=None):
        if file is not None:
            params = []    
            w = read_file(file, (10,3,3,3))
            b = read_file(file, (10,))
            alphas = read_file(file, (10,1,1))
            params.append((w,b,alphas))

            w = read_file(file, (16,10,3,3))
            b = read_file(file, (16,))
            alphas = read_file(file, (16,1,1))
            params.append((w,b,alphas))
            
            w = read_file(file, (32,16,3,3))
            b = read_file(file, (32,))
            alphas = read_file(file, (32,1,1))
            params.append((w,b,alphas))
            
            w = read_file(file, (2,32,1,1))
            b = read_file(file, (2,))
            params.append((w,b))
            
            w = read_file(file, (4,32, 1, 1))
            b = read_file(file, (4,))
            params.append((w,b))
            
            w = read_file(file, (10,32, 1, 1))
            b = read_file(file, (10,))
            params.append((w,b))
            
        if param is not None:
            self.params = param
        else:
            self.params = params
     
    def forward(self, img, f):
        conv_param = {'stride':1, 'pad':0}
        x = img
        # conv1
        w,b,alphas = self.params[0]
        x = conv_forward_naive(x, w, b, conv_param)
        x = np.where(x<0,x*alphas,x)
        x = cnn_maxpooling(x, size=2, stride=2, type_pooling='same')
        
        #conv2
        w,b,alphas = self.params[1]
        x = conv_forward_naive(x, w, b, conv_param)
        x = np.where(x<0,x*alphas,x)
        
        #conv3
        w,b,alphas = self.params[2]
        x = conv_forward_naive(x, w, b, conv_param)
        x = np.where(x<0,x*alphas,x)
        
        #conv4
        w,b = self.params[3]
        cls_prob = conv_forward_naive(x, w, b, conv_param)
        cls_prob = np.exp(cls_prob)/np.sum(np.exp(cls_prob),axis=0)
        
        # bbox_pred
        w,b = self.params[4]
        bbox_pred = conv_forward_naive(x, w, b, conv_param)
        
        # landmark_pred
        w,b = self.params[5]
        landmark_pred = conv_forward_naive(x, w, b, conv_param)


        return cls_prob, bbox_pred, landmark_pred
        
    def processed_image(self, img, scale):
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
        img_resized = (img_resized - 127.5) / 128
        return img_resized
        
    def generate_bbox(self, cls_map, reg, scale):
        stride=2, 
        threshold=0.6, 
        scale_factor = 0.79
        net_size = 12
    
        # stride = self.stride
        cellsize = net_size

        t_index = np.where(cls_map > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        #offset
        dx1, dy1, dx2, dy2 = [reg[i, t_index[0], t_index[1]] for i in range(4)]

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
        min_face_size=25
        stride=2
        threshold=0.6
        scale_factor = 0.79

        # self.min_face_size = min_face_size
        # self.stride = stride
        # self.threshold = threshold
        # self.scale_factor = scale_factor
        # # h, w, c = im.shape
        net_size = 12
            
        current_scale = float(net_size) / min_face_size  # find initial scale
        # print("current_scale", net_size, self.min_face_size, current_scale)
    
        # current_scale = self.current_scale
        im_resized = self.processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape
        # print('numpys')
        # print(im_resized.shape)
        # print(im_resized)
        # isprint = True
        # fcn
        all_boxes = list()
        while min(current_height, current_width) > net_size:
            cls_cls_map, reg,_ = self.forward(im_resized.transpose(2,0,1),None)
            boxes = self.generate_bbox(cls_cls_map[1, :, :], reg, current_scale)

            current_scale *= scale_factor
            im_resized = self.processed_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape
            # if isprint:
                # print(cls_cls_map.shape)
                # print(cls_cls_map.transpose(1,2,0))
                # print(reg.shape)
                # print(reg.transpose(1,2,0))
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
        
if __name__ == '__main__':
    
    import os
    root_dir = os.path.dirname(os.path.realpath(__file__))
    imagepath = os.path.join(root_dir, '../test01.jpg')
    image = cv2.imread(imagepath)
    # h,w,_ = image.shape
    # image = cv2.resize(image, (w//2, h//2))
    f = open('face.bin','rb')
    pnet = pnet_numpys(f)
    rnet = rnet_numpys(f)
    onet = onet_numpys(f)
    f.close()
    
    _, boxes, _ = pnet.predict(image.copy())
    _, boxes, _ = rnet.predict(image.copy(),boxes)
    _, boxes, landmarks, animojis, _ = onet.predict(image.copy(), boxes)
   
    for bbox in boxes:
        cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
        
    for landmark in landmarks:
        for x,y in landmark:
            cv2.circle(image, (int(x+0.5),int(y+0.5)), 3, (0,0,255))
    
    for animoji in animojis:
        for x,y in animoji:
            cv2.circle(image, (int(x+0.5),int(y+0.5)), 1, (255,0,0))    
    # ImageName = imagepath.split('/')[-1]
    # imagedir = os.path.
    p,f=os.path.split(imagepath)
    cv2.imwrite(os.path.join(p,'result_{}'.format(f)),image) 