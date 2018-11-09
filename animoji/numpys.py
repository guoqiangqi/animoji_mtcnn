import numpy as np

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
    
    

        # net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope='conv1')
        # net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool1', padding='SAME')
        # net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope='conv2')
        # net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool2')
        # net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope='conv3')
        # net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool3', padding='SAME')
        # net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope='conv4')
        # net = tf.transpose(net, perm=[0,3,1,2])         
        # fc_flatten = slim.flatten(net)
        # fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope='fc1', activation_fn=prelu)

        # cls_prob = slim.fully_connected(fc1,num_outputs=2,scope='cls_fc',activation_fn=tf.nn.softmax)
        # bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope='bbox_fc',activation_fn=None)
        # landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope='landmark_fc',activation_fn=None)
       
        
class onet_predict(object):
    def __init__(self, file=None, param = None):
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
        cls_prob = softmax(cls_prob)
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
     
    def predict(self, im, dets):
        # net_size = 48
        threshold = 0.7
        h, w, c = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        
        # ONet = onet_predict(file)
        cls_scores = []
        reg = []
        landmark = []
        animoji = []
        # f = open('test_onet_predict.bin','wb')
        f = None
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims = (cv2.resize(tmp, (net_size, net_size))-127.5) / 128
            
            cls_scores_i, reg_i,landmark_i, animoji_i = self.forward(cropped_ims.transpose(2,0,1),f)
            cls_scores.append(cls_scores_i.reshape(-1))
            reg.append(reg_i.reshape(-1))
            landmark.append(landmark_i.reshape(-1))
            animoji.append(animoji_i)
            
        # f.close()
        cls_scores = np.asarray(cls_scores)
        reg = np.asarray(reg)
        landmark = np.asarray(landmark)
        animoji = np.asarray(animoji)
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
        boxes = calibrate_box(boxes, reg)
        keep = py_nms(boxes, 0.6, "Minimum")
        boxes = boxes[keep]
        landmark = landmark[keep]
        #width
        w = boxes[:,2] - boxes[:,0] + 1
        #height
        h = boxes[:,3] - boxes[:,1] + 1
        landmark[:,0::2] = (np.tile(w,(5,1)) * landmark[:,0::2].T + np.tile(boxes[:,0],(5,1)) - 1).T
        landmark[:,1::2] = (np.tile(h,(5,1)) * landmark[:,1::2].T + np.tile(boxes[:,1],(5,1)) - 1).T     

        animoji[:,0::2] = (np.tile(w,(5,1)) * animoji[:,0::2].T + np.tile(boxes[:,0],(5,1)) - 1).T
        animoji[:,1::2] = (np.tile(h,(5,1)) * animoji[:,1::2].T + np.tile(boxes[:,1],(5,1)) - 1).T    
        # boxes_c = calibrate_box(boxes, reg)
        
        
        # boxes = boxes[py_nms(boxes, 0.6, "Minimum")]
        # keep = py_nms(boxes, 0.6, "Minimum")
        # boxes = boxes[keep]
        # landmark = landmark[keep]
        return None, boxes, landmark, animoji