import os
import cv2
import numpy as np
import random
import sys
from data_utils import IoU, BBox
      
def rotate(img, bbox, landmark, alpha):
    center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat,(img.shape[1],img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark]) 
    face = img_rotated_by_alpha[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
    return (face, landmark_)


def flip(face, landmark):
    face_flipped_by_x = cv2.flip(face, 1)
    #mirror
    landmark_ = np.asarray([(1-x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return (face_flipped_by_x, landmark_)
    
def getDataFromTxt(txt, with_landmark=True):
    #get dirname
    dirname = os.path.dirname(txt)
    with open(txt, 'r') as fd:
        lines = fd.readlines()

    result = []
    for idx,line in enumerate(lines):
        line = line.strip()
        components = line.split(' ')
        img_path = os.path.join(dirname, components[0]) # file path
        # bounding box, (x1, y1, x2, y2)
        bbox = (components[1], components[3], components[2], components[4])
        bbox = list(map(float, bbox))
        # bbox = [float(_) for _ in bbox]
        bbox = list(map(int,bbox))
        # landmark
        if not with_landmark:
            result.append((img_path, BBox(bbox)))
            continue
        
        landmark = components[5:]
        assert((len(landmark)% 2) == 0)
        assert(len(landmark) in [10, 140]), len(landmark)
        if len(landmark) == 10:
            label = -2
        else:
            label = -3
        
        landmark = list(map(float, landmark))
        landmark = np.asarray(landmark, dtype=np.float32)
        landmark = landmark.reshape(-1, 2)
        
        result.append((img_path, BBox(bbox), label, landmark))
    return result
    
def GenLandmarkData(ftxt, net_type, size, output, argument=False, debug=False):
    dstdir = os.path.join(output, 'train_{}_landmark'.format(net_type))  
    if not os.path.exists(output): os.mkdir(output)
    if not os.path.exists(dstdir): os.mkdir(dstdir)
    assert(os.path.exists(dstdir) and os.path.exists(output))
    
    image_id = 0
    landmark_list_file = os.path.join(output,'landmark_{}.txt'.format(net_type))
    f = open(landmark_list_file,'w')
    data = getDataFromTxt(ftxt)
    if debug:
        data = data[:10]
    idx = 0
    #image_path bbox landmark(5*2)
    print('landmark')
    for (imgPath, bbox, label, landmarkGt) in data:
        #print imgPath
        F_imgs = []
        F_landmarks = []   
        imgPath = imgPath.replace('\\','/')
        img = cv2.imread(imgPath)
        assert(img is not None), imgPath
        img_h,img_w,img_c = img.shape
        gt_box = np.array([bbox.left,bbox.top,bbox.right,bbox.bottom])
        if min([bbox.left,bbox.top,bbox.right,bbox.bottom]) < 0:
            print([bbox.left,bbox.top,bbox.right,bbox.bottom])
            print(imgPath)
        f_face = img[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
        f_face = cv2.resize(f_face,(size,size))
   
        offset = np.array([gt_box[0], gt_box[1]],dtype=np.float32).reshape(-1, 2)
        Length = np.array([gt_box[2]-gt_box[0], gt_box[3]-gt_box[1]], dtype=np.float32).reshape(-1, 2)
        landmark = (landmarkGt - offset)/Length
 
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(-1))
                
        if argument:
            idx = idx + 1
            if idx % 1000 == 0:
                print('{}/{} images done'.format(idx, len(data)))
            x1, y1, x2, y2 = gt_box
            #width
            gt_w = x2 - x1 + 1
            #height
            gt_h = y2 - y1 + 1        
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            #random shift
            for i in range(20):
                bbox_size = np.random.randint(int(min(gt_w, gt_h) * 0.9), np.ceil(1.5 * max(gt_w, gt_h)))
                delta_x = np.random.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = np.random.randint(-gt_h * 0.2, gt_h * 0.2)
                nx1 = max(x1+gt_w//2-bbox_size//2+delta_x,0)
                ny1 = max(y1+gt_h//2-bbox_size//2+delta_y,0)
                
                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size
                if nx2 > img_w or ny2 > img_h:
                    continue
                crop_box = np.array([nx1,ny1,nx2,ny2])
                cropped_im = img[ny1:ny2+1,nx1:nx2+1,:]
                resized_im = cv2.resize(cropped_im, (size, size))

                iou = IoU(crop_box, np.expand_dims(gt_box,0))
                if iou > 0.65:
                    F_imgs.append(resized_im)
                    #normalize

                    offset = np.array([nx1, ny1],dtype=np.float32).reshape(-1, 2)
                    Length = np.array([bbox_size, bbox_size], dtype=np.float32).reshape(-1, 2)
                    landmark_ = (landmarkGt - offset)/Length
                    F_landmarks.append(landmark_.reshape(-1))
                    # print(landmarkGt)
                    # print(landmark_)

                    bbox = BBox([nx1,ny1,nx2,ny2])  

                    #mirror     
                    # print(np.size(landmark_))
                    if np.size(landmark_) == 10:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(-1))
                    #rotate +5
                    if random.choice([0,1]) >= 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), 10)
                        #landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(-1))
                
                        #flip
                        if np.size(landmark_) == 10:
                            face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                            face_flipped = cv2.resize(face_flipped, (size, size))
                            F_imgs.append(face_flipped)
                            F_landmarks.append(landmark_flipped.reshape(-1))                
                    
                    #rotate -5
                    if random.choice([0,1]) >= 0: 
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), -10)
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(-1))
                        if np.size(landmark_) == 10:
                            face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                            face_flipped = cv2.resize(face_flipped, (size, size))
                            F_imgs.append(face_flipped)
                            F_landmarks.append(landmark_flipped.reshape(-1)) 
                    
            F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)

            for i in range(len(F_imgs)):
                if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                    continue
                if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                    continue
                    
                cv2.imwrite(os.path.join(dstdir,'{}.jpg'.format(image_id)), F_imgs[i])
                landmarks = ' '.join(map(str,list(F_landmarks[i])))
                f.write(os.path.join(dstdir,'{}.jpg {} {}\n'.format(image_id, label,landmarks)))
                image_id = image_id + 1
                if debug:
                    debug_path = os.path.join(output,'debug')
                    if not os.path.exists(debug_path):
                        os.mkdir(debug_path)
                    if image_id < 10:
                        image = F_imgs[i].copy()
                        landmark = F_landmarks[i]
                        assert((len(landmark)) % 2 == 0)
                        for j in range(len(landmark)//2):
                            cv2.circle(image, (int(landmark[2*j]*image.shape[1]),int(int(landmark[2*j+1]*image.shape[0]))), 3, (0,0,255))
                        cv2.imwrite(os.path.join(debug_path,'{}.jpg'.format(image_id)), image)

                    
    f.close()
    return F_imgs,F_landmarks,landmark_list_file

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    assert(sys.argv[1] in ["PNet","RNet","ONet"])
    # net = "ONet"
    net = sys.argv[1]
    if net == 'PNet':
        size = 12
    elif net == 'RNet':
        size = 24
    elif net == 'ONet':
        size = 48
    else:
        print('Net type error')
        exit()
    train_txt = 'lfw/trainImageList.txt'
    train_txt = '300w/ImageList.txt'
    output = 'landmark_test'
    imgs,landmarks,landmark_list_file = GenLandmarkData(train_txt, net, size, output, argument=True, debug=False)
   
