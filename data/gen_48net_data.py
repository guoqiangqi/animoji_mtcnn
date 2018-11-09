import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../core'))

import numpy as np
import cv2
from detection import detect_pnet, detect_rnet
from data_utils import IoU, read_annotation, convert_to_square


def expand(x1,y1,x2,y2, img_shape):
    w = x2 - x1 + 1
    h = y2 - y1 + 1

    max_side = np.maximum(h,w)*(np.random.rand()*0.5+1.1) 
    # print(max_side, max(h,w), max_side/max(w,h))
    imgh, imgw, _ = img_shape
    out = []
    for _ in range(5):
        _x1 = x1 + w*0.5 - max_side*0.5
        _y1 = y1 + h*0.5 - max_side*0.5
        _x2 = x1 + max_side - 1
        _y2 = y1 + max_side - 1
        
        _x1 = int(_x1 + 0.5)
        _x2 = int(_x2 + 0.5)
        _y1 = int(_y1 + 0.5)
        _y2 = int(_y2 + 0.5)
        
        pad_x1, pad_y1, pad_x2, pad_y2 = 0, 0, 0, 0
        ispad = False
        if _x1 < 0:
            _x1 = 0
            pad_x1 = -_x1
            ispad = True
            
        if _y1 < 0:
            _y1 = 0
            pad_y1 = -_y1
            ispad = True
            
        if _x2 > imgw - 1:
            _x2 = imgw -1
            pad_x2 = _x2 - imgw + 1
            ispad = True
            
        if _y2 > imgh - 1:
            _y2 > imgh -1
            pad_y2 = _y2 - imgh + 1
            ispad = True
            
        out.append(((_x1,_y1,_x2,_y2),(pad_x1,pad_y1,pad_x2,pad_y2),ispad))
    return out

def save_hard_example(images, det_boxes, gt_bboxes, image_size, save_dir):
    pos_save_dir = os.path.join(save_dir,'positive')
    part_save_dir = os.path.join(save_dir, 'part')
    neg_save_dir = os.path.join(save_dir, 'negative')
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
    # save files
    neg_label_file = os.path.join(save_dir, 'neg_{}.txt'.format(image_size))
    neg_file = open(neg_label_file, 'w')

    pos_label_file = os.path.join(save_dir, 'pos_{}.txt'.format(image_size))
    pos_file = open(pos_label_file, 'w')

    part_label_file = os.path.join(save_dir, 'part_{}.txt'.format(image_size))
    part_file = open(part_label_file, 'w')
    num_of_images = len(images)
    print('processing {} images in total'.format(num_of_images))
    assert len(det_boxes) == num_of_images, '{}/{}'.format(len(det_boxes),num_of_images)
    assert len(gt_bboxes) == num_of_images, '{}/{}'.format(len(gt_bboxes),num_of_images)

    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for img, dets, gts in zip(images, det_boxes, gt_bboxes):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if image_done % 100 == 0:
            print('{}/{} images done'.format(image_done, num_of_images))
        image_done += 1

        if dets is None:
            continue
        if dets.shape[0] == 0:
            continue
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1
            _box = expand(x_left,y_top,x_right,y_bottom,img.shape)
            
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)
           
            if np.max(Iou) < 0.3 and neg_num < 60:
                save_file = os.path.join(neg_save_dir, '{}.jpg'.format(n_idx))
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, '{}.jpg'.format(p_idx))
                    pos_file.write('{} 1 {:.2} {:.2} {:.2} {:.2}\n'.format(save_file,\
                                     offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                    for (_x1,_y1,_x2,_y2),_, _ in _box:
                        w = _x2 - _x1 + 1
                        h = _y2 - _y1 + 1
                        box_image = img[_y1:_y2+1, _x1:_x2+1,:]
                        resized_im = cv2.resize(box_image, (image_size, image_size),\
                                    interpolation=cv2.INTER_LINEAR)
                        offset_x1 = (x1 - _x1) / float(w)
                        offset_y1 = (y1 - _y1) / float(h)
                        offset_x2 = (x2 - _x2) / float(w)
                        offset_y2 = (y2 - _y2) / float(h)  
                        
                        # t_x1 = offset_x1*image_size
                        # t_y1 = offset_y1*image_size
                        # t_x2 = image_size + offset_x2*image_size
                        # t_y2 = image_size + offset_y2*image_size
                        # t_w = int(t_x2 - t_x1 + 1)
                        # t_h = int(t_y2 - t_y1 + 1)
                        
                        # cv2.rectangle(resized_im, (int(t_x1),int(t_y1)),\
                                # (int(t_x2),int(t_y2)),(0,0,255))
                        
                        save_file = os.path.join(pos_save_dir, '{}.jpg'.format(p_idx))
                        pos_file.write('{} 1 {:.2} {:.2} {:.2} {:.2}\n'.format(save_file,\
                                     offset_x1, offset_y1, offset_x2, offset_y2))
                        cv2.imwrite(save_file, resized_im)
                        p_idx += 1                        

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, '{}.jpg'.format(d_idx))
                    part_file.write('{} -1 {:.2} {:.2} {:.2} {:.2}\n'.format(save_file, \
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()
    
    return pos_label_file, neg_label_file, part_label_file

def gen_ONet_bbox_data(anno_file, im_dir, save_dir, model_path, debug=False):
    image_size = 48
    print(model_path)
    pnet = detect_pnet(model_path[0])
    rnet = detect_rnet(model_path[1])
    det_boxes = []
    images = []
    imagepath, gt_bboxes = read_annotation(im_dir,anno_file)
    if debug:
        imagepath = imagepath[:10]
        gt_bboxes = gt_bboxes[:10]
    num = len(imagepath)
    print('Number file: {}'.format(num))
    for i, imagepath in enumerate(imagepath):
        image = cv2.imread(imagepath)
        _, boxes_c, _ = pnet.predict(image)
        if boxes_c is None:
            det_boxes.append(boxes_c)
            images.append(image)
            continue 
        # if len(boxes_c) == 0:
            # continue
        _, boxes_c, _ = rnet.predict(image, boxes_c)
        det_boxes.append(boxes_c)
        images.append(image)
        if (i+1) % 100 == 0:
            print(i+1, num)   

    print(len(images), len(det_boxes), len(gt_bboxes))
    print('Save hard example.')
    return save_hard_example(images, det_boxes, gt_bboxes, image_size, save_dir)
    

if __name__ == '__main__':
    debug = True
    image_size = 48
    anno_file = 'widerface/wider_face_train_bbx_gt.txt'
    im_dir = 'widerface/WIDER_train/images'
    save_dir = './48'
    
    model_path = ['../models/PNet/PNet-16', '../models/RNet/RNet-18']
    print(gen_ONet_bbox_data(anno_file, im_dir, save_dir, model_path, debug=debug))
