import sys
import numpy as np
import cv2
import os
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data'))
from utils import IoU, convert_to_square

def gen_mouse_bbox_data(neg_file, pos_file, save_dir, debug=False):
    pos_save_dir = os.path.join(save_dir,'positive')
    neg_save_dir = os.path.join(save_dir, 'negative')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    pos_list_file = os.path.join(save_dir, 'pos.txt')
    neg_list_file = os.path.join(save_dir, 'neg.txt')

    f1 = open(pos_list_file, 'w')
    f2 = open(neg_list_file, 'w')

    
    with open(neg_file, 'r') as f:
        neg_list = f.readlines()
        
    with open(pos_file, 'r') as f:
        pos_list = f.readlines()

    wmin, wmax, hmin, hmax = 1e10, 0, 1e10, 0
    for pos in pos_list:
        pos = pos.strip().split(' ')
        _x1, _y1, _x2, _y2 = list(map(int, pos[1:5]))
        w = _x2 - _x1 + 1
        h = _y2 - _y1 + 1
        
        wmin = min(wmin, w)
        wmax = max(wmax, w)
        hmin = min(hmin, h)
        hmax = max(hmax, h)
    print(wmin, wmax, hmin, hmax)
    # wmin, wmax, hmin, hmax = 21, 1399, 7, 773
        
    if debug:
        from random import shuffle
        shuffle(neg_list)
        shuffle(pos_list)
        neg_list = neg_list[:5]
        pos_list = pos_list[:10]
        
    neg_root = os.path.dirname(neg_file)
    pos_root = os.path.dirname(pos_file)
    num = len(pos_list) + len(neg_list)

    print('{} positive file.'.format(len(pos_list)))
    print('{} negative file.'.format(len(neg_list)))
    print('{} pics in total'.format(num))
    
    p_idx = 0 # positive
    n_idx = 0 # negative
    idx = 0
    for pos in pos_list:
        pos = pos.strip().split(' ')
        im_path = pos[0]
        mouse_box = list(map(int, pos[1:5]))
        bbox = list(map(int, pos[5:9]))
        img = cv2.imread(os.path.join(pos_root, im_path))
        idx += 1
        if idx % 500 == 0:
            print('{}/{} images done'.format(idx, num))
            
        x1,y1,x2,y2 = bbox 
        width = x2 - x1 + 1
        height = y2 - y1 + 1

        _x1, _y1, _x2, _y2 = mouse_box[:4]

        w = _x2 - _x1 + 1
        h = _y2 - _y1 + 1

        box = np.asarray(mouse_box)
        boxes = box.reshape(1, 4)
        boxes = convert_to_square(boxes)
        num_neg = 0
        while num_neg < 20:
            #neg_num's size [40,min(width, height) / 2],min_size:40 
            size = np.random.randint(10, min(width, height) / 2)
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])
            _, iou = IoU(crop_box, boxes)  
            
            if np.max(iou) < 0.3:
                cropped_im = img[ny : ny + size, nx : nx + size, :]
                save_file = os.path.join(neg_save_dir, '{}.jpg'.format(n_idx))
                f2.write('{} 0\n'.format(save_file))
                cv2.imwrite(save_file, cropped_im)
                n_idx += 1
                num_neg += 1
                
        num_neg = 0
        num_pos = 0
        i = 0
        while (i < 50 or num_pos < 5):
            i += 1
            # print(idx, p_idx)
            if i > 100:
                cropped_im = img[_y1:_y2, _x1:_x2, :]
                save_file = os.path.join(pos_save_dir, '{}.jpg'.format(p_idx))
                f1.write('{} 1\n'.format(save_file))
                cv2.imwrite(save_file, cropped_im)
                p_idx += 1
                num_pos += 1
                break
        
            size = np.random.randint(int(max(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)
            nx1 = int(max(_x1 + w / 2 + delta_x - size / 2, x1))
            ny1 = int(max(_y1 + h / 2 + delta_y - size / 2, y1))
            nx2 = min(nx1 + size, x2)
            ny2 = min(ny1 + size, y2)
            

            # if nx2 > x2 or ny2 > y2:
                # continue 
            crop_box = np.array([nx1, ny1, nx2, ny2])

            iou_Union, iou_Minimum = IoU(crop_box, boxes)
            # if i > 20:
                # cropped_im = img[_y1:_y2, _x1:_x2, :]
                # save_file = os.path.join(pos_save_dir, '{}.jpg'.format(p_idx))
                # f1.write('{} 1\n'.format(save_file))
                # cv2.imwrite(save_file, cropped_im)
                # p_idx += 1
                # num_pos += 1
                # continue

            if np.max(iou_Union) < 0.65 and num_neg >= 15:
                continue
                
            cropped_im = img[ny1 : ny2, nx1 : nx2, :]
            if np.max(iou_Union) >= 0.65:
                save_file = os.path.join(pos_save_dir, '{}.jpg'.format(p_idx))
                f1.write('{} 1\n'.format(save_file))
                cv2.imwrite(save_file, cropped_im)
                p_idx += 1
                num_pos += 1
            elif np.max(iou_Minimum) < 0.3:
                save_file = os.path.join(neg_save_dir, '{}.jpg'.format(n_idx))
                f2.write('{} 0\n'.format(save_file))
                cv2.imwrite(save_file, cropped_im)
                n_idx += 1 
                num_neg += 1
                
    # print('--------------------')
    for neg in neg_list: 
        neg = neg.strip().split(' ')
        im_path = neg[0]
        img = cv2.imread(os.path.join(neg_root, im_path))
        idx += 1
        if idx % 500 == 0:
            print('{}/{} images done'.format(idx, num))
            
        height, width, channel = img.shape

        for i in range(10):
            if max(height, width) < 11:
                continue
            size = np.random.randint(10, np.ceil(max(height, width)))

            if width - size < 1:
                x1 = 0
            else:
                x1 = np.random.randint(0, width-size)
            if height-size < 1:
                y1 = 0
            else:
                y1 = np.random.randint(0, height-size)
            x2 = x1 + size
            y2 = y1 + size
            x2 = min(x2, width)
            y2 = min(y2, height)

            crop_box = np.array([x1, y1, x2, y2])
            cropped_im = img[y1:y2, x1:x2, :]

            save_file = os.path.join(neg_save_dir, '{}.jpg'.format(n_idx))
            f2.write('{} 0\n'.format(save_file))
            cv2.imwrite(save_file, cropped_im)
            n_idx += 1

         
    print('{} images done, pos: {} neg: {}'.format(idx, p_idx, n_idx))    
    f1.close()
    f2.close()
    return pos_list_file, neg_list_file

if __name__ == '__main__':
    import shutil
    neg_list_file = 'face_detection/neg.txt'
    pos_list_file = 'rotated_image/mouseRectangelList.txt'
    save_dir = './train_image/mouse'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    files = gen_mouse_bbox_data(neg_list_file, pos_list_file, save_dir, debug=True)