import sys
import numpy as np
import cv2
import os
from data_utils import IoU

def gen_PNet_bbox_data(anno_file, im_dir, save_dir, debug=False):
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

    pos_list_file = os.path.join(save_dir, 'pos_12.txt')
    neg_list_file = os.path.join(save_dir, 'neg_12.txt')
    part_list_file = os.path.join(save_dir, 'part_12.txt')
    f1 = open(pos_list_file, 'w')
    f2 = open(neg_list_file, 'w')
    f3 = open(part_list_file, 'w')
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
        
    if debug:
        annotations = annotations[:500]
        
    num = len(annotations)
    print("{} pics in total".format(num))
    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # dont care
    idx = 0
    box_idx = 0
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_path = annotation[0]
        bbox = map(float, annotation[1:])
        boxes = np.array(list(bbox), dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
        idx += 1
        if idx % 1000 == 0:
            print('{}/{} images done'.format(idx, num))
            
        height, width, channel = img.shape

        neg_num = 0
        while neg_num < 50:
            #neg_num's size [40,min(width, height) / 2],min_size:40 
            size = np.random.randint(12, min(width, height) / 2)
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])
            Iou = IoU(crop_box, boxes)
            
            cropped_im = img[ny : ny + size, nx : nx + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                save_file = os.path.join(neg_save_dir, '{}.jpg'.format(n_idx))
                f2.write('{} 0\n'.format(save_file))
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue
            for i in range(5):
                size = np.random.randint(12, min(width, height) / 2)
                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes)
        
                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
        
                if np.max(Iou) < 0.3:
                    save_file = os.path.join(neg_save_dir, '{}.jpg'.format(n_idx))
                    f2.write('{} 0\n'.format(save_file))
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1        
            # generate positive examples and part faces
            for i in range(20):
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue 
                crop_box = np.array([nx1, ny1, nx2, ny2])
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
                cropped_im = img[ny1 : ny2, nx1 : nx2, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, '{}.jpg'.format(p_idx))
                    f1.write('{} 1 {:.2} {:.2} {:.2} {:.2}\n'.format(save_file, offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, '{}.jpg'.format(d_idx))
                    f3.write('{} -1 {:.2} {:.2} {:.2} {:.2}\n'.format(save_file, offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1
         
    print('{} images done, pos: {} part: {} neg: {}'.format(idx, p_idx, d_idx, n_idx))    
    f1.close()
    f2.close()
    f3.close() 
    return pos_list_file, neg_list_file, part_list_file

if __name__ == '__main__':
    anno_file = 'widerface/wider_face_train.txt'
    im_dir = 'widerface/WIDER_train/images'
    save_dir = './12'
    files = gen_PNet_bbox_data(anno_file, im_dir, save_dir, debug=False)