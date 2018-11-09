import sys
import numpy as np
import cv2
import os
from data_utils import IoU, read_annotation, convert_to_square
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../core'))
from detection import detect_pnet, detect_rnet


# def save_hard_example(images, det_boxes, gt_bboxes, image_size, save_dir):
    # pos_save_dir = os.path.join(save_dir,'positive')
    # part_save_dir = os.path.join(save_dir, 'part')
    # neg_save_dir = os.path.join(save_dir, 'negative')
    
    # if not os.path.exists(save_dir):
        # os.mkdir(save_dir)
    # if not os.path.exists(pos_save_dir):
        # os.mkdir(pos_save_dir)
    # if not os.path.exists(part_save_dir):
        # os.mkdir(part_save_dir)
    # if not os.path.exists(neg_save_dir):
        # os.mkdir(neg_save_dir)
    # # save files
    # neg_label_file = os.path.join(save_dir, 'neg_{}.txt'.format(image_size))
    # neg_file = open(neg_label_file, 'w')

    # pos_label_file = os.path.join(save_dir, 'pos_{}.txt'.format(image_size))
    # pos_file = open(pos_label_file, 'w')

    # part_label_file = os.path.join(save_dir, 'part_{}.txt'.format(image_size))
    # part_file = open(part_label_file, 'w')
    # num_of_images = len(images)
    # print('processing {} images in total'.format(num_of_images))
    # assert len(det_boxes) == num_of_images, '{}/{}'.format(len(det_boxes),num_of_images)
    # assert len(gt_bboxes) == num_of_images, '{}/{}'.format(len(gt_bboxes),num_of_images)

    # n_idx = 0
    # p_idx = 0
    # d_idx = 0
    # image_done = 0
    # for img, dets, gts in zip(images, det_boxes, gt_bboxes):
        # gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        # if image_done % 100 == 0:
            # print("%d images done" % image_done)
        # image_done += 1

        # if dets is None:
            # continue
        # if dets.shape[0] == 0:
            # continue
        # dets = convert_to_square(dets)
        # dets[:, 0:4] = np.round(dets[:, 0:4])
        # neg_num = 0
        # for box in dets:
            # x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            # width = x_right - x_left + 1
            # height = y_bottom - y_top + 1

            # if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                # continue

            # Iou = IoU(box, gts)
            # cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            # resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    # interpolation=cv2.INTER_LINEAR)
           
            # if np.max(Iou) < 0.3 and neg_num < 60:
                # save_file = os.path.join(neg_save_dir, '{}.jpg'.format(n_idx))
                # neg_file.write(save_file + ' 0\n')
                # cv2.imwrite(save_file, resized_im)
                # n_idx += 1
                # neg_num += 1
            # else:
                # idx = np.argmax(Iou)
                # assigned_gt = gts[idx]
                # x1, y1, x2, y2 = assigned_gt

                # offset_x1 = (x1 - x_left) / float(width)
                # offset_y1 = (y1 - y_top) / float(height)
                # offset_x2 = (x2 - x_right) / float(width)
                # offset_y2 = (y2 - y_bottom) / float(height)

                # if np.max(Iou) >= 0.65:
                    # save_file = os.path.join(pos_save_dir, '{}.jpg'.format(p_idx))
                    # pos_file.write('{} 1 {:.2} {:.2} {:.2} {:.2}\n'.format(save_file,\
                                     # offset_x1, offset_y1, offset_x2, offset_y2))
                    # cv2.imwrite(save_file, resized_im)
                    # p_idx += 1

                # elif np.max(Iou) >= 0.4:
                    # save_file = os.path.join(part_save_dir, '{}.jpg'.format(d_idx))
                    # part_file.write('{} -1 {:.2} {:.2} {:.2} {:.2}\n'.format(save_file, \
                        # offset_x1, offset_y1, offset_x2, offset_y2))
                    # cv2.imwrite(save_file, resized_im)
                    # d_idx += 1
    # neg_file.close()
    # part_file.close()
    # pos_file.close()
    
    # return pos_label_file, neg_label_file, part_label_file
    
def save_hard_example(images, det_boxes, gt_bboxes, image_size, save_dir):
    print('Save hard example.')
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
    assert len(det_boxes) == num_of_images, '{}/{}'.format(len(det_boxes), num_of_images)
    assert len(gt_bboxes) == num_of_images, '{}/{}'.format(len(gt_bboxes), num_of_images)

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

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, '{}.jpg'.format(d_idx))
                    part_file.write('{} -1 {:.2} {:.2} {:.2} {:.2}\n'.format(save_file, \
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()
    
    return [pos_label_file, neg_label_file, part_label_file]

def gen_PNet_bbox_data(anno_file, im_dir, save_dir, debug=False):
    print('gen PNet bbox data.')
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
    print "{} pics in total".format(num)
    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # dont care
    idx = 0
    box_idx = 0
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_path = annotation[0]
        bbox = map(float, annotation[1:])
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
        idx += 1
        if idx % 1000 == 0:
            print '{}/{} images done'.format(idx, num)
            
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
         
    print '{} images done, pos: {} part: {} neg: {}'.format(idx, p_idx, d_idx, n_idx)    
    f1.close()
    f2.close()
    f3.close() 
    return pos_list_file, neg_list_file, part_list_file
    
def gen_RNet_bbox_data(anno_file, im_dir, save_dir, model_path, debug=False):
    print('gen RNet bbox data.')
    image_size = 24
    print model_path
    pnet = detect_pnet(model_path)
    det_boxes = []
    images = []
    imagepath, gt_bboxes = read_annotation(im_dir,anno_file)
    if debug:
        imagepath = imagepath[:100]
        gt_bboxes = gt_bboxes[:100]
    num = len(imagepath)
    print 'Number file: {}'.format(num)
    for i, imagepath in enumerate(imagepath):
        image = cv2.imread(imagepath)
        _, boxes_c, _ = pnet.predict(image)
        det_boxes.append(boxes_c)
        images.append(image)
        if (i+1) % 100 == 0:
            print i+1, num

    print(len(images), len(det_boxes), len(gt_bboxes))
    
    return save_hard_example(images, det_boxes, gt_bboxes, image_size, save_dir)

def gen_ONet_bbox_data(anno_file, im_dir, save_dir, model_path, debug=False):
    print('gen ONet bbox data.')
    image_size = 48
    print model_path
    pnet = detect_pnet(model_path[0])
    rnet = detect_rnet(model_path[1])
    det_boxes = []
    images = []
    imagepath, gt_bboxes = read_annotation(im_dir,anno_file)
    if debug:
        imagepath = imagepath[:100]
        gt_bboxes = gt_bboxes[:100]
    num = len(imagepath)
    print 'Number file: {}'.format(num)
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
            print i+1, num   

    print(len(images), len(det_boxes), len(gt_bboxes))
    return save_hard_example(images, det_boxes, gt_bboxes, image_size, save_dir)


if __name__ == '__main__':
    debug = True
    
    anno_file = 'widerface/wider_face_train.txt'
    im_dir = 'widerface/WIDER_train/images'
    save_dir = './12'
    files = gen_PNet_bbox_data(anno_file, im_dir, save_dir, debug=debug)
    
    # image_size = 24
    # anno_file = 'widerface/wider_face_train_bbx_gt.txt'
    # im_dir = 'widerface/WIDER_train/images'
    # save_dir = './24'    
    # model_path = '../models/PNet/PNet-18'
    # print gen_RNet_bbox_data(anno_file, im_dir, save_dir, model_path, debug=debug)
    
    # image_size = 48
    # anno_file = 'widerface/wider_face_train_bbx_gt.txt'
    # im_dir = 'widerface/WIDER_train/images'
    # save_dir = './48'   
    # model_path = ['../models/PNet/PNet-16', '../models/RNet/RNet-18']
    # print gen_ONet_bbox_data(anno_file, im_dir, save_dir, model_path, debug=debug)