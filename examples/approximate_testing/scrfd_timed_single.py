import datetime
import numpy as np
import os
import os.path as osp
import glob
import insightface
import cv2
import time
import os
import importlib.util
import sys
import os

assert insightface.__version__>='0.4'


def detect_person(img, detector):
    bboxes, kpss = detector.detect(img)
    bboxes = np.round(bboxes[:,:4]).astype(int)
    kpss = np.round(kpss).astype(int)
    kpss[:,:,0] = np.clip(kpss[:,:,0], 0, img.shape[1])
    kpss[:,:,1] = np.clip(kpss[:,:,1], 0, img.shape[0])
    vbboxes = bboxes.copy()
    vbboxes[:,0] = kpss[:, 0, 0]
    vbboxes[:,1] = kpss[:, 0, 1]
    vbboxes[:,2] = kpss[:, 4, 0]
    vbboxes[:,3] = kpss[:, 4, 1]
    return bboxes, vbboxes


def run_inference(detector, img_paths):

    total_time = 0
    count = 0
    first_check = 0
    first_time = 0

    for img_path in img_paths:

        count+=1

        img = cv2.imread(img_path)
        time_start = time.perf_counter()
        bboxes, vbboxes = detect_person(img, detector)
        time_stop = time.perf_counter()
        time_elapsed = time_stop - time_start
        total_time += time_elapsed

        if first_check < 1:
            first_time = time_elapsed

        first_check+=1

        # print("Elapsed Time for file: ", img_path, round(time_elapsed, 4))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            vbbox = vbboxes[i]
            x1,y1,x2,y2 = bbox
            vx1,vy1,vx2,vy2 = vbbox
            cv2.rectangle(img, (x1,y1)  , (x2,y2) , (0,255,0) , 1)
            alpha = 0.8
            color = (255, 0, 0)
            for c in range(3):
                img[vy1:vy2,vx1:vx2,c] = img[vy1:vy2, vx1:vx2, c]*alpha + color[c]*(1.0-alpha)
            cv2.circle(img, (vx1,vy1) , 1, color , 2)
            cv2.circle(img, (vx1,vy2) , 1, color , 2)
            cv2.circle(img, (vx2,vy1) , 1, color , 2)
            cv2.circle(img, (vx2,vy2) , 1, color , 2)
        filename = img_path.split('/')[-1]
        cv2.imwrite('output/%s'%filename, img)
    
    return (count, total_time, first_time)


if __name__ == '__main__':
    import glob
    image_dirs = ['10--People_Marching', '23--Shoppers', '33--Running']
    dir_choice = 2
    detector = insightface.model_zoo.get_model('scrfd_person_2.5g.onnx', download=True)
    detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))
    # img_paths = glob.glob(f'data/images/{image_dirs[0]}/*.jpg')
    # img_paths = glob.glob(f'data/images/{image_dirs[1]}/*.jpg')
    img_paths = glob.glob(f'data/images/{image_dirs[2]}/*.jpg')
    count, total_time, first_time = run_inference(detector, img_paths)
    print(f"Image Directory: {image_dirs[dir_choice]} \nImage Count: {round(count, 4)} \nTotal Time: {round(total_time, 4)} \nAvg. Time: {round(total_time/count, 4)} \nFirst Time: {round(first_time, 4)}")
    print('')
    print('')

