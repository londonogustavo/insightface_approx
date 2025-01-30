import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
import time

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

if __name__ == '__main__':
    import glob
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cap.set(cv2.CAP_PROP_FPS, 60)
    detector = insightface.model_zoo.get_model('scrfd_person_2.5g.onnx', download=True)
    detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))
    img_paths = glob.glob('data/images/*.jpg')
    while(True):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        _, frame = cap.read()
        #img = cv2.imread(img_path)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        time_start = time.perf_counter()
        bboxes, vbboxes = detect_person(img, detector)
        time_stop = time.perf_counter()
        print("Elapsed Time: ", time_stop - time_start)
        if(len(vbboxes) > 0 and len(bboxes) > 0):
            print("Detecting a person")
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
        
        cv2.imshow("window", img)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    filename = img_paths.split('/')[-1]
    cv2.imwrite('output/%s'%filename, img)
cap.release() 

