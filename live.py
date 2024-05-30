from ultralytics import YOLO
import cv2
import math
import numpy as np
from generator.annotate import seg_intersect
import time
import os
import json
from target_detector import TargetDetector
from board import Board, transform_points
from videocapture import ScreenVideoCapture


# start webcam
#cap = cv2.VideoCapture("./datasets/real/vid/20240430_180548.mp4")
#cap = cv2.VideoCapture("./datasets/real/vid/20240430_180635.mp4")
#cap = cv2.VideoCapture("./datasets/real/vid/winmau_blade_6_A.mp4")
#cap = cv2.VideoCapture("./datasets/real/vid/output3.avi")
#cap = cv2.VideoCapture("./datasets/real/vid/winmau_blade_6_C.avi")
cap = ScreenVideoCapture(pick=True)

time_mult=1.5#0.0001
fps = 21.0

#board_img_path = 'generator/3D/Boards/canaveral_t520.jpg'
#board_img_path = 'generator/3D/Boards/unicorn-eclipse-hd2.jpg'
board_img_path = 'generator/3D/Boards/winmau_blade_6.jpg'
board = Board(board_img_path.replace(".jpg",".json"))

with open(board_img_path.replace(".jpg",".json")) as f:
    board_def = json.load(f)

detector = TargetDetector(board_img_path)

status_colors = {"not_detected": (0,0,255), "detected": (0,255,0), "opencv_detected": (0,160,75)}

# cap.set(3, 640)
# cap.set(4, 480)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# model

#model = YOLO("best_m.pt")
model = YOLO("best_s_tip_boxes640_B.pt")
#model = YOLO("last.pt")
model_train_size = 640
force_opencv_detector = True
use_clahe = False
stablize = False

# object classes
classNames = ["tip", "cal1", "cal2", "cal3", "cal4", "dart"]
pts_cal_dst = None
crop = None

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))

def find_cal_pts(res, confidence_min = 0.5):
    pts_cal_conf= [0.0,0.0,0.0,0.0]
    pts_cal = np.zeros((4,2))
    pts_cal_found = 0
    
    # coordinates
    for box in res:
        # confidence
        confidence = box["conf"]
        #print("Confidence --->",confidence)

        if(confidence<confidence_min):
            continue

        # class name
        cls = box["cls"]
        if(cls>0 and cls<5):
            #if(confidence<pts_cal_conf[cls-1]): continue
            pts_cal_conf[cls-1] = confidence
            pts_cal_found += 1
            pts_cal[cls-1] = np.array([(box["x1"]+box["x2"])*0.5,(box["y1"]+box["y2"])*0.5])

    return None if pts_cal_found <4 else pts_cal

def filter_res(res):
    pts_cal_conf= [0.0,0.0,0.0,0.0]
    best_id = [None,None,None,None]
    
    keep = []
    # coordinates
    for i,box in enumerate(res):
        # confidence
        confidence = box["conf"]

        # class name
        cls = box["cls"]
        if(cls>0 and cls<5):
            if(confidence<=pts_cal_conf[cls-1]): continue
            pts_cal_conf[cls-1] = confidence
            best_id[cls-1] = i
        else:
            keep.append(i)

    keep.extend([i for i in best_id if i is not None])
    return [res[i] for i in keep]


def infer(img, mod = None):
    if(mod is None):
        mod = model
    results = mod(img, stream=True, max_det=25, conf = 0.3, augment=False,agnostic_nms=True, vid_stride=28)

    res = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])
            res.append({"x1":x1, "y1":y1,"x2":x2, "y2":y2, "conf":confidence, "cls": cls})
    return res

ts = time.time()
last = None
last_diff = None

while True:
    if(time_mult > 0):
        cap.set(cv2.CAP_PROP_POS_MSEC, (time.time()-ts)*1000*time_mult)
    success, img = cap.read()
    ratio = None
    cropped = False
    if(crop is not None):
        img = (img[crop[1]:crop[3],crop[0]:crop[2],:])
        ratio = model_train_size/np.max(img.shape)
        img= cv2.resize(img,(int(img.shape[1]*ratio),int(img.shape[0]*ratio)),interpolation=cv2.INTER_LANCZOS4)
        #img = clahe.apply(img)
        if(use_clahe):
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        cropped = True

    res = infer(img)

    pts_cal = find_cal_pts(res)

    opencv_detected = False
    if(force_opencv_detector or pts_cal is None):
        ts = time.time()
        found_cals, M, conf = detector.detect(img)
        if(M is not None):
            pts_cal = found_cals
            for i,p in enumerate(pts_cal):
                v = {"x1": p[0]-10, "y1": p[1]-10,"x2": p[0]+10, "y2": p[1]+10, 'conf':conf, "cls":i+1}
                res.append(v)
            opencv_detected = True
        print(f"OpenCV target Detector: {int((time.time()-ts)*1000)} ms")

    res = filter_res(res)
    pts_cal = find_cal_pts(res)

    if(cropped and pts_cal_dst is None and pts_cal is not None):
        pts_cal_dst = pts_cal.copy()

    if(stablize and pts_cal is not None and pts_cal_dst is not None):
        M = cv2.getPerspectiveTransform(pts_cal.astype(np.float32), pts_cal_dst.astype(np.float32))
        im = cv2.warpPerspective(img.copy(), M, (img.shape[1],img.shape[0]))

        cv2.imshow("Mon Image", im)
        img_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) / 255.0
        if(last is not None):
            diff = abs(img_gray-last)
            #diff = diff *diff
            #diff[diff<0.4]=0
            if(last_diff is not None):
                delta = abs(diff-last_diff)
                b = 50
                delta[:b,:] = 0
                delta[-b:,:] = 0
                delta[:,:b] = 0
                delta[:,-b:] = 0

                kernel = np.ones((3,3),np.uint8)
                delta = cv2.erode(delta,kernel,iterations = 1)
                delta = cv2.dilate(delta,kernel,iterations = 10)
                delta = cv2.erode(delta,kernel,iterations = 5)

                cv2.imshow("delta", (delta>0.35)*1.0)
            last_diff = diff
            
        else:
            last = img_gray
  
    # Filter calibration points if multiple

    if(pts_cal is not None):
        #center = sum(pts_cal)
        if(crop is None):
            center = seg_intersect(pts_cal[0],pts_cal[1],pts_cal[2],pts_cal[3])
            x1 = int(center[0]-12)
            y1 = int(center[1]-12)
            x2 = int(center[0]+12)
            y2 = int(center[1]+12)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            mode = 0
            if(mode == 0):
                dis = (pts_cal-center) * 1.6
                max = np.max(np.abs(dis))
                # print(center)
                # print(max)
                crop = [int(np.clip(center[0]-max,0,width)),
                        int(np.clip(center[1]-max,0,height)),
                        int(np.clip(center[0]+max,0,width)),
                        int(np.clip(center[1]+max,0,height))]
                
                # sm = (img[crop[1]:crop[3],crop[0]:crop[2],:])
                # ratio = model_train_size/np.max(img.shape)
                # sm= cv2.resize(img,(int(img.shape[1]*ratio),int(img.shape[0]*ratio)),interpolation=cv2.INTER_LANCZOS4)
                # cv2.imshow("sm",sm)
                # r2 = infer(sm)
                # pts_cal_dst = find_cal_pts(r2)
                # print("FOUND DST CAL:",pts_cal_dst)


            # elif(mode==1):
            #     outer = center +(pts_cal-center) * 1.3
    # else:
    #     crop = None
    #     cropped = False

    def draw(img, res, box_cols = [(255,255,0),(0,215,255),(180, 105, 255),(112,255,202),(114,128,250),(255,62,191)], filter=None, status = "not_detected"):
        if(status != "not_detected"):
            for box in res:
                # confidence
                confidence = box["conf"]
                #print("Confidence --->",confidence)

                # class name
                cls = box["cls"]
                if(filter is not None and cls not in filter):
                    continue


                x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"]) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), box_cols[cls], 1)



                text = f"{classNames[cls]} ({confidence:.2f})"
                score = False
                if(cls == 0):
                    scores = board.get_dart_scores(pts_cal,[[(x1+x2)*0.5,(y1+y2)*0.5]])
                    text = scores[0]
                    score = True


                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1.2 if score else 0.6
                color = box_cols[cls]
                thickness = 4 if score else 1

                cv2.putText(img, text, org, font, fontScale, color, thickness)

        
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_ISOLATED,value=status_colors[status])
        return img
    
    status = "not_detected"
    if(pts_cal is not None):
        status = "opencv_detected" if opencv_detected else "detected"
    img = draw(img, res, status=status)
    
    # res2 = infer(img,model2)
    # draw(img, res2, (0,0,255),filter=[5])
    #img_disp = img.resize((int(height*0.3),int(width*0.3)),refcheck=False)
    if(img is not None and img.shape[0]>0 and img.shape[1]>0):
        cv2.imshow('Webcam', img)
    # else:
    #     print(img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()