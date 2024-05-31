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
import playsound

MAX_CLASSES = 6

# start webcam
#cap = cv2.VideoCapture("./datasets/real/vid/20240430_180548.mp4")
#cap = cv2.VideoCapture("./datasets/real/vid/20240430_180635.mp4")
#cap = cv2.VideoCapture("./datasets/real/vid/unicorn_eclipse_hd2_A.mp4")
#cap = cv2.VideoCapture("./datasets/real/vid/output3.avi")
#cap = cv2.VideoCapture("./datasets/real/vid/winmau_blade_6_C.avi")
#cap = ScreenVideoCapture(pick=True)
cap = cv2.VideoCapture("./datasets/real/vid/home01.avi")
time_mult=.25#0.0001
fps = 21.0

board_img_path = 'generator/3D/Boards/canaveral_t520.jpg'
#board_img_path = 'generator/3D/Boards/unicorn-eclipse-hd2.jpg'
#board_img_path = 'generator/3D/Boards/winmau_blade_5.jpg'

board = Board(board_img_path.replace(".jpg",".json"))

with open(board_img_path.replace(".jpg",".json")) as f:
    board_def = json.load(f)

detector = TargetDetector(board_img_path)

status_colors = {"not_detected": (0,0,255), "detected": (0,255,0), "opencv_detected": (0,160,75), "locked":(160,160)}

# cap.set(3, 640)
# cap.set(4, 480)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# model

#model = YOLO("best_m.pt")
model = YOLO("best_s_tip_boxes640_B.pt")

temporal_model = YOLO("best_temporal_A.pt")
#model = YOLO("last.pt")
model_train_size = 640
force_opencv_detector = True
use_clahe = False
temporal_detection = True
temporal_detection_on = False
temporal_filter = 300 # ms to wait to validate (try to filter dart in flight, not yet landed) TODO: flight detection?

locked_target_delay = 5 # 10 frames without moving => stable
locked_frames = 0
locked = False
last_cal_pts = None

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

#keep only best confidence of a given class
def filter_res(res, filter_classes = [1,2,3,4]):
    pts_cal_conf= [0.0] * MAX_CLASSES
    best_id = [None] * MAX_CLASSES
    
    keep = []
    # coordinates
    for i,box in enumerate(res):
        # confidence
        confidence = box["conf"]

        # class name
        cls = box["cls"]
        if(cls in filter_classes):
            if(confidence<=pts_cal_conf[cls]): continue
            pts_cal_conf[cls] = confidence
            best_id[cls] = i
        else:
            keep.append(i)

    keep.extend([i for i in best_id if i is not None])
    return [res[i] for i in keep]


def infer(img, mod = None):
    if(mod is None):
        mod = model
    results = mod(img, stream=True, max_det=25, conf = 0.3, augment=False,agnostic_nms=True, vid_stride=28,verbose=False)

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


def draw(img, res, box_cols = [(255,255,0),(0,215,255),(180, 105, 255),(112,255,202),(114,128,250),(255,62,191)], filter=None, status = "not_detected", force_draw_all = False):
    if(force_draw_all or status != "not_detected"):
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
                text = f"{scores[0]} ({confidence:.2f})"
                score = True

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1.2 if score else 0.6
            color = box_cols[cls]
            thickness = 4 if score else 1

            cv2.putText(img, text, org, font, fontScale, color, thickness)

        for p in pts_cal:
            cv2.circle(img, p.astype(np.int32),4,(200,180,60),cv2.FILLED)
    
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_ISOLATED,value=status_colors[status])
    return img

ts = time.time()
last = None
last_diff = None

last_dart = None
last_dart_time = 0

while True:
    if(time_mult > 0):
        cap.set(cv2.CAP_PROP_POS_MSEC, (time.time()-ts)*1000*time_mult)
    success, img = cap.read()
    if(not success):
        continue
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

    pts_cal = None
    if(locked):
        pts_cal = last_cal_pts
    else:
        res = infer(img)
        pts_cal = find_cal_pts(res)

        opencv_detected = False
        if(not locked and (force_opencv_detector or pts_cal is None)):
            tps = time.time()
            found_cals, M, conf = detector.detect(img)
            if(M is not None):
                pts_cal = found_cals
                for i,p in enumerate(pts_cal):
                    v = {"x1": p[0]-10, "y1": p[1]-10,"x2": p[0]+10, "y2": p[1]+10, 'conf':conf, "cls":i+1}
                    res.append(v)
                opencv_detected = True
            print(f"OpenCV target Detector: {int((time.time()-tps)*1000)} ms")

        # Filter calibration points if multiple
        res = filter_res(res)
        pts_cal = find_cal_pts(res)

    if(cropped and temporal_detection and pts_cal is not None):
        tps = time.time()
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) / 255.0
        if(last is not None):
            #diff = abs(img_gray-last)
            diff = cv2.absdiff(img_gray, last)
            #diff = diff *diff
            #diff[diff<0.4]=0
            if(last_diff is not None):
                #delta = abs(diff-last_diff)
                delta = cv2.absdiff(diff, last_diff)

                # Check if right amount of pixels is beeing modified before inference
                non_null = np.sum(delta>0.04)
                pct = non_null * 100.0 / (delta.shape[0]*delta.shape[1])
                #print(f"PCT:{pct:.2f}")
                if(pct>0.1 and pct < 13):
                    delta = cv2.cvtColor((delta*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
                    cv2.imshow("delta", delta)
                    temporal_detection_on = True
                    
                    res = infer(delta, temporal_model)
                    res = [r for r in res if r['conf']>0.4]
                    cpt_darts = 0
                    cpt_tips = 0
                    res = filter_res(res,filter_classes=[0,1])
                    for i in range(len(res)):
                        if(res[i]["cls"]==1):
                            cpt_darts += 1
                            res[i]["cls"] = 5
                        elif(res[i]["cls"]==0):
                            cpt_tips += 1
                    if(cpt_tips>0 and cpt_darts>0):
                        print(res)
                        if(last_dart is None):
                            last_dart = res
                            last_dart_time = time.time()
                        else:
                            last_dart.extend(res)
                            last_dart = filter_res(last_dart,filter_classes=[0,5])

                        # win = f"{int((time.time()-ts)*1000)} ms"
                        # delta = draw(delta,res,force_draw_all=True)
                        # # img = draw(img,res,force_draw_all=True)
                        # # cv2.imshow(f"{win}", img)
                        # cv2.imshow(f"{win}_temporal", delta)
                        
                        # # for r in res:
                        # playsound.playsound("sound/bow-release-bow-and-arrow-4-101936.mp3", False)
                else:
                    res = []
                    #cv2.waitKey()
            #cv2.imshow("diff", diff)
            last_diff = diff
            
        else:
            last = img_gray
        #print(f"Temporal computation: {int((time.time()-tps)*1000)} ms")

    # check if moved
    if(locked_target_delay>0 and not locked and pts_cal is not None):
        if(last_cal_pts is not None):
            diff = (last_cal_pts-pts_cal)
            moved = np.max(diff)
            if(moved <2):
                locked_frames += 1
                if(locked_frames>=locked_target_delay):
                    locked = True
        last_cal_pts = pts_cal

    # First crop resize after initial detection
    if(pts_cal is not None):
        if(crop is None):
            center = seg_intersect(pts_cal[0],pts_cal[1],pts_cal[2],pts_cal[3])
            x1 = int(center[0]-12)
            y1 = int(center[1]-12)
            x2 = int(center[0]+12)
            y2 = int(center[1]+12)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            mode = 0
            if(mode == 0):
                dis = (pts_cal-center) * 1.35
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

    # Check if last dart ios validated
    # print(last_dart)
    if(last_dart is not None and (time.time()-last_dart_time)*1000 >= temporal_filter):
        print("Last dart:",last_dart)
        last_dart = None

        # scores = board.get_dart_scores(pts_cal,[[(x1+x2)*0.5,(y1+y2)*0.5]])
        # text = f"{scores[0]} ({confidence:.2f})"
        
        playsound.playsound("sound/bow-release-bow-and-arrow-4-101936.mp3", False)

    
    status = "not_detected"
    if(pts_cal is not None):
        status = "opencv_detected" if opencv_detected else "detected"
        if(locked):
            status = "locked"
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