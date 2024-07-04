from ultralytics import YOLO
import cv2
import math
import numpy as np
import time
import os
import json
from target_detector import YoloTargetDetector
from board import Board, transform_points
from videocapture import ScreenVideoCapture, IPWebCamVideoCapture
import playsound

from tools import *

# start webcam
print("Initilize video capture...")
#cap = cv2.VideoCapture("./datasets/real/vid/20240430_180548.mp4")
#cap = cv2.VideoCapture("./datasets/real/vid/20240430_180635.mp4")
#cap = cv2.VideoCapture("./datasets/real/vid/unicorn_eclipse_hd2_A.mp4")
#cap = cv2.VideoCapture("./datasets/real/vid/output3.avi")
#cap = cv2.VideoCapture("./datasets/real/vid/winmau_blade_6_D.avi")
#cap = ScreenVideoCapture(pick=True)
#cap = cv2.VideoCapture("./datasets/real/vid/home02.avi")
cap = IPWebCamVideoCapture("https://BilboX:testip42@192.168.33.35:8080/shot.jpg")
time_mult=.25#0.0001#
fps = 21.0

board_img_path = 'generator/3D/Boards/canaveral_t520.jpg'
#board_img_path = 'generator/3D/Boards/unicorn-eclipse-hd2.jpg'
#board_img_path = 'generator/3D/Boards/winmau_blade_6.jpg'

print("Load board data...")
board = Board(board_img_path.replace(".jpg",".json"))

# print("Initialize OpenCV SIFT detector...")
# detector = SiftTargetDetector(board_img_path)
print("Initialize OpenCV YOLO detector...")
detector = YoloTargetDetector(board_img_path)

# cap.set(3, 640)
# cap.set(4, 480)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# model

#model = YOLO("best_m.pt")
print("Load general model...")
model = YOLO("best_s_tip_boxes640_B.pt")

print("Load temporal model...")
temporal_model = YOLO("best_temporal_A.pt")

#model = YOLO("last.pt")
model_train_size = 640
force_opencv_detector = True
use_clahe = False
temporal_detection = True
temporal_filter = 200 # ms to wait to validate (try to filter dart in flight, not yet landed) TODO: flight detection?

locked_target_delay = 5 # 10 frames without moving => stable
locked_frames = 0
locked = False
last_cal_pts = None
show_hit_debug = False

pts_cal_dst = None
crop = None

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))


def draw(img, res, filter=None, status = "not_detected", force_draw_all = False, detector=None):
    if(force_draw_all or status != "not_detected"):
        draw_inference_boxes(img, res, filter=filter, detector=detector)
        if(detector is not None):
            detector.draw_board(img)
    
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_ISOLATED,value=status_colors[status])
    return img

# Infer models to prevent slow first calls
print("Initializing models...")
infer(np.zeros((model_train_size,model_train_size,3)), model)
infer(np.zeros((model_train_size,model_train_size,3)), temporal_model)

pts_cal = None
perform_simple_interactive_calibration = True

if(perform_simple_interactive_calibration):
    print("Initial fit on first capture...")

    success, img = cap.read()
    key = -1
    res = []
    test = img.copy()
    manual_rot = 0
    pts_cal, M, conf = detector.detect(test,dbg=test)
    if(pts_cal is not None):
        crop = auto_crop(pts_cal, width, height)
        img = crop_img(img, crop, clahe= clahe if use_clahe else None)
    while True:
        test = img.copy()
        pts_cal, M, conf = detector.detect(img,dbg=test)

        # if(M is not None and manual_rot is not None):
        #     Rt = cv2.getRotationMatrix2D(center=(0,0), angle=manual_rot, scale=1)
        #     R = np.array([[1,0,0],[0,1,0],[0,0,1]],np.float32)
        #     R[:-1,:] = Rt
        #     print(R)
        #     tmp  = board.transform_cals(R)
        #     pts_cal = transform_points(tmp, M)
        #     test = img.copy()
        #     board.draw(test,pts_cal)

        cv2.imshow("Inital Calib", test)
        key = cv2.waitKey(0)
        if(key==ord('q')):
            exit(0)
        elif(key==ord('s')):
            break
        elif(key==ord('r')):
            pts_cal = None
            break
        elif(key==ord('o')):
            manual_rot += 18
        elif(key==ord('p')):
            manual_rot -= 18

    if(pts_cal is not None):
        locked = True
        last_cal_pts = pts_cal
        for i,p in enumerate(pts_cal):
            v = {"x1": p[0]-10, "y1": p[1]-10,"x2": p[0]+10, "y2": p[1]+10, 'conf':conf, "cls":i+1}
            res.append(v)    
        opencv_detected = True


print("Starting main loop...")
ts = time.time()
last = None
last_diff = None
last_dart_time = -1

while True:
    if(time_mult > 0):
        try:
            #cap.set(cv2.CAP_PROP_POS_MSEC, (time.time()-ts)*1000*time_mult)
            if(not cap.set(cv2.CAP_PROP_POS_FRAMES, (time.time()-ts)*fps*time_mult)):
                exit(0)
        except:
            exit(0)
    success, img = cap.read()
    if(not success):
        continue
    ratio = None
    cropped = False
    if(crop is not None):
        img = crop_img(img, crop, clahe=clahe if use_clahe else None)
        cropped = True

    if(locked):
        pts_cal = last_cal_pts
    else:
        res = infer(img)
        pts_cal = find_cal_pts(res)

        opencv_detected = False
        if(not locked and (force_opencv_detector or pts_cal is None)):
            tps = time.time()
            detect_dbg = img.copy()
            found_cals, M, conf = detector.detect(img, refine_pts=True, dbg = detect_dbg)
            cv2.imshow("Detection Debug", detect_dbg)
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
            diff = cv2.absdiff(img_gray, last)
            if(last_diff is not None):
                delta = cv2.absdiff(diff, last_diff)
                #delta = delta*delta
                
                # Check if right amount of pixels is beeing modified before inference
                non_null = np.sum(delta>0.04)
                pct = non_null * 100.0 / (delta.shape[0]*delta.shape[1])
                #print(f"PCT:{pct:.2f}")
                
                potential_dart_movement = (pct>0.4 and pct < 10)
                cv2.displayOverlay('Webcam', f"Potential movement:{pct:.2f} > {potential_dart_movement}")

                # dbg = cv2.cvtColor((delta*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
                # dbg = cv2.copyMakeBorder(dbg, 10, 10, 10, 10, cv2.BORDER_ISOLATED,value=(255,0,0) if potential_dart_movement else (50,0,0))
                # cv2.imshow("delta", dbg)
                cv2.imshow("delta",delta)
                if(potential_dart_movement):
                    print(f"{int((time.time()-ts)*1000)}: potential_dart_movement {pct:.1f}%")
                    playsound.playsound("sound/start-13691.mp3",False)
                    # induce a small delay to let dart land and avoid detect while flying
                    detect = False
                    if(temporal_filter<0):
                        detect = True
                    else:
                        if(last_dart_time<0):
                            last_dart_time = time.time()
                        else:
                            elapsed = time.time() - last_dart_time
                            detect = (elapsed*1000>=temporal_filter)
                    if(detect):
                        last_dart_time = -1
                        tip, res = infer_temporal_detection(temporal_model, delta)

                        if(tip is not None):
                            # for r in res:
                            playsound.playsound("sound/bow-release-bow-and-arrow-4-101936.mp3", False)
                            #scores = board.get_dart_scores(pts_cal,[[tip[0],tip[1]]])
                            scores = detector.get_dart_scores([[tip[0],tip[1]]])
                            text = f"{scores[0]}"# ({tip["conf"]:.2f})"
                            score = True
                            print(f"{int((time.time()-ts)*1000)}:{tip}=>{text}")
                            playsound.playsound(f"sound/hits/{scores[0]}.mp3", False)

                            if(show_hit_debug):
                                win = f"{int((time.time()-ts)*1000)} ms"
                                delta_dbg = cv2.cvtColor( cv2.normalize(delta,None,0,255,cv2.NORM_MINMAX, cv2.CV_8U),cv2.COLOR_GRAY2BGR)
                                delta_dbg = draw(delta_dbg,res,force_draw_all=True)
                                if(img is not None):
                                    img = draw(img.copy(),res,force_draw_all=True)
                                    cv2.imshow(win, concat_images([img, delta_dbg]))
                                else:
                                    cv2.imshow(f"{win}_temporal", delta_dbg)
                        #last = None

                    #res = infer_temporal_detection(delta, debug=1,dbg_img=img)
                else:
                    res = []
                    #cv2.waitKey()
            #cv2.imshow("diff", diff)
            if(last_diff is None or last_dart_time<0):
                last_diff = diff

            cv2.imshow("diff",diff)
            
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
    if(pts_cal is not None and crop is None):
        crop = auto_crop(pts_cal)

    
    status = "not_detected"
    if(pts_cal is not None):
        status = "opencv_detected" if opencv_detected else "detected"
        if(locked):
            status = "locked"
    img = draw(img, res, status=status, detector=detector)
    
    # res2 = infer(img,model2)
    # draw(img, res2, (0,0,255),filter=[5])
    #img_disp = img.resize((int(height*0.3),int(width*0.3)),refcheck=False)
    if(img is not None and img.shape[0]>0 and img.shape[1]>0):
        cv2.imshow('Webcam', img)
    # else:
    #     print(img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        cropped=False
        crop = None
        locked_frames = 0
        lockex = False

cap.release()
cv2.destroyAllWindows()