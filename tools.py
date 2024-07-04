
import math
from ultralytics import YOLO
import numpy as np
import time
from functools import wraps
import cv2

from generator.annotate import seg_intersect

# object classes
classNames = ["tip", "cal1", "cal2", "cal3", "cal4", "dart", "cross", "D-Bull", "Bull"]
MAX_CLASSES = len(classNames)

# Colors definition
box_cols = [(255,255,0),(0,215,255),(180, 105, 255),(112,255,202),(114,128,250),(255,62,191),(255,200,30),(0,255,0),(0,0,255)]
status_colors = {"not_detected": (0,0,255), "detected": (0,255,0), "opencv_detected": (0,160,75), "locked":(160,160)}


# Timing function
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

# Misc. utils functions
# ---------------------

def auto_crop(pts_cal, width, height):
    center = seg_intersect(pts_cal[0],pts_cal[1],pts_cal[2],pts_cal[3])
    x1 = int(center[0]-12)
    y1 = int(center[1]-12)
    x2 = int(center[0]+12)
    y2 = int(center[1]+12)
    #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
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
    return crop

def crop_img(img, crop, model_train_size = 640, clahe = None):
    img = (img[crop[1]:crop[3],crop[0]:crop[2],:])
    ratio = model_train_size/np.max(img.shape)
    img= cv2.resize(img,(int(img.shape[1]*ratio),int(img.shape[0]*ratio)),interpolation=cv2.INTER_LANCZOS4)
    #img = clahe.apply(img)
    if(clahe is not None):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img


def filter_percentiles(data, minim=25, maxim=75, axis=0):
    try:
        q1 = np.percentile(data, minim, axis=axis)
        q3 = np.percentile(data, maxim, axis=axis)
        iqr = q3 - q1
        threshold = 1.5 * iqr
        pts = np.where((data < q1 - threshold) | (data > q3 + threshold))
        outliers = set(pts[0])
        return outliers
    except:
        pass
    return set()


# Drawing tools section
# ---------------------

def concat_images(image_set, how='horizontal'):
    return cv2.hconcat(image_set)


def draw_inference_boxes(img, res, filter=[6], detector=None):
    for box in res:
        # confidence
        confidence = box["conf"]
        #print("Confidence --->",confidence)

        # class name
        cls = box["cls"]

        if(filter is not None and cls in filter):
            continue

        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"]) # convert to int values

        # put box in cam
        cv2.rectangle(img, (x1, y1), (x2, y2), box_cols[cls], 1)

        text = f"{classNames[cls]} ({confidence:.2f})"
        score = False
        if(cls == 0 and detector is not None):
            #scores = board.get_dart_scores(pts_cal,[[(x1+x2)*0.5,(y1+y2)*0.5]])
            scores = detector.get_dart_scores([[(x1+x2)*0.5,(y1+y2)*0.5]])
            text = f"{scores[0]} ({confidence:.2f})"
            score = True

        # object details
        org = [x1, y1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1.2 if score else 0.6
        color = box_cols[cls]
        thickness = 4 if score else 1

        cv2.putText(img, text, org, font, fontScale, color, thickness)

    if(detector is not None):
        detector.draw_board(img)
    return img

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    # https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

# Inference section
# -----------------

def infer(img, mod, **inference_params):
    ''' Infer model and return as list of dict containing detected boxes '''
    if(mod is None):
        return None
    
    if(len(inference_params) == 0):
        inference_params = { "stream": True, "max_det": 25, "conf": 0.3, "augment":False, "agnostic_nms":True, "vid_stride":28,"verbose":False }

    results = mod(img, **inference_params)

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

def infer_temporal_detection(model, delta, min_conf = 0.4):
    delta = cv2.cvtColor((delta*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
   
    res = infer(delta, model)
    res = [r for r in res if r['conf']>min_conf]
    cpt_darts = 0
    cpt_tips = 0
    res = filter_res(res,filter_classes=[0,1])
    tip = None
    dart = None
    for i in range(len(res)):
        if(res[i]["cls"]==1):
            cpt_darts += 1
            res[i]["cls"] = 5
            dart = res[i]
        elif(res[i]["cls"]==0):
            cpt_tips += 1
            tip = [(float(res[i]["x1"])+float(res[i]["x2"]))*0.5,(float(res[i]["y1"])+float(res[i]["y2"]))*0.5]
    return tip, res

def find_cal_pts(res, confidence_min = 0.5):
    ''' Analyze inference results and return found calibration points if all have been found '''
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


def filter_res(res, filter_classes = [1,2,3,4]):
    ''' Analyze inference results and return and keep only best confidence of a given class '''

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

