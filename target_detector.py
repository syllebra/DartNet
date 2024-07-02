import numpy as np
import cv2
from matplotlib import pyplot as plt

from time import time

import random
import math
from board import Board, transform_points
import statistics
import time


from gen_ransac import *
from tools import *

from sklearn.neighbors import NearestNeighbors


from ultralytics import YOLO

from icp import icp


SECTORS_DICT = {
    0: '6', 1: '10', 2: '15', 3: '2', 4: '17', 5: '3', 6: '19', 7: '7', 8: '16', 9: '8',
    10: '11', 11: '14', 12: '9', 13: '12', 14: '5', 15: '20', 16: '1', 17: '18', 18: '4', 19: '13'
}

box_cols = [(255,255,0),(0,215,255),(180, 105, 255),(112,255,202),(114,128,250),(255,62,191),(255,200,30),(0,255,0),(0,0,255)]

def draw(img, res, filter=None, status = "not_detected", force_draw_all = False):
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

            text = f"{confidence:.2f}"
            score = False
            # if(cls == 0):
            #     scores = board.get_dart_scores(pts_cal,[[(x1+x2)*0.5,(y1+y2)*0.5]])
            #     text = f"{scores[0]} ({confidence:.2f})"
            #     score = True

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1.2 if score else 0.3
            color = box_cols[cls]
            thickness = 4 if score else 1

            cv2.putText(img, text, org, font, fontScale, color, thickness)

        # color_tgt = (200,180,60)
        # board.draw(img, pts_cal,color_tgt)
    
    #img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_ISOLATED,value=status_colors[status])
    return img

def transform_points( xy, M):
    xyz = np.concatenate((np.array(xy), np.ones((len(xy), 1))), axis=-1).astype(np.float32)
    xyz_dst = np.matmul(M, xyz.T).T
    xy_dst = xyz_dst[:, :2] / xyz_dst[:, 2:]

    return xy_dst

def four_point_transform(image, rect):
	# # obtain a consistent order of the points and unpack them
	# # individually
	# rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst), cv2.RANSAC
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def signed_angle(a,b,c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    sign = np.dot([0,0,1],np.cross(ba,bc))
    if(sign[2]<0):
        angle=-angle
    return np.degrees(angle)


def refine_sub_pix(pts, img):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    img_gray = img if len(img.shape) == 2 else cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    tr_xy = [[c for c in p] for p in pts if (p[1]>=0 and p[1]<img_gray.shape[0]) and (p[0]>=0 and p[0]<img_gray.shape[1])]
    tr_xy = cv2.cornerSubPix(img_gray, np.array(tr_xy).astype(np.float32),(5,5),(-1,-1),criteria)
    return tr_xy

def autocanny(imCal, sigma = 0.33):
    # apply automatic Canny edge detection using the computed median
    v = np.median(imCal)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    #edged = cv2.Canny(imCal, 250, 255)
    edged = cv2.Canny(imCal, lower, upper)

    return edged

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


def rotated_ellipse_line_intersection(el, p0, p1):
    # from http://quickcalcbasic.com/ellipse%20line%20intersection.pdf

    dx = p1[0]-p0[0]
    vertical = False
    if(abs(dx)<0.000001):
        vertical = True
        m=0
        b1 = 0
    else:
        vertical = False
        m = (p1[1]-p0[1]) / dx
        b1 =  p1[1] - m * p1[0]

    alpha =  np.deg2rad(el[2])
    sina, cosa = np.sin(alpha), np.cos(alpha)
    h,v, e,f = el[1][0]*0.5, el[1][1]*0.5, el[0][0], el[0][1]
    
    m2, v2, h2 = m*m, v*v, h*h

    sina2, cosa2 = sina*sina, cosa* cosa
    tmp =  2*m*cosa*sina 

    if(vertical):
        x = p1[0] - e
        a = v2*sina2 + h2*cosa2
        b = 2*x*cosa*sina * (v2-h2)
        c = x*x*(v2*cosa2 + h2*sina2) - h2*v2
    else:
        B = b1 + m*e -f
        a = v2*(cosa2+ tmp + m2*sina2) + h2*(m2*cosa2 - tmp + sina2)
        b = 2*v2*B*(cosa*sina + m*sina2) + 2*h2*B*(m*cosa2-cosa*sina)
        c = B*B*(v2*sina2 + h2*cosa2) - h2*v2 

    interm = np.sqrt((b*b) - 4*a*c) 
    
    if(vertical):
        y1 = f + (-b - interm) / (2*a)
        y2 = f + (-b + interm) / (2*a)
        x1 = p0[0]
        x2 = p1[0]
    else:
        x1 = e + (-b - interm) / (2*a)
        x2 = e + (-b + interm) / (2*a)
        y1 = m*x1 + b1# + m*e -f
        y2 = m*x2 + b1# + m*e -f

    return [[x1, y1],[x2, y2]]

class FixedCenterCircleBoardModel(Model):
    def __init__(self, center, board, min_dist=0) -> None:
        super().__init__()
        self.N = 1
        self.center = np.array(center)
        self.radius = 0
        self.board = board
        self.radii = np.array([board.r_double, board.r_double - board.w_double_treble,
                            board.r_treble, board.r_treble - board.w_double_treble])#,
                            #board.r_outer_bull, board.r_inner_bull])
        self.min_dist = min_dist

    def build(self, points) -> None:
        self.radius= np.linalg.norm(points[0]-self.center)
        return self.radius
    
    def calc_errors(self, points):
        rads = np.linalg.norm(points-self.center, axis=-1)
        off = np.where(rads<self.min_dist)
        ratio = self.radius/self.radii[0]

        tmp= np.tile(rads,(self.radii.shape[0],1)).T
        
        min_d = np.min(np.abs(tmp-(self.radii*ratio)), axis = -1) # diff to closest radii
        min_d[off] = 100000
        return min_d
        
# pts = np.random.randint(0,640,size=(200*7,2))
# mod = FixedCenterCircleBoardModel([320,320],Board(""))
# mod.build(pts)
# mod.calc_errors(pts)
# exit(0)

MIN_MATCH_COUNT = 10
class SiftTargetDetector():
    def __init__(self, img, board = None) -> None:
        if isinstance(img, str):
            self.img1 = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            self.board = Board(img.replace(".jpg",".json")) if board is None else board
        else:
            self.img1 = img.copy()
            self.board = board
        
        self.sift = cv2.SIFT_create(500)
        # find the keypoints and descriptors with SIFT
        self.kp1, self.des1 = self.sift.detectAndCompute(self.img1,None)
        print( self.kp1[0], self.des1[0])


    def match(self, img2, compute_inverse=False):
        Mi = None
        if(self.des1 is None):
            return None, 0, Mi
        
        self.kp2, des2 = self.sift.detectAndCompute(img2,None)

        #print(self.kp1[0], self.des1[0])
        if(des2 is None):
            return None, 0, Mi
        if ( len(self.des1)==0 or len(des2)==0 ):
            return None, 0, Mi
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(self.des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        self.good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                self.good.append(m)

        if len(self.good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ self.kp1[m.queryIdx].pt for m in self.good ]).reshape(-1,1,2)
            dst_pts = np.float32([ self.kp2[m.trainIdx].pt for m in self.good ]).reshape(-1,1,2)
        
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            if(compute_inverse):
                Mi, _ =  cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
            self.matchesMask = mask.ravel().tolist()
        else:
            #print( "Not enough matches are found - {}/{}".format(len(self.good), MIN_MATCH_COUNT) )
            self.matchesMask = None
            return None, 0, Mi

        return M, min(len(self.good)/(MIN_MATCH_COUNT), 1.0), Mi

    def detect(self, img2, refine_pts=True):
        M, conf, _ = self.match(img2)
        if(M is None):
            return None, None, 0
        tr_xy = self.board.transform_cals(M,True)
        if(len(tr_xy)<4):
            return None, None, 0
                
        if(refine_pts):
            tr_xy = refine_sub_pix(tr_xy, img2)
            
        if(len(tr_xy)<4):
            return None, None, 0

        return tr_xy, M, conf

class PerspectiveBoardFit(Model):
    def __init__(self, src, dst, min_dist = 5) -> None:
        super().__init__()
        self.N = 4
        self.M = None
        #self.Mi = None
        self.src = src
        self.dst = dst

    def build(self, pairs) -> None:
        src = np.array(self.src[pairs[:,0]]).astype(np.float32)
        dst = np.array(self.dst[pairs[:,1]]).astype(np.float32)
        self.M = cv2.getPerspectiveTransform(src,dst)
        return self.M
    
    def calc_errors(self, pairs):
        proj = transform_points(self.src, self.M)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(proj)

        distances, indices = nbrs.kneighbors(self.dst)
        #print("src:", len(self.src)," dst:", len(self.dst), " pairs:", len(pairs)," indices:",len(indices)," max index:",np.max(indices)," distances:",len(distances))
        
        for i, id in enumerate(indices):
            num_corr = np.count_nonzero(indices==id)
            if(num_corr!=1):
                distances[i] = 10000
        # for nn_index in range(len(distances)):
        #     if distances[nn_index][0] < distance_threshold:        
        #print(len(distances))
        return distances

        # rads = np.linalg.norm(pairs_id-self.center, axis=-1)
        # off = np.where(rads<self.min_dist)
        # ratio = self.radius/self.radii[0]

        # tmp= np.tile(rads,(self.radii.shape[0],1)).T
        
        # min_d = np.min(np.abs(tmp-(self.radii*ratio)), axis = -1) # diff to closest radii
        # min_d[off] = 100000
        # return min_d
    
class YoloTargetDetector():
    def __init__(self, board, model_path="best_s_tip_boxes_cross_640_B.pt", auto_am_calib=False) -> None:
        print("Load general model...")
        self.model = YOLO(model_path)
        self.board = Board("dummy")
        if(board is not None):
            self.board = board if isinstance(board, Board) else Board(board.replace(".jpg",".json"))
        
        self.pts, self.outer_ids = self.board.get_cross_sections_pts()
        self.auto_am_calib = auto_am_calib
        
        self.bouter = None
        self.binner = None
        self.pts_cal = None
        # img = self.build_img(pts,(512,512))
        # self.sift = SiftTargetDetector(img, self.board)
        # print( len(self.sift.kp1), len(self.sift.des1))
        # cv2.imshow("src", self.sift.img1)

    def build_img(self, pts, size=(512,512)):
        img = np.zeros(size, np.uint8)
        for p in pts.astype(np.int32):
            cv2.circle(img,p,3,255,-1,cv2.LINE_AA)
        return img

    def infer(self, img, dbg = None):
        res = infer(img, self.model, stream=True, max_det=120, conf = 0.3, augment=False,agnostic_nms=True, vid_stride=28, verbose=False)

        res = filter_res(res)
        infered_calib = [None,None,None,None]
        infered_calib_conf = [0,0,0,0]

        cross = []
        binner = None
        bouter = None
        for r in res:
            x1, y1, x2, y2 = r["x1"],r["y1"],r["x2"],r["y2"]

            confidence = r["conf"]

            # class name
            cls = r["cls"]

            if(cls>0 and cls<5):
                if(confidence > infered_calib_conf[cls-1]):
                    infered_calib[cls-1] = np.array([(x1+x2)*0.5, (y1+y2)*0.5])
            elif(cls == 6):
                cross.append([(x1+x2)*0.5, (y1+y2)*0.5])
            elif(cls == 7):
                bouter = [x1, y1, x2, y2]
            elif(cls == 8):
                binner = [x1, y1, x2, y2]

        if(dbg is not None):
            draw_inference_boxes(dbg, res, filter=[0,5,6])

        return np.array(cross), bouter, binner, infered_calib, infered_calib_conf

    def detect(self, img, refine_pts=True, dbg = None):
        self.bouter = None
        self.binner = None
        self.pts_cal = None

        # Step 1: Infer using trained model to find board intersections
        # -------------
        corners, self.bouter, self.binner, infered_calib, infered_calib_conf = self.infer(img, dbg= dbg)

        if(dbg is not None):
            for p in corners.astype(np.int32):
                cv2.circle(dbg,p,3,(255,255,0),1,cv2.LINE_AA)

        # Step 2: Coarse initialisation using rough center/scale
        # -------------
        orig= (self.pts/ self.board.r_board)
        center = np.mean(corners, axis=0)
        # if(dbg is not None):
        #     cv2.drawMarker(dbg,center.astype(np.int32),(255,255,0),cv2.MARKER_TILTED_CROSS, 30,4, cv2.LINE_AA)
        scale = np.max(abs(corners-center),axis=0)
        pts = orig * scale *1.2 + center

        # Step 3: Iteratice closest point algorithm to find some matching pairs
        # -------------
        transformation_history, aligned_points, closest_point_pairs = icp(pts, corners,distance_threshold=15,point_pairs_threshold=10, verbose=False)
        if(dbg is not None):
            print("Pairs:",len(closest_point_pairs))
        # for p in aligned_points.astype(np.int32):
        #     cv2.drawMarker(dbg,p,(255,0,255),cv2.MARKER_TILTED_CROSS, 20)
        # for p in pts.astype(np.int32):
        #     cv2.drawMarker(dbg,p,(0,255,255),cv2.MARKER_TILTED_CROSS, 20)
            
        if(dbg is not None):
            for [a,b] in closest_point_pairs:
                # cv2.drawMarker(dbg,pts[a].astype(np.int32),(0,255,0),cv2.MARKER_TILTED_CROSS, 20,1, cv2.LINE_AA)
                # cv2.line(dbg,pts[a].astype(np.int32), corners[b].astype(np.int32),(0,0,255),1, cv2.LINE_AA)
                cv2.circle(dbg,corners[b].astype(np.int32),2,(0,255,255),-1,cv2.LINE_AA)

        # Step 4: First ransac fitting to find reasonable target pose
        # -------------
        M = ransac_fit(PerspectiveBoardFit(self.pts,corners), np.array(closest_point_pairs,np.int32), success_probabilities=0.99, outliers_ratio=0.6, inliers_thres=5)
        if(M is None):
            print("Error")
            return None, None, 0

        tr_xy = self.board.transform_cals(M,False)
        if(len(tr_xy)<4):
            return None, None, 0

        # if(dbg is not None):
        #     self.board.draw(dbg, tr_xy, color=(150,130,30),cal_cols=(0,150,200))

        # Step 5: match 4 calibrations points to refine pose
        # -------------
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(corners)
        # distances, indices = nbrs.kneighbors(tr_xy)
        # for i,d in enumerate(distances):
        #     if(d<10):
        #         tr_xy[i] = corners[indices[i]][0]
        
        # M, mask = cv2.findHomography(self.board.board_cal_pts, tr_xy, cv2.RANSAC, 5)
        
        # Step 6: Second ransac fitting using more correspondence pair for finer fit
        # -------------
        projected = transform_points(self.pts,M)
        distances, indices = nbrs.kneighbors(projected)
        valid_pairs = []
        for i,d in enumerate(distances):
            if(d<10):
                # print( corners[indices[i]].shape)
                if(dbg is not None):
                    cv2.line(dbg,projected[i].astype(np.int32), corners[indices[i]][0].astype(np.int32),(0,0,255),2, cv2.LINE_AA)
                valid_pairs.append([i,indices[i][0]])

        # if(dbg is not None):
        #     for id in self.outer_ids:
        #         cv2.circle(dbg,projected[id].astype(np.int32),6,(0,0,255),2,cv2.LINE_AA)

        valid_pairs = [p for p in valid_pairs if p[0] in self.outer_ids] # optimisation for only outer ring pairs
        valid_pairs = np.array(valid_pairs,np.int32)


        M = ransac_fit(PerspectiveBoardFit(self.pts,corners), valid_pairs, success_probabilities=0.999, outliers_ratio=0.5, inliers_thres=1.5)
        if(M is None):
            print("Error")
            return None, None, 0

        tr_xy = self.board.transform_cals(M,False)
        if(len(tr_xy)<4):
            return None, None, 0

        # Step 7: Use camera distortion for even finer result on some cameras
        # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        # -------------
        if(self.auto_am_calib):
            cameraMatrixInit = np.array([[ 2000.,    0., img.shape[1]/2.],
                                    [    0., 2000., img.shape[0]/2.],
                                    [    0.,    0.,           1.]])

            def _2d_to_3d_vec(pts):
                ret = np.zeros((pts.shape[0],3))
                ret[:,:-1] = pts
                return ret.astype(np.float32)

            distCoeffsInit = np.zeros((5,1))
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL)
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)
            model = _2d_to_3d_vec(self.pts[valid_pairs[:,0]])
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(np.array([model]).astype(np.float32), np.array([corners[valid_pairs[:,1]]]).astype(np.float32), (img.shape[1],img.shape[0]),
                                                            cameraMatrixInit, distCoeffsInit, flags=flags, criteria = criteria)

            # transform the matrix and distortion coefficients to writable lists
            data = {'reprojection_error':ret, 'camera_matrix': np.asarray(mtx).tolist(),
                    'dist_coeff': np.asarray(dist).tolist()}
            
            
            # We can take in distortion image and return the undistorted image with the help of distortion coefficient and the camera matrix.
            #dst = cv2.undistort(img, mtx, dist, None, mtx)
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (img.shape[1],img.shape[0]), 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            #imgpoints2, _ = cv2.projectPoints(model, rvecs[0], tvecs[0], mtx, dist)
            #undist = cv2.undistortImagePoints(np.array([transform_points(self.pts,M)]).astype(np.float32),mtx,dist)
            undist = cv2.undistortImagePoints(np.array([transform_points(self.board.board_cal_pts,M)]).astype(np.float32),mtx,dist)
            for p in undist.astype(np.int32):
                cv2.circle(dst, p[0], 4, (255,120,60),1, cv2.LINE_AA)
            undist = [u[0] for u in undist]
            self.board.draw(dst, undist) 
            cv2.imshow("undistort",dst)


        # Step 8: Auto rotation if one of the calib corners have been detected by model
        id_best = np.argmax(infered_calib_conf)
        if(infered_calib[id_best] is not None):
            outer_ids = [15,5,10,0] # maps calib point to outer pts id
            print("best calib for auto_rotation:", infered_calib[id_best],id_best)
            #Mi = cv2.getPerspectiveTransform(np.array(tr_xy).astype(np.float32), np.array(self.board.board_cal_pts).astype(np.float32))
            double_outer_pts = self.board.get_cross_sections_pts([self.board.r_double])[0]
            double_outer_pts_img = transform_points(double_outer_pts,M)

            distances = np.linalg.norm(double_outer_pts_img-infered_calib[id_best], axis=-1)
            print(distances)
            closest = np.argmin(distances)

            infered_calib_board = double_outer_pts[outer_ids[id_best]]
            diff_angle = signed_angle(double_outer_pts[closest],np.array([0,0]),infered_calib_board)
            rot_angle = np.around(diff_angle/18.0) * 18
            print("diff_angle:",diff_angle, " => rot_angle:",rot_angle)

            if(dbg is not None):
                cv2.circle(dbg, double_outer_pts_img[outer_ids[id_best]].astype(np.int32),20,(255,0,255),2)
                cv2.circle(dbg, double_outer_pts_img[closest].astype(np.int32),20,(255,255,255),2)
                cv2.circle(dbg, infered_calib[id_best].astype(np.int32),20,(0,100,255),2)
                

            #     cv2.circle(dbg, double_outer_pts_img[0].astype(np.int32),10,(255,0,0),-1)
            #     cv2.circle(dbg, double_outer_pts_img[5].astype(np.int32),10,(0,255,0),-1)
            #     cv2.circle(dbg, double_outer_pts_img[10].astype(np.int32),10,(0,0,255),-1)
            #     cv2.circle(dbg, double_outer_pts_img[15].astype(np.int32),10,(0,255,255),-1)

            # make rotation
            if(np.abs(rot_angle) > 0.0001 ):
                Rt = cv2.getRotationMatrix2D(center=(0,0), angle=rot_angle, scale=1)
                R = np.array([[1,0,0],[0,1,0],[0,0,1]],np.float32)
                R[:-1,:] = Rt
                tmp  = self.board.transform_cals(R)
                tr_xy = transform_points(tmp, M)


        self.pts_cal = tr_xy

        if(dbg is not None):
            self.draw_board(dbg)

        return tr_xy, M, 0
    
    def draw_board(self, img):
        detected_center = (np.array([self.bouter[0],self.bouter[1]])+np.array([self.bouter[2],self.bouter[3]])) * 0.5 if self.bouter is not None else None
        #print("detected_center:",detected_center)
        if(self.pts_cal is not None):
            self.board.draw(img, self.pts_cal, detected_center=detected_center)
        cv2.rectangle(img, (int(self.bouter[0]),int(self.bouter[1])), (int(self.bouter[2]),int(self.bouter[3])), (0,255,0), 1, cv2.LINE_AA)
        cv2.rectangle(img, (int(self.binner[0]),int(self.binner[1])), (int(self.binner[2]),int(self.binner[3])), (0,0,255), 1, cv2.LINE_AA)


    def get_dart_scores(self, tips_pts, numeric=False):
        if(self.bouter is None and self.binner is None):
            return self.board.get_dart_scores(self.pts_cal, tips_pts,numeric)

        detected_center = None
        # More precise implementation if bull/ outer bull are detected
        if(self.bouter is not None):
            detected_center = (np.array([self.bouter[0],self.bouter[1]])+np.array([self.bouter[2],self.bouter[3]])) * 0.5
        elif(self.binner is not None):
            detected_center = (np.array([self.binner[0],self.binner[1]])+np.array([self.binner[2],self.binner[3]])) * 0.5

        M = cv2.getPerspectiveTransform(np.array(self.pts_cal).astype(np.float32), np.array(self.board.board_cal_pts).astype(np.float32))
        tips_board = transform_points(tips_pts,M)
        
        # compute distances from rectified center
        detected_center_board = transform_points([detected_center],M)[0]
        distances = np.linalg.norm(tips_board[:]-detected_center_board, axis=-1)
        double_outer_pts = self.board.get_cross_sections_pts([self.board.r_double])[0]
        #return self.board.get_dart_scores(self.pts_cal, tips_pts,numeric)

        def _is_ccw(a, b, c):
            #https://stackoverflow.com/questions/37600118/test-if-point-inside-angle
            return ((a[0] - c[0])*(b[1] - c[1]) - (a[1] - c[1])*(b[0] - c[0])) > 0
        
        def _is_in_region(o,a,b,p):
            return _is_ccw(o,a,p) and not _is_ccw(o,b,p)

        def _get_sector(p):
            for i in range(20):
                a = double_outer_pts[i]
                b = double_outer_pts[(i+1)%20]
                if(_is_in_region(detected_center_board,a,b,p)):
                    return i
            return 0
        
        scores = []
        sectors = [_get_sector(p) for p in tips_board]
        for sector, dist in zip(sectors, distances):
            if dist > self.board.r_double:
                scores.append('0')
            elif dist <= self.board.r_inner_bull:
                scores.append('DB')
            elif dist <= self.board.r_outer_bull:
                scores.append('B')
            else:
                number = SECTORS_DICT[sector]
                if dist <= self.board.r_double and dist > self.board.r_double - self.board.w_double_treble:
                    scores.append('D' + number)
                elif dist <= self.board.r_treble and dist > self.board.r_treble - self.board.w_double_treble:
                    scores.append('T' + number)
                else:
                    scores.append(number)
        if numeric:
            for i, s in enumerate(scores):
                if 'B' in s:
                    if 'D' in s:
                        scores[i] = 50
                    else:
                        scores[i] = 25
                else:
                    if 'D' in s or 'T' in s:
                        scores[i] = int(s[1:])
                        scores[i] = scores[i] * 2 if 'D' in s else scores[i] * 3
                    else:
                        scores[i] = int(s)
        return scores



def createLineIterator(P1, P2, img):
    # from https://stackoverflow.com/questions/32328179/opencv-3-0-lineiterator
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int32) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int32) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer

if __name__ == "__main__":
    # detector = TargetDetector(r'C:\Users\csyllebran\Documents\PERSONNEL\words\DartNet\generator\3D\Boards\winmau_blade_6.jpg')
    # #detector = TargetDetector(r'C:\Users\CSY\PERSO\Darts\ai\DartNet\generator\3D\Boards\canaveral_t520.jpg')

    # dst = cv2.imread(r'C:\Users\csyllebran\Documents\PERSONNEL\words\DartNet\datasets\real\vlcsnap-2024-05-21-12h50m06s148.png') 
    # #img2 = cv2.imread(r'C:\Users\CSY\PERSO\Darts\ai\DartNet\datasets\real\vlcsnap-2024-05-12-12h55m07s339c.jpg', cv2.IMREAD_GRAYSCALE) # trainImage
    # img2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    
    # mult = 640/np.max(img2.shape)
    # sz = (int(img2.shape[0] * mult), int(img2.shape[1] * mult))

    # img2 = cv2.resize(img2,sz)
    # dst = cv2.resize(dst,sz)
    # tr_xy, M , conf = detector.detect(img2, True)

    # cols = [(0,215,255),(180, 105, 255),(112,255,202),(114,128,250),(255,62,191)]
    # for i, p in enumerate(tr_xy):
    #     cv2.drawMarker(dst, [int(p[0]), int(p[1])], color = cols[i], markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=1)

    # cv2.waitKey(0)
    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    # singlePointColor = None,
    # matchesMask = detector.matchesMask, # draw only inliers
    # flags = 2)

    # img3 = cv2.drawMatches(detector.img1,detector.kp1,img2,detector.kp2,detector.good,None,**draw_params)

    # plt.imshow(img3, 'gray'),plt.show()

    # exit(0)


    import cv2 
    import numpy as np
    import os, json
    
    # Read image. 
    #img = cv2.imread(r'C:\Users\csyllebran\Documents\PERSONNEL\words\DartNet\generator\3D\Boards\orig\unicorn-eclipse-hd2.jpg', cv2.IMREAD_COLOR) 
    
    dir = os.path.dirname(__file__)
   
    dir = 'datasets/real/target_detector_test'
    #dir = r'generator\_GENERATED'
    tests = [os.path.join(dir,f) for f in os.listdir(dir) if ".jpg" in f or ".png" in f]

    detector = YoloTargetDetector(None)

    debug_test = None
    # mouse callback function
    def drawfunction(event,x,y,flags,param):
        if event == cv2.EVENT_MOUSEMOVE:
            # if(debug_test is not None):
                # draw_inference_boxes(debug_test,[{"x1":x,"y1":y,"x2":x,"y2":y,"conf":1.0,"cls":0}],filter=[0],detector = detector)
                # cv2.imshow("Dbg", debug_test)
            
            scores = detector.get_dart_scores([[x,y]])
            #print(scores[0])
            cv2.displayOverlay("Dbg",scores[0])
    cv2.namedWindow('Dbg')#, cv2.WINDOW_GUI_EXPANDED)
    cv2.setMouseCallback('Dbg',drawfunction)

    for path in tests:
        board_path = path.replace(".jpg",".json") if ".jpg" in path  else path.replace(".png",".json")
        if(os.path.exists(board_path)):
            with open(board_path,"r") as f:
                data = json.load(f)
                if("board_file" in data):
                    board_path = os.path.join("generator",data["board_file"])
        else:
            tmp = os.path.basename(path).split("_")
            board_path = '_'.join(tmp[:-1])+".json"
            board_path = os.path.join("generator/3D/Boards", board_path)
        board = Board(board_path)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        mult = 640/np.min(img.shape[:2])
        sz = (int(img.shape[1] * mult), int(img.shape[0] * mult))        
        img = cv2.resize(img,sz)

        # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # # equalize the histogram of the Y channel
        # #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # alpha = 1.8  # Contrast control (1.0-3.0)
        # beta = 0  # Brightness control (0-100)

        # #img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        # clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(2, 2))
        # img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])        

        # convert the YUV image back to RGB format
        # img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        W = img.shape[1]
        H = img.shape[0]
        def get_points_1(img):
            # Convert to grayscale. 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            # Blur using 3 * 3 kernel. 
            gray_blurred = cv2.blur(gray, (3, 3)) 
            
            # Apply Hough transform on the blurred image. 
            detected_circles = cv2.HoughCircles(gray_blurred,  
                            cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                        param2 = 30, minRadius = 600, maxRadius = 3000) 
            
            # Draw circles that are detected. 
            if detected_circles is not None: 
            
                # Convert the circle parameters a, b and r to integers. 
                detected_circles = np.uint16(np.around(detected_circles)) 
            
                for pt in detected_circles[0, :]: 
                    a, b, r = pt[0], pt[1], pt[2]

                    if(abs(a-W*0.5)>W*0.2): continue
                    if(abs(b-H*0.5)>H*0.2): continue
            
                    # Draw the circumference of the circle. 
                    cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
            
                    # Draw a small circle (of radius 1) to show the center. 
                    cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
                    cv2.imshow("Detected Circle", img) 

        def get_points_2(img):
            if(len(img.shape)>2):
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            #gray = cv2.Canny(gray,100,200)

            # find Harris corners
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray,4,3,0.04)
            #cv2.imshow("corner_canny", dst) 
            #dst = cv2.dilate(dst,None)
            ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
            dst = np.uint8(dst)
            cv2.imshow("corner", dst) 
            # find centroids
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            # define the criteria to stop and refine the corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
            # print(corners)
            # # Now draw them
            # res = np.hstack((centroids,corners))
            # res = np.int8(res)
            # img[res[:,1],res[:,0]]=[0,0,255]
            # img[res[:,3],res[:,2]] = [0,255,0]
            for c in  corners:
                cv2.drawMarker(img,position = (int(c[0]),int(c[1])),color = (255,255,0),markerSize=20, markerType=cv2.MARKER_TILTED_CROSS)
            cv2.imshow("Corners Subpix", img) 
        
        def get_points_3(img):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            # Applying the function 
            corners = cv2.goodFeaturesToTrack( 
                gray_image, maxCorners=200, qualityLevel=0.005, minDistance=20) 
            corners = np.float32(corners) 
            
            for item in corners: 
                x, y = item[0] 
                x = int(x) 
                y = int(y) 
                cv2.circle(img, (x, y), 6, (0, 255, 0), -1) 
            
            # Showing the image 
            cv2.imshow('good_features', img) 

        def get_points_4(img, pyr = False):
            im_src = img.copy()
            def extract_double_treble_from_color(img,ksize=5,thres= 180):            
                imCalHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(20, 20))
                kernel = np.ones((ksize, ksize), np.float32) / (ksize*ksize)
                blur = cv2.filter2D(imCalHSV, -1, kernel)
                h, s, imCal = cv2.split(blur)
                s = np.clip(s *1.4, 0, 255).astype(np.uint8)                
                s = clahe.apply(s)
                s[s<thres]=0

                blur_sz = 1
                kernel = np.ones((blur_sz, blur_sz), np.uint8)
                s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, kernel)
                s = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)

                #cv2.imshow("sat", s)
                s[s<thres] = 0
#                cv2.imshow("sat2", img)
                return s

            #img = cv2.convertScaleAbs(img, alpha=3.0, beta=0)

            imCalHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            kernel = np.ones((3, 3), np.float32) / 25
            blur = cv2.filter2D(imCalHSV, -1, kernel)
            h, s, imCal = cv2.split(blur)
            ## threshold important -> make accessible
            ret, thresh = cv2.threshold(imCal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


            if(pyr):
                tspyr = time.time()
                shifted = cv2.pyrMeanShiftFiltering(im_src, 21, 51)
                print(f"Shifted time:{(time.time()-tspyr)*1000}")
                cv2.imshow("shifted",shifted)
                gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
                res, thresh = cv2.threshold(gray, 0, 255,
                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            shifted = im_src.copy()

            
            # return the edged image
            edged = autocanny(thresh)  # imCal            
# #             img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #             img_gray = cv2.blur(img_gray,(6,5))
# # #            cv2.imshow("thresh2", thresh)
# #             # return the edged image
#             edged = autocanny(img_gray)  # imCal
            cv2.imshow("edges",edged)

            # line intersection
            def intersectLines(pt1, pt2, ptA, ptB):

                DET_TOLERANCE = 0.00000001

                # the first line is pt1 + r*(pt2-pt1)
                # in component form:
                x1, y1 = pt1
                x2, y2 = pt2
                dx1 = x2 - x1
                dy1 = y2 - y1

                # the second line is ptA + s*(ptB-ptA)
                x, y = ptA
                xB, yB = ptB
                dx = xB - x
                dy = yB - y

                DET = (-dx1 * dy + dy1 * dx)

                if math.fabs(DET) < DET_TOLERANCE:
                    return None

                # now, the determinant should be OK
                DETinv = 1.0 / DET

                # find the scalar amount along the "self" segment
                r = DETinv * (-dy * (x - x1) + dx * (y - y1))

                # find the scalar amount along the input line
                s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

                # return the average of the two descriptions
                x = (x1 + r * dx1 + x + s * dx) / 2.0
                y = (y1 + r * dy1 + y + s * dy) / 2.0
                return (x, y)

            def find_center_from_sectors_lines(edged, dbg):
                # Probabilistic Line Transform
                linesP = cv2.HoughLinesP(edged, 0.5, np.pi / 720, 100, None, 150, 100)
                if linesP is None or len(linesP) == 0:
                    return None
                pts_i = []
            
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv2.line(dbg, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
                    for j in range(i, len(linesP)):
                        l2 = linesP[j][0]

                        pt = intersectLines((l[0], l[1]), (l[2], l[3]),(l2[0], l2[1]), (l2[2], l2[3]))
                        if(pt is not None):
                            pts_i.append(pt)

                if(len(pts_i) == 0):
                    return None

                
                outliers = filter_percentiles(pts_i)
                # for i,pt in enumerate(pts_i):
                #     col = (0,255,255) if i in outliers else (255,255,0)
                #     cv2.drawMarker(img,(int(pt[0]),int(pt[1])),col, cv2.MARKER_CROSS,20,1)
                center = np.array([ p for i,p in enumerate(pts_i) if i not in outliers]).mean(axis = 0)
                cv2.drawMarker(img,(int(center[0]),int(center[1])),(255,255,0), cv2.MARKER_TILTED_CROSS,40,3)

                angles = []
                distances = []
                # filter lines
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    def angle(a,b,c):
                        ba = a - b
                        bc = c - b
                        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                        angle = np.arccos(cosine_angle)
                        return np.degrees(angle)
                    
                    
                    p0 = np.array([l[0], l[1]])
                    p1 = np.array([l[2], l[3]])

                    ang = angle(p0, center, p1)
                    if(ang>90):
                        ang = 180 - ang
                    
                    signed0 = signed_angle(p0, center, [center[0]+100,center[1]])
                    if(signed0 <0):  signed0 = signed0 + 180
                    signed1 = signed_angle(p1, center, [center[0]+100,center[1]])
                    if(signed1 <0):  signed1 = signed1 + 180

                    if(abs(ang)<4):
                        angles.append(signed0)
                        angles.append(signed1)
                        distances.append(np.linalg.norm(p0-center))
                        distances.append(np.linalg.norm(p1-center))
                        #cv2.line(dbg, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0,255,255), 3, cv2.LINE_AA)
                    # else:
                    #     print("Thresholeded angle:",abs(ang))
   
                def filter_by_clustering(vals, threshold=2):
                    vals.sort()
                    # group close ones and pick closest to mean
                    grp = []
                    filtered = []

                    def _closest_mean(g):
                        if(len(g) == 0):
                            return None
                        if(len(g) == 1):
                            return g[0]
                        grp = np.array(g)
                        mean = grp.mean()
                        diff = abs(grp-mean)
                        id, = np.where(np.isclose(diff, np.min(diff))) # floating-point
                        return grp[id[0]]
                    # group^2 degrees apart lines
                    for a in vals:
                        if(len(grp) == 0):
                            grp.append(a)
                        else:
                            if(abs(a-grp[0])<threshold):
                                grp.append(a)
                            else:
                                filtered.append(_closest_mean(grp))
                                grp = [a]
                    if(len(grp)>0):
                        filtered.append(_closest_mean(grp))
                    return filtered
                angles.sort()
                filtered_angles = filter_by_clustering(angles,4)
                for a in filtered_angles:
                    rd = -np.deg2rad(a)
                    dx = np.cos(rd) * 200
                    dy = np.sin(rd) * 200              
                    col = (255,128,128)
                    cv2.line(dbg, (int(center[0]+dx*2), int(center[1]+dy*2)), (int(center[0]-dx*2), int(center[1]-dy*2)), col, 1, cv2.LINE_AA)
                    cv2.putText(dbg, f"{a:.2f}", (int(center[0]+dx), int(center[1]+dy)), 0,0.5, color = col, thickness=2)


                mean_radius = None
                if(len(distances)>2):
                    fd = filter_percentiles(distances)
                    mean_radius = np.array([d for i,d in enumerate(distances) if i not in fd]).mean()
                    #cv2.circle(dbg,center.astype(np.int32), int(mean_radius), (0,255,255),3, cv2.LINE_AA)

                return center, filtered_angles, mean_radius
            #findSectorLines(edged, img, angleZone1=(80, 120), angleZone2=(30, 40))
            center, lines_a, mean_radius = find_center_from_sectors_lines(edged,img)
            
            s = extract_double_treble_from_color(im_src.copy())
            def segment_intersection_points(p0, p1, src, THRES=128,fuse_dist=2):
                pts = []
                vals = createLineIterator(p0.astype(np.int32),p1.astype(np.int32),src)
                if(len(vals)==0):
                    return np.array(pts), np.array([])
                inside = vals[0,2]>THRES
                last_i=0
                for i,v in enumerate(vals):
                    new_in = v[2]>THRES
                    if(new_in != inside):
                        if(i-last_i >  fuse_dist):
                            pts.append([v[0],v[1]])
                            last_i = i
                    inside = new_in

                dec = vals[:,:2]-center
                dist = np.linalg.norm(dec, axis=-1)
                return np.array(pts), dist
            
            def find_points_from_close_intersections(src, center, a, da=2, dbg=None):
                rd = np.deg2rad(-a-da)
                dx = np.cos(rd) * 600
                dy = np.sin(rd) * 600
                ptsla, dista = segment_intersection_points(center,center+np.array([dx,dy]),s, THRES=128)
                rd = np.deg2rad(-a+da)
                dx = np.cos(rd) * 600
                dy = np.sin(rd) * 600
                ptslb, distb = segment_intersection_points(center,center+np.array([dx,dy]),s, THRES=128)

                if(dbg is not None):
                    for p in ptsla:
                        cv2.drawMarker(dbg,p.astype(np.int32),(255,128,0),cv2.MARKER_CROSS,20,1,cv2.LINE_AA)
                    for p in ptslb:
                        cv2.drawMarker(dbg,p.astype(np.int32),(0,128,255),cv2.MARKER_CROSS,20,1,cv2.LINE_AA)
           

            def sobel_threshold(img, ksize=5,thres=200):
                # print(lines_a)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0
                #gray=cv2.blur(gray,(3,3))
                #laplacian = cv2.Laplacian(gray,cv2.CV_64F,ksize=5)
                sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=ksize)
                sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=ksize)

                #cv2.imshow(f"laplacian",cv2.convertScaleAbs(laplacian)*15)
                sob = cv2.convertScaleAbs(sobelx*sobely)*15
                #sob[sob<150] = 10
                ret, sob = cv2.threshold(sob,thres,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                kernel = np.array([[0.05,0.25,0.05],[0.25,0.5,0.25],[0.05,0.25,0.05]])#np.ones((2,2), np.uint8)
                # # # for i in range(1):
                # # #     #sob=cv2.erode(sob,kernel)
                # # #     sob = cv2.morphologyEx(sob, cv2.MORPH_ERODE, kernel)      

                for i in range(1):
                    sob = cv2.morphologyEx(sob, cv2.MORPH_DILATE, kernel)
                for i in range(1):
                    sob = cv2.morphologyEx(sob, cv2.MORPH_ERODE, kernel)
                # # for i in range(3):
                # #     sob = cv2.morphologyEx(sob, cv2.MORPH_DILATE, kernel)
                #     #sob=cv2.dilate(sob,kernel)
                # # for i in range(3):
                # #     sob = cv2.morphologyEx(sob, cv2.MORPH_ERODE, kernel)      
                return sob
               



            
            #cv2.imshow(f"sobely",cv2.convertScaleAbs(sobely)*5)

            def ellipse_from_watershed(im_src, img, mean_radius=160):
                markers = np.ones((im_src.shape[0],im_src.shape[1]), np.int32)*-1
                cpt = 1

                if(mean_radius is None):
                    mean_radius = 160
                spawn_dists = np.array([0.2,0.5,0.8]) * mean_radius * 1.1
                #for a in lines_a:
                for a in range(9,180,18):

                    for dec in spawn_dists:
                        rd = np.deg2rad(-a-9)
                        up= np.array([np.cos(rd),np.sin(rd)]) * dec
                        b = center + up
                        c = center - up

                        if(b[1]>=0 and b[1]<markers.shape[0] and b[0]>=0 and b[0]<markers.shape[1]):
                            markers[int(b[1]),int(b[0])] = cpt
                            cpt+=1
                        if(c[1]>=0 and c[1]<markers.shape[0] and c[0]>=0 and c[0]<markers.shape[1]):
                            markers[int(c[1]),int(c[0])] = cpt
                            cpt+=1

                        # cv2.drawMarker(img,(int(b[0]),int(b[1])), (255,255,0), cv2.MARKER_TILTED_CROSS, 20, 2)
                        # cv2.drawMarker(img,(int(c[0]),int(c[1])), (255,255,0), cv2.MARKER_TILTED_CROSS, 20, 2)
                        #tol=160
                        #cv2.floodFill(shifted, markers,(int(b[0]),int(b[1])),cpt,0,(tol,tol,tol),  (8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY))# | 255 << 8))

                #tmp_t = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                markers = cv2.watershed(shifted,markers) 
                # 
                # #cv2.imshow("markers", markers)
                markers = markers+1
                # #print(np.max(markers))
                #markers = markers[1:-1, 1:-1]
                markers[markers==255] = 0
                
                labels = np.unique(markers)
                cpt = {l:np.sum(markers == l) for l in labels}
                maxi = np.max(list(cpt.values()))
                cpt = {k:v for k,v in cpt.items() if int(v) < maxi}
                print(cpt, maxi)


                dbg = np.zeros(img.shape,img.dtype)
                # dbg[markers == 0] = (0,0,255)

                selected_lbls = []
                for label in cpt.keys():
                    if(label == 0): continue
                    # y, x = np.nonzero(markers == label)
                    # cx = int(np.mean(x))
                    # cy = int(np.mean(y))
                    # color = (255, 255, 255)
                    #dbg[markers == label] = np.random.randint(0, 255, size=3)
                    contours, hierarchy = cv2.findContours(((markers == label)*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        epsilon = 0.008 * cv2.arcLength(cnt, True)
                        approximations = cv2.approxPolyDP(cnt, epsilon, True)
                        if(np.sum(approximations[:,:,0]<10) == 0 and np.sum(approximations[:,:,0]>markers.shape[1]-10) == 0
                        and np.sum(approximations[:,:,1]<10) == 0 and np.sum(approximations[:,:,1]>markers.shape[0]-10) == 0 ):
                            cv2.drawContours(dbg, [approximations], 0, (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 2)
                            selected_lbls.append(label)
                cv2.imshow("DBG",dbg)

                tmp_ellipse = np.zeros((im_src.shape[0],im_src.shape[1]), np.uint8)
                tmp_ellipse[markers == 0] = 255
                for l in selected_lbls:
                    tmp_ellipse[markers == l] = 255
                for _ in range(3):
                    tmp_ellipse = cv2.morphologyEx(tmp_ellipse, cv2.MORPH_ERODE, kernel)

                # for _ in range(3):
                #     tmp_ellipse = cv2.erode(tmp_ellipse,(3,3))
                #tmp_ellipse = cv2.dilate(tmp_ellipse,(5,5))
                tmp_ellipse = autocanny(tmp_ellipse)
                print(tmp_ellipse.shape)

                xn, yn = tmp_ellipse.nonzero()
                nzero = np.array([(y1,x1) for x1, y1 in zip(xn, yn)])
                el = FitEllipse_RANSAC(nzero, tmp_ellipse.astype(np.float32), graphics = False, success_probabilities=0.99, outliers_ratio=0.5,inliers_dist=1.5)
                
                if(el is not None):
                    print(el)
                    #tmp_ellipse = cv2.cvtColor(tmp_ellipse,cv2.COLOR_GRAY2BGR)
                    cv2.ellipse(img, center = np.array(el[0]).astype(np.int32), axes=(int(el[1][0] * 0.5),int(el[1][1] * 0.5)), angle = el[2],startAngle=0,endAngle =360, color= (0,255,255), thickness = 2, lineType=cv2.LINE_AA)
                    #cv2.imshow("tmp_ellipse",tmp_ellipse)
                up = np.array([np.cos(rd),np.sin(rd)]) * 400
            #ellipse_from_watershed(im_src, img, mean_radius)

            # for a in lines_a:
            #     rd = np.deg2rad(-a)
            #     up = np.array([np.cos(rd),np.sin(rd)]) * 400
            #     left = np.array([-np.sin(rd),np.cos(rd)]) * 6
            #     #find_points_from_close_intersections(s, center,a,2,img)
            #     a = center - left
            #     b = center + left
            #     c = center + left + up
            #     d = center - left + up
            #     im = four_point_transform(shifted, np.array([a,b,c,d]).astype(np.float32))
            #     #im = cv2.pyrMeanShiftFiltering(im, 21, 51)
            #     # im = cv2.blur(im, (4,3))
            #     # im = autocanny(im)
            #     #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)       
            #     #ret, gray = cv2.threshold(im, 127, 255, 0)         

            #     # linesP = cv2.HoughLinesP(im, 1, np.pi / 180, 100, None, 3, 1)
            #     # if linesP is None or len(linesP) == 0:
            #     #     continue
            #     # pts_i = []
            
            #     # for i in range(0, len(linesP)):
            #     #     l = linesP[i][0]
            #     #     if(abs(l[3]-l[1]) < 10):
            #     #         cv2.line(im, (l[0], l[1]), (l[2], l[3]), (255,0,255), 1, cv2.LINE_AA)

            #     # im = np.mean(im.astype(np.float32), axis = 1)
            #     # inside = im[0] > 128
            #     # for i in range(len(im)):
            #     #     is_in = im[i] > 128
            #     #     if(inside != is_in):
            #     #         pt = center + up * (i / len(im))
            #     #         print(pt)
            #     #         cv2.drawMarker(img, pt.astype(np.int32), (124,255,255),cv2.MARKER_TILTED_CROSS, 20, 1, cv2.LINE_AA)
            #     #     inside = is_in
            #     base_tmp = im.copy()
            #     # im = cv2.blur(im, (4,3))
            #     # im = cv2.absdiff(im[:,0:-2:2,:],im[:,1:-1:2,:])
            #     # im = np.mean(im.astype(np.float32), axis = 1)

            #     #im = cv2.absdiff(im[:,0:2,:],im[:,-3:-1,:])
            #     # im = cv2.cvtColor(im,cv2.COLOR_BGR2HLS)
            #     # im = np.sum(im.astype(np.float32)/(255*3), axis = 2)
            #     # im = np.mean(im, axis = 1)
            #     # thres = 0.3
            #     # im[im<thres]=0
            #     # im[im>=thres]=1
            #     #im = cv2.threshold((im*255).astype(np.uint8),0.4,255,cv2.THRESH_BINARY)
            #     # white = np.zeros(im.shape[:-1],im.dtype)
            #     # cv2.imshow(f"extracted_{a}",cv2.hconcat([im[:,:,0],white, im[:,:,1],white,im[:,:,2]]))
            #     # im = cv2.Sobel(im/255.0,cv2.CV_64F,0,1,ksize=5)
            #     # im = cv2.convertScaleAbs(im)*10
            #     cv2.imshow(f"extracted_{a}",im)

            #     ##im = autocanny(im,0.01)
            #     # cv2.imshow(f"extracted",im)
            #     # key = cv2.waitKey(0)
            #     # if(key == ord('q')):
            #     #     exit()


            def ellipse_detector_test_from_sat_old(im_src, dbg):
                # cv2.imshow("dt",s)

                # contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                # ellipse = cv2.fitEllipse(contours[0])
                # image = cv2.ellipse(img, ellipse, (0, 255, 0), 2)
                # cv2.imshow('Direct Ellipse Fitting from Edges', edges)
                ret, thresh = cv2.threshold(s, 127, 255, 0)
                blur_sz = 3
                kernel = np.ones((blur_sz, blur_sz), np.uint8)
                # for i in range(1):
                #     thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)
                for i in range(5):
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
                for i in range(3):
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)                

                #edges = autocanny(thresh)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                roundnesses = []
                ellipses = []
                for i in range(len(contours)):
                    cnt = contours[i]
                    epsilon = 0.1*cv2.arcLength(cnt,True)
                    if(epsilon<50):
                        continue

                    ellipse = cv2.fitEllipse(cnt)

                    # filter if center is far from detected center
                    dist = np.linalg.norm(np.array(ellipse[0])-center)
                    if(dist > 40):
                        continue

                    #cv2.drawContours(img, contours, i, (0,255,0), 1)

                    tmp = np.linalg.norm(np.array(ellipse[1]))
                    
                    roundness = 1.0 - abs((ellipse[1][0]-ellipse[1][1])/tmp)
                    roundnesses.append(roundness)
                    ellipses.append(ellipse)
                if(len(roundnesses)>0):
                    mean_rnd = statistics.median(roundnesses)
                    print(roundnesses)
                    print(mean_rnd)
                    for i, r in enumerate(roundnesses):
                        if(abs(r-mean_rnd)<0.01):
                            cv2.ellipse(dbg, ellipses[i], (0,255,255), 1, cv2.LINE_AA)
                        # else:
                        #     cv2.ellipse(img, ellipses[i], (0,50,255), 2)

                cv2.imshow("img", dbg)
                cv2.imshow("thres", thresh)
            
            
            def ellipse_detector_test_from_sat(center, angular_resol = 200, dis_max = 400):
                

                #filt = cv2.bilateralFilter(im_src, 11, 75, 75)

                s = extract_double_treble_from_color(shifted)
                cv2.imshow("sat",s)
                # s = cv2.dilate(s, (9,9),iterations=20)
                # #s = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
                # cv2.imshow("test2",s)            
                # print(np.max(edged))
                # s[edged<128] = 0
                # cv2.imshow("test3",s)

                sob = sobel_threshold(im_src)
                cv2.imshow("Sobel",sob)

                pts = []
                angles = []
                ext_ids = []
                for i,a in enumerate(np.arange(0,360,360/angular_resol)):
                    rd = np.deg2rad(a)
                    dx = np.cos(rd) * dis_max
                    dy = np.sin(rd) * dis_max

                    ptsl, dist = segment_intersection_points(np.array(center,dtype=np.int32),np.array([center[0]+dx,center[1]+dy],dtype=np.int32),s, THRES=128)

                    if(len(ptsl))>0:
                        # ptsl = np.array([ptsl[-1]])
                        # dist = np.array([dist[-1]])

                        angles.extend([a]*len(ptsl))

                        dec = ptsl-center
                        dist = np.linalg.norm(dec, axis=-1)
                        dist = dist.astype(np.int32)
                        ids = np.where(dist<dis_max)
                        pts.extend([[p[0],p[1]] for p in ptsl])

                        ext_ids.append(len(pts)-1)
                        #im[i,dist[ids]] = vals[ids,2]

                # for id,p in enumerate(pts):
                #     col = (255,255,255) if id in ext_ids else (255,0,255)
                #     cv2.drawMarker(img,(int(p[0]),int(p[1])),col,cv2.MARKER_TILTED_CROSS, 10, 1)

                #cv2.imshow("test", edged)
                #cv2.imshow("tmp", im)
                #cv2.imshow("img", img)

                # if(len(pts)>0):
                #     angles = np.array(angles)
                #     dis = np.linalg.norm(np.array(pts)[:,:2] - center, axis=-1)
                #     print(angles.shape, dis.shape)
                    
                #     plt.scatter(dis,angles,s = 0.2)
                #     plt.show()
                
                #https://math.stackexchange.com/questions/3364241/mapping-an-ellipse-to-a-circle-with-the-circles-center-offset-inside-the-ellips
                def get_ellipse_perspective_tr(e, center, sz = 512, compute_inverse = False, dbg = False, **kwargs):
                    c = np.array(el[0])
                    axes = np.array(el[1]) * 0.5

                    # draw major axis line in red
                    rmajor = np.max(axes)
                    rminor = np.min(axes)

                    angle = el[2]
                    angle  = angle -90 if angle > 90 else angle +90
                    a1 = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]) * rmajor
                    angle  = angle -90 if angle > 90 else angle +90
                    a2 = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]) * rminor

                    if(dbg):
                        if(axes[0]>0.0001 and axes[1]>0.0001):
                            cv2.ellipse(img, center = c.astype(np.int32), axes=tuple(axes.astype(np.int32)), angle = el[2], startAngle=0,endAngle =360, **kwargs)
                        cv2.line(img, (c+a1).astype(np.int32),(c-a1).astype(np.int32),color=(255,0,0))
                        cv2.line(img, (c+a2).astype(np.int32),(c-a2).astype(np.int32),color=(0,0,255))

                    ptis1 = rotated_ellipse_line_intersection(el, center, center+a1)
                    ptis2 = rotated_ellipse_line_intersection(el, center, center+a2)
                    if(dbg):
                        for pti in ptis1:
                            cv2.circle(img, np.array(pti).astype(np.int32),3, (255,0,255),-1)
                        cv2.line(img, np.array(ptis1[0]).astype(np.int32),np.array(ptis1[1]).astype(np.int32),color=(255,0,255))
                        for pti in ptis2:
                            cv2.circle(img, np.array(pti).astype(np.int32),3, (50,128,255),-1)
                        cv2.line(img, np.array(ptis2[0]).astype(np.int32),np.array(ptis2[1]).astype(np.int32),color=(50,128,255))
                        
                    src_pts = np.array([ptis1[0],ptis2[0],ptis1[1],ptis2[1]], np.float32)
                    mid = sz*0.5
                    dst_pts = np.array([[mid,0],[sz,mid],[mid,sz],[0,mid]], np.float32)
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    Mi = cv2.getPerspectiveTransform(dst_pts,src_pts) if compute_inverse else None
                    return M, Mi                    

                # sob = sobel_threshold(im_src)
                # cv2.imshow(f"sobel",sob)
                # y,x  = np.nonzero(edged)
                # pts_sob = np.array([y,x]).T
                # pts_sob = pts_sob[np.random.randint(0,len(pts_sob), 3000)]
                # print(pts_sob.shape)

                pts = np.array(pts)
                extern_pts = pts[ext_ids]
                if(len(extern_pts)<5):
                    return None
                
                for ii, p in enumerate(pts):
                    color = (0,255,255) if ii in ext_ids else (255,255,0)
                    cv2.drawMarker(img, p.astype(np.int32),color,cv2.MARKER_TILTED_CROSS, 5,1)

                el = ransac_fit(EllipseModel(), extern_pts, success_probabilities=0.99, outliers_ratio=0.5, inliers_thres=5)
                if(el is None):
                    return None

                sz = 640
                M, Mi = get_ellipse_perspective_tr(el, center = center, sz = sz, dbg = True, color= (0,255,255))
                if(M is None):
                    return None
                
                pts_u = transform_points(pts,M)
                # center_u = transform_points([center],M)[0]
                # print(center_u)
                center_u = np.array([sz*0.5,sz*0.5])
                
                unwarped = cv2.warpPerspective(im_src.copy(), M, (sz, sz))
                for ii, p in enumerate(pts_u):
                    color = (0,255,255) if ii in ext_ids else (255,255,0)
                    cv2.drawMarker(unwarped, p.astype(np.int32),color,cv2.MARKER_TILTED_CROSS, 5,1)
                
                cv2.drawMarker(unwarped, center_u.astype(np.int32),(0,255,255),cv2.MARKER_TILTED_CROSS, 20,2)
                


                
                radius = ransac_fit(FixedCenterCircleBoardModel(center_u,board, min_dist=50), pts_u, success_probabilities=0.995, outliers_ratio=0.6, inliers_thres=5)
                if(math.isnan(radius)):
                    radius = sz*0.5
                    #center_u = [sz*0.5,sz*0.5]
                print(radius)

                radii = np.array([board.r_board,
                                  board.r_double, board.r_double - board.w_double_treble,
                                  board.r_treble, board.r_treble - board.w_double_treble,
                                  board.r_outer_bull, board.r_inner_bull])
                
                for ii in range(len(radii)):
                    rad = radii[ii]*radius/radii[1]
                    cv2.circle(unwarped,center_u.astype(np.int32),int(rad),(0,255,0),1,cv2.LINE_AA)
                
                        
                cv2.imshow("unwarped", unwarped)
                # # try to detect which circle has been detected
                # board = Board("")
                # radii = np.array([board.r_board,
                #                   board.r_double, board.r_double - board.w_double_treble,
                #                   board.r_treble, board.r_treble - board.w_double_treble,
                #                   board.r_outer_bull, board.r_inner_bull])
                # err_sums = np.zeros(5)
                # fccm = FixedCenterCircleModel(center_u)
                # for ii in range(5):
                #     err = 0
                #     for rad in radii:
                #         ratio = rad/radii[ii]
                #         fccm.radius = radius * ratio
                #         #err += np.sum(np.abs(EllipseNormError(pts,elt)))/ratio
                #         norm_err = fccm.calc_errors(pts_u)
                #         inliers = np.nonzero(norm_err**2 < 3.5*3.5)[0]
                #         err += len(inliers)
                        
                #     err_sums[ii] = err
                # estimated_radius_id = np.where(err_sums == np.max(err_sums))[0][0]
                # print(err_sums, estimated_radius_id)


                # for rd in radii:
                #     ratio = rd/radii[estimated_radius_id]
                #     elt = (el[0], np.array(el[1])*ratio, el[2])
                #     color = (0,255,255) if abs(1.0-ratio)<0.0001 else (125,255,180)
                #     cv2.ellipse(img, center = np.array(elt[0]).astype(np.int32), axes=(int(elt[1][0] * 0.5),int(elt[1][1] * 0.5)), angle = elt[2],startAngle=0,endAngle =360, color= color, thickness = 2, lineType=cv2.LINE_AA)

            #board_ransac_detection(center)
            ellipse_detector_test_from_sat(center)
            cv2.imshow("image",img)


        def get_points_5(img):
            # imCalHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # kernel = np.ones((5, 5), np.float32) / 25
            # blur = cv2.filter2D(imCalHSV, -1, kernel)
            # h, s, imCal = cv2.split(blur)
            # ## threshold important -> make accessible
            # #ret, thresh = cv2.threshold(imCal, 140, 255, cv2.THRESH_BINARY_INV)
            # ret, thresh = cv2.threshold(imCal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # ## kernel size important -> make accessible
            # # very important -> removes lines outside the outer ellipse -> find ellipse
            # blur_sz = 1
            # kernel = np.ones((blur_sz, blur_sz), np.uint8)
            # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.blur(img_gray,(6,5))
#            cv2.imshow("thresh2", thresh)
            # return the edged image
            edged = autocanny(img_gray)  # imCal
            cv2.imshow("edges",edged)

            gray = np.float32(edged)
            dst = cv2.cornerHarris(gray,4,3,0.01)
            #cv2.imshow("corner_canny", dst) 
            #dst = cv2.dilate(dst,None)
            ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
            dst = np.uint8(dst)
            cv2.imshow("corner", dst)             
        #get_points_1(img.copy())
        #get_points_2(img.copy())
        #get_points_3(img.copy())
        #get_points_4(img.copy())
        #get_points_5(img.copy())
        
        @timeit
        def viola_jones_test(img):
            dbg = img.copy()
            cascade = cv2.CascadeClassifier("dartcascade/cascade.xml")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            boards = cascade.detectMultiScale(gray, 1.05)#,3)#, 5)
            for (x, y, w, h) in boards:
                cv2.rectangle(dbg, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            #circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1 , gray.shape[0]/8, None, 200, 100, 0, 0)
            gray_blurred = cv2.blur(gray, (3, 3)) 
            
            # Apply Hough transform on the blurred image. 
            detected_circles = cv2.HoughCircles(gray_blurred,  
                            cv2.HOUGH_GRADIENT, 1, gray.shape[0]/8, param1 = 200, 
                        param2 = 100, minRadius = 0, maxRadius = 0) 
            
            # Draw circles that are detected. 
            if detected_circles is not None: 
                # Convert the circle parameters a, b and r to integers. 
                detected_circles = np.uint16(np.around(detected_circles)) 
            
                for pt in detected_circles[0, :]: 
                    a, b, r = pt[0], pt[1], pt[2]

                    # if(abs(a-W*0.5)>W*0.2): continue
                    # if(abs(b-H*0.5)>H*0.2): continue
            
                    # Draw the circumference of the circle. 
                    cv2.circle(dbg, (a, b), r, (0, 255, 0), 2) 
            
                    # Draw a small circle (of radius 1) to show the center. 
                    cv2.circle(dbg, (a, b), 1, (0, 0, 255), 3) 

            #gray_eq = cv2.equalizeHist(gray)

            #canny = cv2.Canny(gray_eq, 100,300,None,3)
            canny = autocanny(gray_blurred)
            cv2.imshow("edges",canny)
            lines = cv2.HoughLinesP(canny, 1, math.pi/180.0, 50,None, 50, 10)

            def filter_lines(lines, sz, radius=3):
                tmp = np.zeros(sz,np.uint8)
                keep = []
                for i,line in enumerate(lines):
                    l = line[0]
                    if((l[0]>=0 and l[0]<sz[1] and l[1]>=0 and l[1]<sz[0] and tmp[l[1],l[0]]==0)
                       or (l[2]>=0 and l[3]<sz[1] and l[2]>=0 and l[3]<sz[0] and tmp[l[3],l[2]]==0)):
                        cv2.circle(tmp,(l[0],l[1]),radius,255,-1)
                        cv2.circle(tmp,(l[2],l[3]),radius,255,-1)
                        keep.append(i)
                return lines[keep]

            if(lines is not None):
                lines = filter_lines(lines,canny.shape,5)
                for line in lines:
                    l = line[0]
                    cv2.line(dbg, (l[0], l[1]), (l[2], l[3]), (255,255,0), 2, cv2.LINE_AA)
            
            mini = np.inf
            selected = None
            selected_lines = None
            circle_found = False
            min_sz = 40

            for bdi, bd in  enumerate(boards):
                bxmax = bd[0]+bd[2]
                bymax = bd[1]+bd[3]
                if(bd[2]<min_sz or bd[3]<min_sz):
                    continue                
                if(bd[2]>mini and bd[3]>mini):
                    continue

                if detected_circles is not None:
                    for pt in detected_circles[0, :]: 
                        a, b, r = pt[0], pt[1], pt[2]
                        if(r<min_sz):
                            continue
                        if(a>=bd[0] and a<=bxmax and b>=bd[1] and b<=bymax):
                            selected = bdi
                            mini = min(bd[2],bd[3])
                            circle_found = True
                
                if lines is not None:
                    ids_in = []
                    for li,line in enumerate(lines):
                        l = line[0]
                        if((l[0]>=bd[0] and l[0]<bxmax and l[1]>=bd[1] and l[1]<bymax) or (l[2]>=bd[0] and l[2]<bymax and l[3]>=bd[1] and l[3]<bymax)):
                            ids_in.append(li)
                        
                    if(circle_found):
                        if(bdi == selected):
                            selected_lines = ids_in.copy()
                    elif( len(ids_in)>10):
                        selected = bdi
                        mini = min(bd[2],bd[3])
                        selected_lines = ids_in.copy()

            if(selected is not None):
                (x, y, w, h) = boards[selected]
                cv2.rectangle(dbg, (x,y), (x+w,y+h), (0, 0, 255), 4)
                if(selected_lines is not None):
                    for li in selected_lines:
                        l = lines[li][0]
                        cv2.line(dbg, (l[0], l[1]), (l[2], l[3]), (255,0,255), 2, cv2.LINE_AA)

            cv2.imshow("VJ",dbg)

        #viola_jones_test(img)
        
        # def detector_distortion_test(img):
        #     img_path = board_path.replace(".json",".jpg")
        #     if(not os.path.exists(img_path)):
        #         cv2.imshow("Img", img)
        #         return
        #     detector = TargetDetector(img_path)
        #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #     #found_cals, M, conf = detector.detect(img, refine_pts=True)
        #     M, conf, Mi = detector.match(img, compute_inverse=True)
        #     if(Mi is not None):
        #         img = cv2.warpPerspective(img, Mi, (2048,2048))
        #         img = cv2.resize(img,(640,640))
        #         # tr_xy = board.transform_cals(M,True)
        #         # if(len(tr_xy)<4):
        #         #     return None, None, 0
                        
        #         # if(refine_pts):
        #         #     tr_xy = refine_sub_pix(tr_xy, img2)
                    
        #         # if(len(tr_xy)<4):
        #         #     return None, None, 0

        #         # return tr_xy, M, conf
    
        #         color_tgt = (200,180,60)
        #         tr_xy = detector.board.image_cal_pts * 640 / 2048
        #         board.draw(img, tr_xy, color_tgt)
        #     cv2.imshow("Img", img)

        # detector_distortion_test(img)
            
        def detector_yolo(img):
            detector.board = board
            debug_test = img.copy()
            found_cals, M, conf = detector.detect(img, False, debug_test)
            res = []
            if(M is not None):
                pts_cal = found_cals
                for i,p in enumerate(pts_cal):
                    v = {"x1": p[0]-10, "y1": p[1]-10,"x2": p[0]+10, "y2": p[1]+10, 'conf':0.9, "cls":i+1}
                    res.append(v)
                    print(v)

            #cv2.imshow("Img", img)
            cv2.imshow("Dbg", debug_test)
            

        for i in range(3):
            detector_yolo(img.copy())

            key = cv2.waitKey(0)
            if(key == ord('q')):
                exit(0)