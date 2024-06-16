import numpy as np
import cv2
from matplotlib import pyplot as plt

from time import time

import random
import math
from board import Board
import statistics
import time

from gen_ransac import *

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
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

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
class TargetDetector():
    def __init__(self, board_img_path) -> None:
        self.board = Board(board_img_path.replace(".jpg",".json"))
        self.img1 = cv2.imread(board_img_path, cv2.IMREAD_GRAYSCALE) # queryImage
        self.sift = cv2.SIFT_create(500)
        # find the keypoints and descriptors with SIFT
        self.kp1, self.des1 = self.sift.detectAndCompute(self.img1,None)
        # print( len(self.kp1), len(self.des1))


    def match(self, img2):
        if(self.des1 is None):
            return None, 0
        
        self.kp2, des2 = self.sift.detectAndCompute(img2,None)

        #print(self.kp1[0], self.des1[0])
        if(des2 is None):
            return None, 0
        if ( len(self.des1)==0 or len(des2)==0 ):
            return None, 0
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 2)
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
            
            self.matchesMask = mask.ravel().tolist()
        else:
            #print( "Not enough matches are found - {}/{}".format(len(self.good), MIN_MATCH_COUNT) )
            self.matchesMask = None
            return None, 0

        return M, min(len(self.good)/(MIN_MATCH_COUNT), 1.0)

    def detect(self, img2, refine_pts=True):
        M, conf = self.match(img2)
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
   
    dir = r'datasets\real\target_detector_test'
    dir = r'generator\_GENERATED'
    tests = [os.path.join(dir,f) for f in os.listdir(dir) if ".jpg" in f or ".png" in f]

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
                    #cv2.line(dbg, (l[0], l[1]), (l[2], l[3]), (255,0,255), 1, cv2.LINE_AA)
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
                    
                    def signed_angle(a,b,c):
                        ba = a - b
                        bc = c - b
                        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                        angle = np.arccos(cosine_angle)
                        sign = np.dot([0,0,1],np.cross(ba,bc))
                        if(sign[2]<0):
                            angle=-angle
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
        #get_points_3(img.copy())q
        get_points_4(img.copy())
        #get_points_5(img.copy())
        key = cv2.waitKey(0)
        if(key == ord('q')):
            break