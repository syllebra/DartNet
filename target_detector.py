import numpy as np
import cv2
from matplotlib import pyplot as plt

from time import time

import random
import math
from board import Board
import statistics


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
    import os
    
    # Read image. 
    #img = cv2.imread(r'C:\Users\csyllebran\Documents\PERSONNEL\words\DartNet\generator\3D\Boards\orig\unicorn-eclipse-hd2.jpg', cv2.IMREAD_COLOR) 
    
    dir = os.path.dirname(__file__)
   
    dir = r'datasets\real\target_detector_test'
    tests = [os.path.join(dir,f) for f in os.listdir(dir) if ".jpg" in f or ".png" in f]

    for path in tests:
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

        def get_points_4(img):
            im_src = img.copy()
            def extract_double_treble_from_color(img):            
                imCalHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(20, 20))
                kernel = np.ones((5, 5), np.float32) / 25
                blur = cv2.filter2D(imCalHSV, -1, kernel)
                h, s, imCal = cv2.split(blur)
                s = clahe.apply(s)
                s[s<180]=0

                blur_sz = 1
                kernel = np.ones((blur_sz, blur_sz), np.uint8)
                s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, kernel)
                s = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)

#                cv2.imshow("sat", s)
                img[s<180] = 0
#                cv2.imshow("sat2", img)
                return s


            imCalHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            kernel = np.ones((3, 3), np.float32) / 25
            blur = cv2.filter2D(imCalHSV, -1, kernel)
            h, s, imCal = cv2.split(blur)
            ## threshold important -> make accessible
            #ret, thresh = cv2.threshold(imCal, 140, 255, cv2.THRESH_BINARY_INV)
            ret, thresh = cv2.threshold(imCal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # ## kernel size important -> make accessible
            # # very important -> removes lines outside the outer ellipse -> find ellipse
            # blur_sz = 1
            # kernel = np.ones((blur_sz, blur_sz), np.uint8)
            # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#            cv2.imshow("thresh2", thresh)

            def findEllipse(thresh, image_proc_img):

                contours, hierarchy = cv2.findContours(thresh, 1, 2)

                minThresE = 200000/4
                maxThresE = 1000000/4
                a=0
                b=0
                x=0
                y=0
                angle=0
                ## contourArea threshold important -> make accessible
                for cnt in contours:

                    cv2.drawContours(image_proc_img, [cnt], 0, (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 3)                    
                    try:  # threshold critical, change on demand?
                        if minThresE < cv2.contourArea(cnt) < maxThresE:
                            ellipse = cv2.fitEllipse(cnt)
                            cv2.ellipse(image_proc_img, ellipse, (0, 255, 0), 2)

                            x, y = ellipse[0]
                            a, b = ellipse[1]
                            angle = ellipse[2]

                            center_ellipse = (x, y)

                            a = a / 2
                            b = b / 2
                    
                            cv2.ellipse(image_proc_img, (int(x), int(y)), (int(a), int(b)), int(angle), 0.0, 360.0,
                                        (255, 0, 0))
                    # corrupted file
                    except Exception as e:
                        print ("error", e)
                        return None, image_proc_img

                return (a,b,x,y,angle), image_proc_img
            # # find enclosing ellipse
            # Ellipse, img = findEllipse(thresh, img)

            
            # return the edged image
            edged = autocanny(thresh)  # imCal            

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

                q1 = np.percentile(pts_i, 25, axis=0)
                q3 = np.percentile(pts_i, 75, axis=0)
                iqr = q3 - q1
                threshold = 1.5 * iqr
                pts = np.where((pts_i < q1 - threshold) | (pts_i > q3 + threshold))
                outliers = set(pts[0])
                # for i,pt in enumerate(pts_i):
                #     col = (0,255,255) if i in outliers else (255,255,0)
                #     cv2.drawMarker(img,(int(pt[0]),int(pt[1])),col, cv2.MARKER_CROSS,20,1)
                center = np.array([ p for i,p in enumerate(pts_i) if i not in outliers]).mean(axis = 0)
                cv2.drawMarker(img,(int(center[0]),int(center[1])),(255,255,0), cv2.MARKER_TILTED_CROSS,40,3)

                angles = []
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

                return center, filtered_angles
            #findSectorLines(edged, img, angleZone1=(80, 120), angleZone2=(30, 40))
            center, lines_a = find_center_from_sectors_lines(edged,img)
            
            s = extract_double_treble_from_color(im_src.copy())
            def segment_intersection_points(p0, p1, src, THRES=128,fuse_dist=2):
                pts = []
                vals = createLineIterator(p0.astype(np.int32),p1.astype(np.int32),src)
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

            cv2.imshow("Saturation",s)
            print(lines_a)
            for a in lines_a:
                rd = np.deg2rad(-a)
                up = np.array([np.cos(rd),np.sin(rd)]) * 400
                left = np.array([-np.sin(rd),np.cos(rd)]) * 10
                #find_points_from_close_intersections(s, center,a,2,img)
                a = center - left
                b = center + left
                c = center + left + up
                d = center - left + up
                im = four_point_transform(s, np.array([a,b,c,d]).astype(np.float32))

                #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)       
                #ret, gray = cv2.threshold(im, 127, 255, 0)         

                # linesP = cv2.HoughLinesP(im, 1, np.pi / 180, 100, None, 3, 1)
                # if linesP is None or len(linesP) == 0:
                #     continue
                # pts_i = []
            
                # for i in range(0, len(linesP)):
                #     l = linesP[i][0]
                #     if(abs(l[3]-l[1]) < 10):
                #         cv2.line(im, (l[0], l[1]), (l[2], l[3]), (255,0,255), 1, cv2.LINE_AA)
                im = np.mean(im.astype(np.float32), axis = 1)
                inside = im[0] > 128
                for i in range(len(im)):
                    is_in = im[i] > 128
                    if(inside != is_in):
                        pt = center + up * (i / len(im))
                        print(pt)
                        cv2.drawMarker(img, pt.astype(np.int32), (124,255,255),cv2.MARKER_TILTED_CROSS, 20, 1, cv2.LINE_AA)
                    inside = is_in

                # im = autocanny(im,0.01)
                # cv2.imshow(f"extracted",im)
                # key = cv2.waitKey(0)
                # if(key == ord('q')):
                #     exit()
            cv2.imshow("img",img)                
            cv2.imshow("edged",edged)

            

            def ellipse_detector_test_from_sat(im_src, dbg):
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
            
            
            def analyze_angular(center):
                angular_resol = 100
                dis_max = 400
                im = np.zeros((angular_resol,dis_max))
                s = extract_double_treble_from_color(im_src)
                # s = cv2.dilate(s, (9,9),iterations=20)
                # #s = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
                # cv2.imshow("test2",s)            
                # print(np.max(edged))
                # s[edged<128] = 0
                # cv2.imshow("test3",s)

                pts = []
                angles = []
                for i,a in enumerate(np.arange(0,360,360/angular_resol)):
                    rd = np.deg2rad(a)
                    dx = np.cos(rd) * dis_max
                    dy = np.sin(rd) * dis_max

                    ptsl, dist = segment_intersection_points(np.array(center,dtype=np.int32),np.array([center[0]+dx,center[1]+dy],dtype=np.int32),s, THRES=128)
                    angles.extend([a]*len(ptsl))

                    dec = ptsl-center
                    dist = np.linalg.norm(dec, axis=-1)
                    dist = dist.astype(np.int32)
                    ids = np.where(dist<dis_max)
                    pts.extend([[p[0],p[1]] for p in ptsl])
                    #im[i,dist[ids]] = vals[ids,2]

                for p in pts:
                    cv2.drawMarker(img,(int(p[0]),int(p[1])),(255,0,255),cv2.MARKER_TILTED_CROSS, 10, 1)

                cv2.imshow("test", edged)
                #cv2.imshow("tmp", im)
                cv2.imshow("img", img)

                print(len(pts))
                angles = np.array(angles)
                dis = np.linalg.norm(np.array(pts)[:,:2] - center, axis=-1)
                print(angles.shape, dis.shape)
                
                plt.scatter(dis,angles,s = 0.2)
                plt.show()
            #analyze_angular(center)

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