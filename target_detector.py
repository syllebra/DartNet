import numpy as np
import cv2
from matplotlib import pyplot as plt

from time import time

    
MIN_MATCH_COUNT = 10
class TargetDetector():
    def __init__(self, board_img_path) -> None:
        self.img1 = cv2.imread(board_img_path, cv2.IMREAD_GRAYSCALE) # queryImage
        self.sift = cv2.SIFT_create(250)
        # find the keypoints and descriptors with SIFT
        self.kp1, self.des1 = self.sift.detectAndCompute(self.img1,None)
        print( len(self.kp1), len(self.des1))


    def match(self, img2):
        if(self.des1 is None):
            return None
        t = time()
        
        print(time()-t, "detect")

        t=time()
        self.kp2, des2 = self.sift.detectAndCompute(img2,None)
        print(time()-t, "detect2")
        #print(self.kp1[0], self.des1[0])
        if(des2 is None):
            return None
        if ( len(self.des1)==0 or len(des2)==0 ):
            return None
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 2)
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        


        matches = flann.knnMatch(self.des1,des2,k=2)
        print(time()-t, "match")
        t=time()
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
            print( "Not enough matches are found - {}/{}".format(len(self.good), MIN_MATCH_COUNT) )
            self.matchesMask = None
            return None

        print(time()-t, "homography")
        return M

if __name__ == "__main__":
    import math
    #detector = TargetDetector(r'C:\Users\csyllebran\Documents\PERSONNEL\words\DartNet\generator\3D\Boards\unicorn-eclipse-hd2.jpg')
    detector = TargetDetector(r'C:\Users\CSY\PERSO\Darts\ai\DartNet\generator\3D\Boards\canaveral_t520.jpg')

    #img2 = cv2.imread(r'C:\Users\csyllebran\Documents\PERSONNEL\words\DartNet\generator\3D\Boards\orig\unicorn-eclipse-hd2.jpg', cv2.IMREAD_GRAYSCALE) # trainImage
    img2 = cv2.imread(r'C:\Users\CSY\PERSO\Darts\ai\DartNet\datasets\real\vlcsnap-2024-05-12-12h55m07s339c.jpg', cv2.IMREAD_GRAYSCALE) # trainImage
    img2 = cv2.resize(img2,(640,640))
    # Initiate SIFT detector
    M = detector.match(img2)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    singlePointColor = None,
    matchesMask = detector.matchesMask, # draw only inliers
    flags = 2)

    img3 = cv2.drawMatches(detector.img1,detector.kp1,img2,detector.kp2,detector.good,None,**draw_params)

    plt.imshow(img3, 'gray'),plt.show()

    exit(0)


    import cv2 
    import numpy as np
    import os
    
    # Read image. 
    #img = cv2.imread(r'C:\Users\csyllebran\Documents\PERSONNEL\words\DartNet\generator\3D\Boards\orig\unicorn-eclipse-hd2.jpg', cv2.IMREAD_COLOR) 
    
    tests = [r'C:\Users\CSY\PERSO\Darts\ai\DartNet\datasets\real\vlcsnap-2024-05-12-12h55m07s339c.jpg',
             r'C:\Users\CSY\PERSO\Darts\ai\DartNet\datasets\real\vlcsnap-2024-05-12-12h55m07s339.png']
    
    dir = r'C:\Users\CSY\PERSO\Darts\ai\DartNet\generator\3D\Boards'
    tests.extend([os.path.join(dir,f) for f in os.listdir(dir) if ".jpg" in f])

    for path in tests:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img,(640,640))

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
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray = cv2.Canny(gray,100,200)

            # find Harris corners
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray,4,3,0.04)
            cv2.imshow("corner_canny", dst) 
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
                gray_image, maxCorners=400, qualityLevel=0.02, minDistance=40) 
            corners = np.float32(corners) 
            
            for item in corners: 
                x, y = item[0] 
                x = int(x) 
                y = int(y) 
                cv2.circle(img, (x, y), 6, (0, 255, 0), -1) 
            
            # Showing the image 
            cv2.imshow('good_features', img) 

        def get_points_4(img):
            
            imCalHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            kernel = np.ones((5, 5), np.float32) / 25
            blur = cv2.filter2D(imCalHSV, -1, kernel)
            h, s, imCal = cv2.split(blur)

            ## threshold important -> make accessible
            #ret, thresh = cv2.threshold(imCal, 140, 255, cv2.THRESH_BINARY_INV)
            ret, thresh = cv2.threshold(imCal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            ## kernel size important -> make accessible
            # very important -> removes lines outside the outer ellipse -> find ellipse
            kernel = np.ones((5, 5), np.uint8)
            thresh2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cv2.imshow("thresh2", thresh2)

        #get_points_1(img.copy())
        #get_points_2(img.copy())
        #get_points_3(img.copy())
        get_points_4(img.copy())
        key = cv2.waitKey(0)
        if(key == ord('q')):
            break