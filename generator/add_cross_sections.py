import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")

from board import Board, transform_points
import cv2
import numpy as np
import json

def get_cross_sections(json_path, sz=20, W=640, H=640, dbg = None):
    board = Board(json_path)

    with open(json_path,"r") as md:
        metadata = json.load(md)

    xy_cal =  np.array([p for k,p in metadata["kc"].items( )if k in ["cal1","cal2","cal3","cal4"] ] )

    M, mask = cv2.findHomography(board.board_cal_pts, xy_cal, cv2.RANSAC,5.0)
    
    pts = board.get_cross_sections_pts()

    pts_img = transform_points(pts, M)

    #img = cv2.imread(json_path.replace(".json",".jpg"))

    szh= int(sz *0.5)
    # W = img.shape[1]
    # H = img.shape[0]
    ok = []
    for i, p in enumerate(pts_img.astype(np.int32)):
        if(p[0]<szh or p[0]>=W-szh or p[1]<szh or p[1]>=H-szh):
            continue
        if(dbg):
            cv2.rectangle(dbg,(p[0]-szh,p[1]-szh),(p[0]+szh,p[1]+szh), (255,255,0), 1, cv2.LINE_AA)
        #cv2.circle(img, p, 4, (255,255,0), -1, cv2.LINE_AA)
        ok.append(i)

    if(dbg):
        cv2.imshow("tst", dbg)
        cv2.waitKey(0)
    return pts_img[ok]

if __name__ == "__main__":
    get_cross_sections("./_GENERATED/b6fda41deb88421fe2c31b38096a3fcf86ddd4060596712468d66f8dd3101681_0.json")