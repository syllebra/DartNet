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
    
    pts, _ = board.get_cross_sections_pts()

    pts_img = transform_points(pts, M)

    #img = cv2.imread(json_path.replace(".json",".jpg"))

    szh= int(sz *0.5)
    # W = img.shape[1]
    # H = img.shape[0]
    ok = []
    for i, p in enumerate(pts_img.astype(np.int32)):
        if(p[0]<szh or p[0]>=W-szh or p[1]<szh or p[1]>=H-szh):
            continue
        if(dbg is not None):
            cv2.rectangle(dbg,(p[0]-szh,p[1]-szh),(p[0]+szh,p[1]+szh), (255,255,0), 1, cv2.LINE_AA)
        #cv2.circle(img, p, 4, (255,255,0), -1, cv2.LINE_AA)
        ok.append(i)

    def _reverse_bbox(radius, M):
        pts = board.get_cross_sections_pts([radius])[0]
        pts_img = transform_points(pts, M)
        mini = np.min(pts_img, axis=0)
        maxi = np.max(pts_img, axis=0)
        sz = maxi-mini
        c = (maxi+mini)*0.5
        return  [c[0],c[1],sz[0],sz[1]]
    
    bouter =_reverse_bbox(board.r_outer_bull, M)
    binner =_reverse_bbox(board.r_inner_bull, M)

    if(dbg is not None):
        cv2.rectangle(dbg,
                      np.array([bouter[0]-bouter[2]*0.5,bouter[1]-bouter[3]*0.5], np.int32)
                      ,np.array([bouter[0]+bouter[2]*0.5,bouter[1]+bouter[3]*0.5], np.int32), (0,255,0), 1, cv2.LINE_AA)
        cv2.rectangle(dbg,
                      np.array([binner[0]-binner[2]*0.5,binner[1]-binner[3]*0.5], np.int32)
                      ,np.array([binner[0]+binner[2]*0.5,binner[1]+binner[3]*0.5], np.int32), (0,0,255), 1, cv2.LINE_AA)

    if(dbg is not None):
        cv2.imshow("tst", dbg)
        cv2.waitKey(0)
    return pts_img[ok], bouter, binner

if __name__ == "__main__":
    path = "./_GENERATED/e2b6517a7e7559d6bf49e58eb26ec87eff47667a40b7da0e22b1efa8d08cf5ae_0.json"
    img = cv2.imread(path.replace(".json", ".jpg"))
    get_cross_sections(path, dbg = img)