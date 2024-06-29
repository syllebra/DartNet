
import os
import json
import cv2
import numpy as np

def transform_points(xy, M):
    xyz = np.concatenate((np.array(xy), np.ones((len(xy), 1))), axis=-1).astype(np.float32)
    xyz_dst = np.matmul(M, xyz.T).T
    xy_dst = xyz_dst[:, :2] / xyz_dst[:, 2:]

    return xy_dst

# used to convert dart angle to board number
BOARD_DICT = {
    0: '13', 1: '4', 2: '18', 3: '1', 4: '20', 5: '5', 6: '12', 7: '9', 8: '14', 9: '11',
    10: '8', 11: '16', 12: '7', 13: '19', 14: '3', 15: '17', 16: '2', 17: '15', 18: '10', 19: '6'
}

class Board():
    def __init__(self, metadata_path, name= None) -> None:
        self.metadata_path = metadata_path
        self.image_path = None
        if(not os.path.exists(self.metadata_path)):
            self.metadata_path = None
        else:
            self.image_path = str(self.metadata_path).replace(".json",".jpg")
            if(not os.path.exists(self.image_path)):
                self.image_path = None

        self.name = name if name is not None else "Board"
        self.image_cal_pts = np.zeros((4,2)) # calib point in Board definition image frame
        self.board_cal_pts = np.zeros((4,2)) # calib point in Board frame

        metadata = {"kc": {"cal1":[0,0],"cal2":[0,0],"cal3":[0,0],"cal4":[0,0]},
                        "board":{
                "r_board": 0.2255,  # radius of full Board
                "r_double": 0.170,  # center bull to outside double wire edge, in m (BDO standaself.r_double)
                "r_treble": 0.1064,  # center bull to outside treble wire edge, in m (BDO standaself.r_double)
                "r_outer_bull": 0.0174,
                "r_inner_bull": 0.007,
                "w_double_treble": 0.01,  # wire apex to apex for double and treble
                "width": 0,
                "height": 0
            }}


        if(self.metadata_path is not None):
            if(name is None):
                self.name = os.path.basename(metadata_path).replace(".json","")
            with open(self.metadata_path,"r") as meta_file:
                metadata = json.load(meta_file)
        self.set_metadata(metadata)

    def set_metadata(self, metadata):
        if("board_file" in  metadata):
            with open(metadata["board_file"],"r") as bf:
                md = json.load(bf)
                for k,v in md["board"].items():
                    setattr(self, k, v)
        else:
            for k,v in metadata["board"].items():
                setattr(self, k, v)

        for i, p in enumerate([metadata["kc"][k] for k in ["cal1","cal2","cal3","cal4"]]):
            self.image_cal_pts[i,:] = p

        sin_rad = np.sin(np.deg2rad(9))
        cos_rad = np.cos(np.deg2rad(9))
        self.board_cal_pts = np.array([[- sin_rad, - cos_rad],[  sin_rad,   cos_rad],[- cos_rad,   sin_rad],[  cos_rad, - sin_rad]]).astype(np.float32) * self.r_double 

    def get_cross_sections_pts(self):
        pts = []
        outer_id = []
        for a in range(-9,342, 18):
            a = np.deg2rad(a)
            outer_id.append(len(pts))
            pts.extend([[np.cos(a)*d, np.sin(a)*d] for d in [self.r_double, self.r_double - self.w_double_treble,
                                self.r_treble, self.r_treble - self.w_double_treble]])#, board.r_outer_bull]])
        return np.array(pts), outer_id

    def transform_cals(self, M, image_space=False):
        xy = np.array(self.image_cal_pts if image_space else self.board_cal_pts).astype(np.float32)
        return transform_points(xy, M)

    def get_dart_scores(self, calib_pts, tips_pts, numeric=False):
        M = cv2.getPerspectiveTransform(np.array(calib_pts).astype(np.float32), np.array(self.board_cal_pts).astype(np.float32))
        tips_board = transform_points(tips_pts,M)

        angles = (np.arctan2(-tips_board[:, 1], tips_board[:, 0]) / np.pi * 180) - 9
        angles = [a + 360 if a < 0 else a for a in angles]  # map to 0-360
        distances = np.linalg.norm(tips_board[:], axis=-1)
        #print(angles)
        scores = []
        for angle, dist in zip(angles, distances):
            if dist > self.r_double:
                scores.append('0')
            elif dist <= self.r_inner_bull:
                scores.append('DB')
            elif dist <= self.r_outer_bull:
                scores.append('B')
            else:
                number = BOARD_DICT[int(angle / 18)]
                if dist <= self.r_double and dist > self.r_double - self.w_double_treble:
                    scores.append('D' + number)
                elif dist <= self.r_treble and dist > self.r_treble - self.w_double_treble:
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

    def draw(self, img, calib_pts, color=(200,180,60), cal_cols=(0,200,255)):
        for p in calib_pts:
            cv2.circle(img, p.astype(np.int32),8,cal_cols,2)

        M = cv2.getPerspectiveTransform(np.array(self.board_cal_pts).astype(np.float32), np.array(calib_pts).astype(np.float32))
        center = transform_points([[0,0]],M)[0].astype(np.int32)
        for a in range(-9,342, 18):
            a = np.deg2rad(a)
            pt = transform_points([[np.cos(a)*self.r_double, np.sin(a)*self.r_double]], M)[0]
            cv2.line(img, pt.astype(np.int32), center,color, 1, cv2.LINE_AA)

        def _draw_circle(img, center, radius_real, M, color):
            def _circle(r_real, segments = None):
                center_str = np.array([0,0])
                if(segments is None):
                    segments = max(int(r_real * np.pi * 200),0)
                a = np.arange(0,np.pi*2,np.pi*2/segments)
                pts = np.array([np.cos(a),np.sin(a)]).T
                pts = center_str + pts * r_real
                return pts

            pts = _circle(radius_real, )
            pts = transform_points(pts, M).astype(np.int32)
            pt = pts[0]
            for p in pts[1:]:
                cv2.line(img, pt, p, color, 1, cv2.LINE_AA)
                pt = p.copy()
            cv2.line(img, pt, pts[0], color, 1, cv2.LINE_AA)

        _draw_circle(img, center, self.r_double, M, color)
        _draw_circle(img, center, self.r_double - self.w_double_treble, M, color)
        _draw_circle(img, center, self.r_treble, M, color)
        _draw_circle(img, center, self.r_treble - self.w_double_treble, M, color)
        _draw_circle(img, center, self.r_outer_bull, M, color)
        _draw_circle(img, center, self.r_inner_bull, M, color)
        

if __name__ == "__main__":
    board  = Board('generator/3D/Boards/canaveral_t520.json')
    import matplotlib.pyplot as plot
    plot.scatter(board.board_cal_pts[:,1],board.board_cal_pts[:,0])
    plot.show()
    print(board.board_cal_pts)

    # for n in dir(Board):
    #     if "__" not in n:
    #         print(n, Board.__getattribute__(n))
    # exit(0)