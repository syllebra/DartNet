
# from https://github.com/wmcnally/deep-darts/blob/master/dataset/annotate.py
import numpy as np

def get_circle(xy):
    c = np.mean(xy[:4], axis=0)
    r = np.mean(np.linalg.norm(xy[:4] - c, axis=-1))
    return c, r

def board_radii(r_d, cfg):
    if(isinstance(cfg,dict)):
        r_t = r_d * (cfg["board"]["r_treble"] / cfg["board"]["r_double"])  # treble radius, in px
        r_ib = r_d * (cfg["board"]["r_inner_bull"] / cfg["board"]["r_double"])  # inner bull radius, in px
        r_ob = r_d * (cfg["board"]["r_outer_bull"] / cfg["board"]["r_double"]) # outer bull radius, in px
        w_dt = cfg["board"]["w_double_treble"] * (r_d / cfg["board"]["r_double"])  # width of double and treble
    else:
        r_t = r_d * (cfg.board.r_treble / cfg.board.r_double)  # treble radius, in px
        r_ib = r_d * (cfg.board.r_inner_bull / cfg.board.r_double)  # inner bull radius, in px
        r_ob = r_d * (cfg.board.r_outer_bull / cfg.board.r_double) # outer bull radius, in px
        w_dt = cfg.board.w_double_treble * (r_d / cfg.board.r_double)  # width of double and treble
    return r_t, r_ob, r_ib, w_dt

# used to convert dart angle to board number
BOARD_DICT = {
    0: '13', 1: '4', 2: '18', 3: '1', 4: '20', 5: '5', 6: '12', 7: '9', 8: '14', 9: '11',
    10: '8', 11: '16', 12: '7', 13: '19', 14: '3', 15: '17', 16: '2', 17: '15', 18: '10', 19: '6'
}

def get_dart_scores(xy_transformed, cfg, numeric=False):
    xy = xy_transformed
    valid_cal_pts = xy[:4][(xy[:4, 0] > 0) & (xy[:4, 1] > 0)]
    if xy.shape[0] <= 4 or valid_cal_pts.shape[0] < 4:  # missing calibration point
        return []

    c, r_d = get_circle(xy)
    r_t, r_ob, r_ib, w_dt = board_radii(r_d, cfg)
    xy -= c
    angles = np.arctan2(-xy[4:, 1], xy[4:, 0]) / np.pi * 180
    angles = [a + 360 if a < 0 else a for a in angles]  # map to 0-360
    distances = np.linalg.norm(xy[4:], axis=-1)
    scores = []
    for angle, dist in zip(angles, distances):
        if dist > r_d:
            scores.append('0')
        elif dist <= r_ib:
            scores.append('DB')
        elif dist <= r_ob:
            scores.append('B')
        else:
            number = BOARD_DICT[int(angle / 18)]
            if dist <= r_d and dist > r_d - w_dt:
                scores.append('D' + number)
            elif dist <= r_t and dist > r_t - w_dt:
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